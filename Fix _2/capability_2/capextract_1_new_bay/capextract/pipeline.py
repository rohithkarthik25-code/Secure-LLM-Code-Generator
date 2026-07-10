"""
CapabilityExtractionPipeline — the main entry point.

Flow:
  1. User prompt → Intent analysis (LLM call)
  2. Intent IR constructed
  3. LLM streams code generation
  4. Each chunk → incremental AST parse → new primitive signals
  5. Every CHECKPOINT_TOKENS tokens → incremental graph update
  6. On generation complete → final full graph build
  7. Return CapabilityGraphIR

Usage:
    from capextract.pipeline import CapabilityExtractionPipeline
    from capextract.llm.adapters import get_adapter

    adapter = get_adapter("claude")   # or "openai", "ollama", "mock"
    pipeline = CapabilityExtractionPipeline(adapter)
    graph_ir = pipeline.run("Write a Python script that reads a CSV and trains an ML model")
    print(json.dumps(graph_ir.to_dict(), indent=2))
"""

from __future__ import annotations
import time
import threading
from typing import Callable, Optional

from capextract.core.models import (
    CapabilityGraphIR, IntentIR, NodeType
)
from capextract.core.parser import (
    IncrementalParser, detect_language, signals_to_cap_nodes
)
import sys
import os

intent_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "srija")
sys.path.insert(0, intent_path)
try:
    from prompt_input import ingest_prompt
    from prompt_intent import extract_prompt_intent
    from semantic_ir import build_semantic_representation
except ImportError as e:
    print(f"[ERROR] Could not import upstream 'srija' pipeline modules: {e}")
    print(f"Make sure 'srija' directory is located at {intent_path}")
    raise

from capextract.core.intent import adapt_intent_payload
from capextract.graph.builder import build_graph, incremental_update
from capextract.llm.adapters import LLMAdapter
from capextract.scoring.combined import DynamicScorer
from capextract.core.self_discover import SelfDiscoverEngine
from capextract.core.data_flow import DataFlowTracker
from capextract.core.models import PrimitiveCap


# How often (in chars of new code) to trigger an incremental graph update
CHECKPOINT_CHARS = 200


class CapabilityExtractionPipeline:
    """
    Real-time capability extraction pipeline.

    Designed to run inside an LLM generation loop:
    - Intent is extracted before code generation starts
    - AST is updated incrementally as tokens arrive
    - Graph IR is updated at checkpoints (async-safe)
    - Final graph is built when generation completes
    """

    def __init__(
        self,
        adapter: LLMAdapter,
        on_risk_detected: Optional[Callable[[str, CapabilityGraphIR], None]] = None,
        verbose: bool = False,
        bayesian_beta: float = 0.6,
        decay_lambda: float = 0.995,
        track_history: bool = False,
    ):
        """
        adapter:           LLM adapter (Claude/OpenAI/Ollama/Mock)
        on_risk_detected:  callback(risk_label, current_graph) fired immediately
                           when a RISK node appears during streaming
        verbose:           print progress to stdout
        bayesian_beta:     weight for Bayesian posterior in combined score (0-1)
        decay_lambda:      exponential decay factor for evidence accumulation
        track_history:     if True, store confidence trace per capability
        """
        self.adapter          = adapter
        self.on_risk_detected = on_risk_detected
        self.verbose          = verbose
        self._bayesian_beta   = bayesian_beta
        self._decay_lambda    = decay_lambda
        self._track_history   = track_history

    # ─────────────────────────────────────────────────────────────
    # Main entry point
    # ─────────────────────────────────────────────────────────────

    def run(self, prompt: str, intent_payload: dict = None) -> CapabilityGraphIR:
        """
        Full pipeline: prompt → CapabilityGraphIR.
        Blocking call; returns once code generation is complete.
        """
        t0 = time.time()
        self._log(f"[pipeline] starting for prompt: {prompt[:80]}...")

        # ── Step 1: Extract intent ───────────────────────────────
        if intent_payload:
            self._log("[pipeline] using pre-computed intent payload...")
            semantic_payload = intent_payload
        else:
            self._log("[pipeline] extracting intent via upstream 'Fix' pipeline...")
            # Run Stage 1
            s1 = ingest_prompt(prompt)
            # Run Stage 2
            s2 = extract_prompt_intent(s1)
            # Run Stage 3
            semantic_payload = build_semantic_representation(s2, save_visualization=False, save_json=False)
        
        # Adapt to IntentIR
        intent = adapt_intent_payload(semantic_payload)
        
        self._log(f"[pipeline] intent: {intent.goal}")
        self._log(f"[pipeline] expected caps: {intent.expected_caps}")
        self._log(f"[pipeline] scope constraints: {intent.scope_constraints}")

        # ── Step 1b: Initialize DynamicScorer from intent ──────
        scorer = DynamicScorer(
            beta=self._bayesian_beta,
            decay_lambda=self._decay_lambda,
        )
        scorer.init(intent, track_history=self._track_history)
        self._log("[pipeline] dynamic scorer initialized (Bayesian + Evidence)")

        # ── Step 1c: Generate Dynamic Threat Model (SELF-DISCOVER)
        self._log("[pipeline] generating dynamic threat model (SELF-DISCOVER)...")
        discover_engine = SelfDiscoverEngine()
        threat_model = discover_engine.generate_threat_model(intent, self.adapter)
        self._log(f"[pipeline] dynamic threat model generated: {list(threat_model.keys())}")

        # ── Step 2: Detect or use declared language ──────────────
        language = intent.detected_language or "python"
        if language == "unknown":
            language = "python"

        # ── Step 3: Set up incremental state ─────────────────────
        parser         = IncrementalParser(language)
        data_flow      = DataFlowTracker()
        all_prims      = []       # accumulated CapNodes
        code_buffer    = []       # streaming code chunks
        chars_since_ck = [0]      # mutable counter for checkpoint
        graph_ref      = [None]   # latest graph, updated at checkpoints
        lock           = threading.Lock()

        def on_chunk(chunk: str):
            code_buffer.append(chunk)
            chars_since_ck[0] += len(chunk)

            # Incremental parse on every chunk (fast path)
            current_code = "".join(code_buffer)
            new_signals = parser.feed(current_code)
            new_nodes = signals_to_cap_nodes(new_signals, language)

            with lock:
                all_prims.extend(new_nodes)

                # Feed signals to dynamic scorer and data flow tracker
                for sig in new_signals:
                    scorer.on_signal(sig)
                    try:
                        pc = sig.capability
                        target_var = sig.metadata.get("target_var")
                        source_var = sig.metadata.get("source_var")
                        if target_var and not source_var:
                            data_flow.bind(target_var, pc, sig.source_line)
                        elif target_var and source_var:
                            data_flow.propagate(source_var, target_var)
                        elif source_var and not target_var:
                            data_flow.consume(source_var, pc, sig.source_line)
                    except ValueError:
                        pass

                # Apply decay step on every chunk
                scorer.on_step()

                # Log high-risk primitives immediately
                for n in new_nodes:
                    if n.metadata.get("high_risk"):
                        self._log(f"  [!] HIGH-RISK primitive: {n.label} at line {n.source_line}")

                # Checkpoint: incremental graph update
                if chars_since_ck[0] >= CHECKPOINT_CHARS:
                    chars_since_ck[0] = 0
                    if graph_ref[0] is None:
                        graph_ref[0] = build_graph(all_prims, intent, language, current_code, scorer=scorer, data_flow=data_flow)
                    else:
                        incremental_update(graph_ref[0], new_nodes)
                        graph_ref[0].generated_code = current_code

                    # Fire risk callbacks
                    if self.on_risk_detected and graph_ref[0].risk_nodes:
                        for risk in graph_ref[0].risk_nodes:
                            self.on_risk_detected(risk, graph_ref[0])

        def on_done(full_code: str):
            self._log(f"[pipeline] generation complete ({len(full_code)} chars)")

        # ── Step 4: Stream code generation ───────────────────────
        self._log(f"[pipeline] streaming code generation (language: {language})...")
        self.adapter.stream_code(
            prompt=f"Write {language} code: {prompt}\n\nReturn only code, no explanation.",
            on_chunk=on_chunk,
            on_done=on_done,
        )

        # ── Step 5: Final graph build ─────────────────────────────
        final_code = "".join(code_buffer)
        self._log("[pipeline] building final capability graph...")

        # Re-parse the full code for completeness
        final_parser = IncrementalParser(language)
        final_signals = final_parser.feed(final_code)
        final_nodes   = signals_to_cap_nodes(final_signals, language)

        # Feed final signals to scorer and data flow
        for sig in final_signals:
            scorer.on_signal(sig)
            try:
                pc = sig.capability
                target_var = sig.metadata.get("target_var")
                source_var = sig.metadata.get("source_var")
                if target_var and not source_var:
                    data_flow.bind(target_var, pc, sig.source_line)
                elif target_var and source_var:
                    data_flow.propagate(source_var, target_var)
                elif source_var and not target_var:
                    data_flow.consume(source_var, pc, sig.source_line)
            except ValueError:
                pass

        graph = build_graph(final_nodes, intent, language, final_code, scorer=scorer, data_flow=data_flow)

        elapsed = time.time() - t0
        self._log(f"[pipeline] done in {elapsed:.2f}s")
        self._log(f"[pipeline] primitive caps:  {graph.primitive_caps}")
        self._log(f"[pipeline] functional caps: {graph.functional_caps}")
        self._log(f"[pipeline] risk flags:      {graph.risk_nodes}")

        # Log top dynamic scores
        ranked = scorer.get_ranked(threshold=0.20)
        if ranked:
            self._log("[pipeline] dynamic confidence ranking:")
            for sc in ranked[:10]:
                self._log(f"  {sc.label}: {sc.combined_score:.3f} "
                          f"(bayes={sc.bayesian_score:.3f}, ev={sc.evidence_score:.3f})")

        return graph

    # ─────────────────────────────────────────────────────────────
    # Convenience: run on already-generated code (no LLM needed)
    # ─────────────────────────────────────────────────────────────

    def analyse_code(self, code: str, prompt: str = "") -> CapabilityGraphIR:
        """
        Analyse existing code directly (no LLM generation step).
        Useful for testing or post-hoc analysis comparison.
        """
        language = detect_language(code)
        if prompt:
            s1 = ingest_prompt(prompt)
            s2 = extract_prompt_intent(s1)
            semantic_payload = build_semantic_representation(s2, save_visualization=False, save_json=False)
            intent = adapt_intent_payload(semantic_payload)
        else:
            intent = IntentIR(
                raw_prompt=code[:200],
                goal="analyse existing code",
                detected_language=language,
            )

        # Initialize dynamic scorer for static analysis too
        scorer = DynamicScorer(
            beta=self._bayesian_beta,
            decay_lambda=self._decay_lambda,
        )
        scorer.init(intent, track_history=self._track_history)

        signals = IncrementalParser(language).feed(code)
        nodes   = signals_to_cap_nodes(signals, language)

        # Feed signals to scorer
        for sig in signals:
            scorer.on_signal(sig)

        return build_graph(nodes, intent, language, code, scorer=scorer)

    def _log(self, msg: str):
        if self.verbose:
            print(msg)
