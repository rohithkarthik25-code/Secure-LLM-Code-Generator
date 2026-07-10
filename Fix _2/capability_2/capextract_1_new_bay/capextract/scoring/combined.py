"""
Combined dynamic scorer — orchestrates Bayesian updating and
streaming evidence accumulation into a single confidence score.

    Confidence_t(C) = β × P_t(C) + (1 - β) × S_t(C)

Where:
    P_t(C) = Bayesian posterior (intent-aware)
    S_t(C) = Evidence accumulation score (code-grounded)
    β      = Combination weight (default 0.6)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from capextract.core.models import IntentIR, PrimitiveCap, FunctionalCap
from capextract.core.parser import ParsedSignal
from capextract.scoring.matrix_engine import MatrixEngine
from capextract.scoring.evidence import EvidenceAccumulator
FUNCTIONAL_CONF_THRESHOLD = 0.50

# Default combination weight: Bayesian vs Evidence
DEFAULT_BETA = 0.6

# Minimum combined score to include in ranked output
RANKED_THRESHOLD = 0.20

@dataclass
class ScoredCapability:
    """A capability with its dynamic confidence breakdown."""
    label:           str
    combined_score:  float
    bayesian_score:  float
    evidence_score:  float
    cap_type:        str = "primitive"   # "primitive" or "functional"

    def __repr__(self):
        return (f"ScoredCap({self.label}: combined={self.combined_score:.3f} "
                f"[bayes={self.bayesian_score:.3f}, ev={self.evidence_score:.3f}])")

class DynamicScorer:
    """
    Combines MatrixEngine and EvidenceAccumulator into a single
    dynamic scoring engine for the streaming pipeline.
    """

    def __init__(
        self,
        beta: float = DEFAULT_BETA,
        decay_lambda: float = 0.995,
    ):
        self._beta = beta
        self._matrix = MatrixEngine()
        self._evidence = EvidenceAccumulator(decay_lambda=decay_lambda)
        self._history: dict[str, list[float]] = {}   # cap -> score trace
        self._initialized = False
        self._track_history = False

    def init(self, intent: IntentIR, track_history: bool = False) -> None:
        """
        Initialize both sub-scorers from intent.
        Call once before streaming begins.
        """
        self._matrix.init_priors(intent)
        self._matrix.init_edge_priors([])
        self._track_history = track_history
        self._initialized = True

    def on_signal(self, signal: ParsedSignal) -> None:
        """
        Process a new ParsedSignal from the code parser.
        Updates both matrix posteriors and evidence accumulation.
        """
        if not self._initialized:
            return
        self._matrix.update(signal)
        self._evidence.accumulate(signal)

    def on_step(self) -> None:
        """
        Called on each token step (chunk arrival).
        Applies exponential decay to evidence scores.
        Optionally records history snapshot.
        """
        if not self._initialized:
            return
        self._evidence.decay_step()

        if self._track_history:
            self._snapshot_history()

    def get_confidence(self, cap_label: str) -> float:
        """
        Return the combined confidence for a capability.

            Confidence = β × P(C) + (1 - β) × S(C)
        """
        p = self._matrix.get_posterior(cap_label)
        s = self._evidence.get_score(cap_label)
        return self._beta * p + (1.0 - self._beta) * s

    def get_edge_confidence(self, edge_key: str) -> float:
        """
        Return the combined confidence for an edge.
            Confidence = β × P(E) + (1 - β) × S(E)
        """
        p = self._matrix.get_edge_posterior(edge_key)
        s = self._evidence.get_edge_score(edge_key)
        return self._beta * p + (1.0 - self._beta) * s

    def get_scored(self, cap_label: str) -> ScoredCapability:
        """Return full scoring breakdown for a capability."""
        p = self._matrix.get_posterior(cap_label)
        s = self._evidence.get_score(cap_label)
        combined = self._beta * p + (1.0 - self._beta) * s

        # Determine type
        cap_type = "primitive"
        try:
            FunctionalCap(cap_label)
            cap_type = "functional"
        except ValueError:
            pass

        return ScoredCapability(
            label=cap_label,
            combined_score=combined,
            bayesian_score=p,
            evidence_score=s,
            cap_type=cap_type,
        )

    def get_ranked(self, threshold: float = RANKED_THRESHOLD) -> list[ScoredCapability]:
        """
        Return all capabilities above threshold, sorted by combined score descending.
        """
        all_labels: set[str] = set()
        all_labels.update(self._matrix.get_all_posteriors().keys())
        all_labels.update(self._evidence.get_all_scores().keys())

        scored = []
        for label in all_labels:
            sc = self.get_scored(label)
            if sc.combined_score >= threshold:
                scored.append(sc)

        scored.sort(key=lambda x: x.combined_score, reverse=True)
        return scored

    def get_functional_confidences(self) -> dict[str, float]:
        """
        Return combined confidences for functional capabilities only.
        Used by the graph builder to set node confidence values.
        """
        result = {}
        for fc in FunctionalCap:
            if fc == FunctionalCap.UNCLASSIFIED:
                continue
            conf = self.get_confidence(fc.value)
            if conf >= FUNCTIONAL_CONF_THRESHOLD:
                result[fc.value] = conf
        return result

    def get_history(self, cap_label: str) -> list[float]:
        """Return the confidence trace over time for a capability."""
        return self._history.get(cap_label, [])

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    # ── Private helpers ──────────────────────────────────────────

    def _snapshot_history(self) -> None:
        """Record current confidence values for all active capabilities."""
        all_labels = set(self._matrix.get_all_posteriors().keys())
        all_labels.update(self._evidence.get_all_scores().keys())
        for label in all_labels:
            conf = self.get_confidence(label)
            if label not in self._history:
                self._history[label] = []
            self._history[label].append(round(conf, 4))
