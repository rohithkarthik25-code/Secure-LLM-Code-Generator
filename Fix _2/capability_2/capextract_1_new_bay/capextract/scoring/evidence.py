"""
Streaming evidence accumulator — exponential decay weighted scoring.

Maintains a running evidence score per capability that decays over time,
so recent code signals carry more influence than older ones.

Efficient O(1) per step via incremental computation:
    E_t(C) = λ × E_{t-1}(C) + w_t × s_t(C)

Normalized to [0, 1] via sigmoid:
    S_t(C) = 1 / (1 + exp(-k × (E_t - θ)))
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional

from capextract.core.models import PrimitiveCap, FunctionalCap
from capextract.core.parser import ParsedSignal

# Decay and normalization parameters
DEFAULT_LAMBDA   = 0.995    # slow decay — evidence halves every ~139 steps
SIGMOID_K        = 5.0      # steepness of the sigmoid
SIGMOID_THETA    = 0.50     # midpoint threshold

# Signal weights by type
_SIGNAL_WEIGHTS: dict[str, float] = {
    "import":   0.90,
    "call":     0.95,
    "indirect": 0.70,
    "default":  0.80,
}

# Lightweight motif mapping for evidence accumulation
_EVIDENCE_MOTIFS = {
    FunctionalCap.DATA_EXFILTRATION.value: {PrimitiveCap.FILE_READ.value, PrimitiveCap.HTTP_REQUEST.value, PrimitiveCap.SOCKET_OPEN.value, PrimitiveCap.DATA_EXFIL.value},
    FunctionalCap.WEB_SCRAPING.value: {PrimitiveCap.HTTP_REQUEST.value, PrimitiveCap.FILE_WRITE.value, PrimitiveCap.STRING_OP.value},
    FunctionalCap.SYSTEM_AUTOMATION.value: {PrimitiveCap.PROCESS_EXEC.value, PrimitiveCap.SHELL_INVOKE.value, PrimitiveCap.ENV_READ.value, PrimitiveCap.FILE_WRITE.value},
    FunctionalCap.BASIC_COMPUTATION.value: {PrimitiveCap.MATH_OP.value, PrimitiveCap.STRING_OP.value},
    FunctionalCap.DATABASE_OPS.value: {PrimitiveCap.DB_CONNECT.value, PrimitiveCap.DB_READ.value, PrimitiveCap.DB_WRITE.value}
}

class EvidenceAccumulator:
    """
    Maintains exponential-decay-weighted evidence scores for all capabilities.
    """

    def __init__(self, decay_lambda: float = DEFAULT_LAMBDA):
        self._lambda = decay_lambda
        self._evidence: dict[str, float] = {}   # cap_label -> E_t
        self._edge_evidence: dict[str, float] = {} # edge_key -> E_t
        self._step_count: int = 0

    def accumulate(self, signal: ParsedSignal) -> None:
        """
        Add evidence for a signal's capability.
        Also propagates evidence to functional capabilities.
        """
        cap_label = signal.capability.value
        weight = self._signal_weight(signal)

        # Update primitive evidence
        current = self._evidence.get(cap_label, 0.0)
        self._evidence[cap_label] = current + weight

        # Propagate to functional caps via motifs
        for func_label, prims in _EVIDENCE_MOTIFS.items():
            if cap_label in prims:
                func_weight = weight * 0.5
                current_f = self._evidence.get(func_label, 0.0)
                self._evidence[func_label] = current_f + func_weight

    def decay_step(self) -> None:
        """
        Apply one step of exponential decay to all evidence scores.
        Call this on every token step (chunk arrival).
        """
        self._step_count += 1
        for cap in self._evidence:
            self._evidence[cap] *= self._lambda
        for edge_key in self._edge_evidence:
            self._edge_evidence[edge_key] *= self._lambda

    def get_score(self, cap_label: str) -> float:
        """
        Return the sigmoid-normalized evidence score S_t(C) in [0, 1].
        """
        raw = self._evidence.get(cap_label, 0.0)
        return self._sigmoid(raw)

    def get_raw_evidence(self, cap_label: str) -> float:
        """Return the raw (un-normalized) evidence score E_t(C)."""
        return self._evidence.get(cap_label, 0.0)

    def accumulate_edge(self, edge_key: str, weight: float) -> None:
        """Add evidence for an edge (e.g. data flow)."""
        current = self._edge_evidence.get(edge_key, 0.0)
        self._edge_evidence[edge_key] = current + weight

    def get_edge_score(self, edge_key: str) -> float:
        """Return the sigmoid-normalized evidence score S_t(E) in [0, 1]."""
        raw = self._edge_evidence.get(edge_key, 0.0)
        return self._sigmoid(raw)

    def get_all_scores(self) -> dict[str, float]:
        """Return sigmoid-normalized scores for all capabilities with evidence."""
        return {cap: self._sigmoid(e) for cap, e in self._evidence.items()}

    @property
    def step_count(self) -> int:
        """Number of decay steps applied so far."""
        return self._step_count

    # ── Private helpers ──────────────────────────────────────────

    def _sigmoid(self, x: float) -> float:
        """Sigmoid normalization: 1 / (1 + exp(-k * (x - θ)))"""
        exponent = -SIGMOID_K * (x - SIGMOID_THETA)
        # Clamp to avoid overflow
        exponent = max(-500.0, min(500.0, exponent))
        return 1.0 / (1.0 + math.exp(exponent))

    def _signal_weight(self, signal: ParsedSignal) -> float:
        """Determine weight based on signal trigger type."""
        trigger = signal.trigger
        if trigger.startswith("call:"):
            return _SIGNAL_WEIGHTS["call"]
        elif trigger.startswith("import:"):
            return _SIGNAL_WEIGHTS["import"]
        else:
            return _SIGNAL_WEIGHTS["default"]
