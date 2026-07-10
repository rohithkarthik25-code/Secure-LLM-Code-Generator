"""
Test suite for Bayesian updating + Evidence accumulation scoring.

Tests cover:
  1. Prior initialization from IntentIR
  2. Single Bayesian update
  3. Multi-step Bayesian update (DataExfiltration walkthrough)
  4. Evidence accumulation with decay
  5. Combined scoring (β-weighted)
  6. Ranked output
  7. Backward compatibility (build_graph without scorer)
  8. Full pipeline integration
"""
from __future__ import annotations
import math
import pytest

# ── Import project modules ─────────────────────────────────────
from capextract.core.models import (
    IntentIR, PrimitiveCap, FunctionalCap, CapNode, NodeType,
)
from capextract.core.parser import ParsedSignal
from capextract.scoring.bayesian import (
    BayesianScorer, ALPHA_BASE, ALPHA_EXPECTED, ALPHA_RESOURCE,
)
from capextract.scoring.evidence import (
    EvidenceAccumulator, DEFAULT_LAMBDA, SIGMOID_K, SIGMOID_THETA,
)
from capextract.scoring.combined import DynamicScorer, ScoredCapability
from capextract.graph.builder import build_graph
from capextract.rules.tier2_rules import COMPOSITION_RULES, FUNCTIONAL_CONF_THRESHOLD


# ════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════

def _make_intent(**kwargs) -> IntentIR:
    """Convenience helper to create IntentIR with defaults."""
    defaults = dict(
        raw_prompt="test prompt",
        goal="test goal",
        expected_caps=[],
        scope_constraints=[],
        resource_hints=["console"],
        detected_language="python",
    )
    defaults.update(kwargs)
    return IntentIR(**defaults)


def _make_signal(cap: PrimitiveCap, trigger: str = "call:test",
                 line: int = 1) -> ParsedSignal:
    """Convenience helper to create a ParsedSignal."""
    return ParsedSignal(
        capability=cap,
        source_line=line,
        source_col=0,
        trigger=trigger,
        confidence=1.0,
    )


# ════════════════════════════════════════════════════════════════
# 1. Prior Initialization Tests
# ════════════════════════════════════════════════════════════════

class TestPriorInitialization:
    """Test that priors are correctly derived from IntentIR."""

    def test_base_prior_for_unknown_caps(self):
        """All caps should have at least ALPHA_BASE prior."""
        scorer = BayesianScorer()
        intent = _make_intent()
        scorer.init_priors(intent)

        # HTTP_REQUEST has no resource hint from console → should be ALPHA_BASE
        p = scorer.get_posterior("HTTP_REQUEST")
        assert p == pytest.approx(ALPHA_BASE, abs=0.01)

    def test_resource_hint_boost(self):
        """Primitives matching resource hints should get ALPHA_RESOURCE boost."""
        scorer = BayesianScorer()
        intent = _make_intent(resource_hints=["filesystem"])
        scorer.init_priors(intent)

        p_file = scorer.get_posterior("FILE_READ")
        p_http = scorer.get_posterior("HTTP_REQUEST")

        assert p_file == pytest.approx(ALPHA_BASE + ALPHA_RESOURCE, abs=0.01)
        assert p_http == pytest.approx(ALPHA_BASE, abs=0.01)

    def test_expected_cap_boost_for_functional(self):
        """Functional caps in expected_caps should get ALPHA_EXPECTED boost."""
        scorer = BayesianScorer()
        intent = _make_intent(
            expected_caps=["DataAnalytics", "MachineLearning"],
            resource_hints=["filesystem"],
        )
        scorer.init_priors(intent)

        p_da = scorer.get_posterior("DataAnalytics")
        p_ml = scorer.get_posterior("MachineLearning")
        p_exfil = scorer.get_posterior("DataExfiltration")

        # DataAnalytics: ALPHA_BASE + ALPHA_EXPECTED + ALPHA_RESOURCE (filesystem)
        assert p_da == pytest.approx(ALPHA_BASE + ALPHA_EXPECTED + ALPHA_RESOURCE, abs=0.01)
        # MachineLearning: ALPHA_BASE + ALPHA_EXPECTED (no filesystem resource match)
        assert p_ml == pytest.approx(ALPHA_BASE + ALPHA_EXPECTED, abs=0.01)
        # DataExfiltration: only ALPHA_BASE (not expected, no matching resource)
        assert p_exfil == pytest.approx(ALPHA_BASE, abs=0.01)

    def test_priors_clamped(self):
        """Priors should be clamped to [PRIOR_MIN, PRIOR_MAX]."""
        scorer = BayesianScorer()
        intent = _make_intent()
        scorer.init_priors(intent)

        for label, p in scorer.get_all_posteriors().items():
            assert p >= 0.01, f"{label} prior below PRIOR_MIN"
            assert p <= 0.90, f"{label} prior above PRIOR_MAX"


# ════════════════════════════════════════════════════════════════
# 2. Single Bayesian Update Tests
# ════════════════════════════════════════════════════════════════

class TestBayesianSingleUpdate:
    """Test that a single Bayes update moves posteriors correctly."""

    def test_import_signal_increases_posterior(self):
        """An import signal should increase the posterior for that cap."""
        scorer = BayesianScorer()
        intent = _make_intent()
        scorer.init_priors(intent)

        prior = scorer.get_posterior("HTTP_REQUEST")
        signal = _make_signal(PrimitiveCap.HTTP_REQUEST, trigger="import:requests")
        scorer.update(signal)
        posterior = scorer.get_posterior("HTTP_REQUEST")

        assert posterior > prior, "Import signal should increase posterior"

    def test_call_signal_stronger_than_import(self):
        """A call signal should have stronger evidence than an import."""
        scorer1 = BayesianScorer()
        scorer2 = BayesianScorer()
        intent = _make_intent()
        scorer1.init_priors(intent)
        scorer2.init_priors(intent)

        scorer1.update(_make_signal(PrimitiveCap.FILE_READ, trigger="import:os"))
        scorer2.update(_make_signal(PrimitiveCap.FILE_READ, trigger="call:open"))

        p_import = scorer1.get_posterior("FILE_READ")
        p_call = scorer2.get_posterior("FILE_READ")

        assert p_call > p_import, "Call signal should be stronger than import"

    def test_functional_cap_updated_on_primitive_signal(self):
        """When a required primitive fires, the functional cap should update."""
        scorer = BayesianScorer()
        intent = _make_intent()
        scorer.init_priors(intent)

        prior_exfil = scorer.get_posterior("DataExfiltration")

        # HTTP_REQUEST is required for DataExfiltration
        signal = _make_signal(PrimitiveCap.HTTP_REQUEST, trigger="import:requests")
        scorer.update(signal)

        posterior_exfil = scorer.get_posterior("DataExfiltration")
        assert posterior_exfil > prior_exfil, \
            "Functional cap should increase when a required primitive fires"


# ════════════════════════════════════════════════════════════════
# 3. Multi-step Bayesian Update (DataExfiltration walkthrough)
# ════════════════════════════════════════════════════════════════

class TestBayesianMultiStep:
    """
    Replays the DataExfiltration scenario from the implementation plan.
    Prior: 0.05 → after HTTP_REQUEST → after FILE_READ → after ENV_READ
    Each step should monotonically increase the score.
    """

    def test_data_exfiltration_walkthrough(self):
        scorer = BayesianScorer()
        intent = _make_intent()  # no exfil expected → low prior
        scorer.init_priors(intent)

        p0 = scorer.get_posterior("DataExfiltration")
        assert p0 == pytest.approx(ALPHA_BASE, abs=0.01)

        # Step 1: import requests → HTTP_REQUEST
        scorer.update(_make_signal(PrimitiveCap.HTTP_REQUEST, "import:requests"))
        p1 = scorer.get_posterior("DataExfiltration")
        assert p1 > p0, f"Step 1 failed: {p1} <= {p0}"

        # Step 2: open("/etc/shadow") → FILE_READ
        scorer.update(_make_signal(PrimitiveCap.FILE_READ, "call:open"))
        p2 = scorer.get_posterior("DataExfiltration")
        assert p2 > p1, f"Step 2 failed: {p2} <= {p1}"

        # Step 3: os.getenv("KEY") → ENV_READ
        scorer.update(_make_signal(PrimitiveCap.ENV_READ, "call:os.getenv"))
        p3 = scorer.get_posterior("DataExfiltration")
        assert p3 > p2, f"Step 3 failed: {p3} <= {p2}"

        # Final should be substantially higher than initial
        assert p3 > 0.30, f"Final posterior {p3} too low after 3 strong signals"

    def test_monotonic_increase_with_evidence(self):
        """Repeated evidence for the same cap should keep pushing posterior up."""
        scorer = BayesianScorer()
        intent = _make_intent()
        scorer.init_priors(intent)

        prev = scorer.get_posterior("FILE_READ")
        for i in range(5):
            scorer.update(_make_signal(PrimitiveCap.FILE_READ, f"call:open", line=i+1))
            curr = scorer.get_posterior("FILE_READ")
            assert curr >= prev, f"Step {i}: posterior should not decrease"
            prev = curr


# ════════════════════════════════════════════════════════════════
# 4. Evidence Accumulation Tests
# ════════════════════════════════════════════════════════════════

class TestEvidenceAccumulation:
    """Test the exponential decay evidence accumulator."""

    def test_initial_score_below_half(self):
        """With no evidence, sigmoid(0) < 0.5 because θ=0.5."""
        acc = EvidenceAccumulator()
        score = acc.get_score("FILE_READ")
        expected = 1.0 / (1.0 + math.exp(SIGMOID_K * SIGMOID_THETA))
        assert score == pytest.approx(expected, abs=0.01)

    def test_accumulate_increases_score(self):
        """Adding evidence should increase the score."""
        acc = EvidenceAccumulator()
        s0 = acc.get_score("FILE_READ")
        acc.accumulate(_make_signal(PrimitiveCap.FILE_READ, "call:open"))
        s1 = acc.get_score("FILE_READ")
        assert s1 > s0, "Evidence should increase score"

    def test_decay_decreases_score(self):
        """Decay steps without new evidence should decrease score."""
        acc = EvidenceAccumulator()
        acc.accumulate(_make_signal(PrimitiveCap.FILE_READ, "call:open"))
        s1 = acc.get_score("FILE_READ")

        for _ in range(50):
            acc.decay_step()
        s2 = acc.get_score("FILE_READ")

        assert s2 < s1, "Decay should decrease score over time"

    def test_functional_propagation(self):
        """Evidence for a required primitive should propagate to functional caps."""
        acc = EvidenceAccumulator()
        acc.accumulate(_make_signal(PrimitiveCap.HTTP_REQUEST, "import:requests"))

        # DataExfiltration requires HTTP_REQUEST
        raw_exfil = acc.get_raw_evidence("DataExfiltration")
        assert raw_exfil > 0, "Evidence should propagate to functional caps"

    def test_incremental_computation(self):
        """Verify E_t = λ × E_{t-1} + w_t × s_t formula."""
        acc = EvidenceAccumulator(decay_lambda=0.9)

        # Step 1: Add evidence (weight 0.95 for call)
        acc.accumulate(_make_signal(PrimitiveCap.FILE_READ, "call:open"))
        e1 = acc.get_raw_evidence("FILE_READ")
        assert e1 == pytest.approx(0.95, abs=0.01)

        # Step 2: Decay
        acc.decay_step()
        e2 = acc.get_raw_evidence("FILE_READ")
        assert e2 == pytest.approx(0.95 * 0.9, abs=0.01)

        # Step 3: Add more evidence
        acc.accumulate(_make_signal(PrimitiveCap.FILE_READ, "call:open", line=2))
        e3 = acc.get_raw_evidence("FILE_READ")
        assert e3 == pytest.approx(0.95 * 0.9 + 0.95, abs=0.01)


# ════════════════════════════════════════════════════════════════
# 5. Combined Scoring Tests
# ════════════════════════════════════════════════════════════════

class TestCombinedScoring:
    """Test the DynamicScorer that combines Bayesian + Evidence."""

    def test_combined_formula(self):
        """Verify Confidence = β × P + (1-β) × S."""
        scorer = DynamicScorer(beta=0.6)
        intent = _make_intent()
        scorer.init(intent)

        # Before any evidence
        conf = scorer.get_confidence("FILE_READ")
        p = scorer._bayesian.get_posterior("FILE_READ")
        s = scorer._evidence.get_score("FILE_READ")
        expected = 0.6 * p + 0.4 * s
        assert conf == pytest.approx(expected, abs=0.001)

    def test_signal_updates_both_components(self):
        """on_signal should update both Bayesian and Evidence."""
        scorer = DynamicScorer()
        intent = _make_intent()
        scorer.init(intent)

        before = scorer.get_confidence("FILE_READ")
        scorer.on_signal(_make_signal(PrimitiveCap.FILE_READ, "call:open"))
        after = scorer.get_confidence("FILE_READ")

        assert after > before, "Signal should increase combined confidence"

    def test_scored_capability_breakdown(self):
        """get_scored should return correct breakdown."""
        scorer = DynamicScorer(beta=0.6)
        intent = _make_intent()
        scorer.init(intent)

        scorer.on_signal(_make_signal(PrimitiveCap.HTTP_REQUEST, "import:requests"))
        sc = scorer.get_scored("HTTP_REQUEST")

        assert isinstance(sc, ScoredCapability)
        assert sc.label == "HTTP_REQUEST"
        assert sc.combined_score == pytest.approx(
            0.6 * sc.bayesian_score + 0.4 * sc.evidence_score, abs=0.001
        )
        assert sc.cap_type == "primitive"

    def test_functional_cap_type_detection(self):
        """Functional caps should be labeled as 'functional' in ScoredCapability."""
        scorer = DynamicScorer()
        intent = _make_intent()
        scorer.init(intent)

        sc = scorer.get_scored("DataExfiltration")
        assert sc.cap_type == "functional"


# ════════════════════════════════════════════════════════════════
# 6. Ranking Tests
# ════════════════════════════════════════════════════════════════

class TestRanking:
    """Test ranked output ordering and thresholding."""

    def test_ranked_output_sorted_descending(self):
        """get_ranked should return caps sorted by combined score descending."""
        scorer = DynamicScorer()
        intent = _make_intent(resource_hints=["filesystem", "network"])
        scorer.init(intent)

        scorer.on_signal(_make_signal(PrimitiveCap.FILE_READ, "call:open"))
        scorer.on_signal(_make_signal(PrimitiveCap.HTTP_REQUEST, "import:requests"))

        ranked = scorer.get_ranked(threshold=0.0)
        scores = [sc.combined_score for sc in ranked]
        assert scores == sorted(scores, reverse=True), "Output should be sorted descending"

    def test_threshold_filtering(self):
        """Capabilities below threshold should be excluded."""
        scorer = DynamicScorer()
        intent = _make_intent()
        scorer.init(intent)

        ranked_low = scorer.get_ranked(threshold=0.01)
        ranked_high = scorer.get_ranked(threshold=0.90)

        assert len(ranked_low) >= len(ranked_high), \
            "Higher threshold should produce fewer results"

    def test_functional_confidences_for_graph_builder(self):
        """get_functional_confidences should return only functional caps above threshold."""
        scorer = DynamicScorer()
        intent = _make_intent(expected_caps=["BasicComputation"], resource_hints=["console"])
        scorer.init(intent)

        # Fire CONSOLE_OUTPUT (required for BasicComputation)
        scorer.on_signal(_make_signal(PrimitiveCap.CONSOLE_OUTPUT, "call:print"))

        fc = scorer.get_functional_confidences()
        for label, conf in fc.items():
            assert conf >= FUNCTIONAL_CONF_THRESHOLD, \
                f"{label} below threshold: {conf}"
            # Should be functional cap labels
            try:
                FunctionalCap(label)
            except ValueError:
                pytest.fail(f"{label} is not a valid FunctionalCap")


# ════════════════════════════════════════════════════════════════
# 7. Backward Compatibility Tests
# ════════════════════════════════════════════════════════════════

class TestBackwardCompatibility:
    """Ensure build_graph still works without a scorer (static fallback)."""

    def test_build_graph_without_scorer(self):
        """build_graph with scorer=None should use static CompositionRule.score()."""
        intent = _make_intent()

        # Create primitive nodes that match DataExfiltration rule
        nodes = [
            CapNode(node_type=NodeType.PRIMITIVE, label="FILE_READ",
                    confidence=0.95, source_line=1, language="python",
                    metadata={"trigger": "call:open", "high_risk": False}),
            CapNode(node_type=NodeType.PRIMITIVE, label="HTTP_REQUEST",
                    confidence=0.95, source_line=2, language="python",
                    metadata={"trigger": "import:requests", "high_risk": False}),
        ]

        graph = build_graph(nodes, intent, "python", "test code", scorer=None)

        # Should still produce functional caps via static scoring
        assert len(graph.functional_caps) > 0, "Static fallback should produce functional caps"

    def test_build_graph_with_scorer_overrides_static(self):
        """build_graph with scorer should use dynamic confidences."""
        intent = _make_intent()
        scorer = DynamicScorer()
        scorer.init(intent)

        # Feed signals to scorer
        scorer.on_signal(_make_signal(PrimitiveCap.FILE_READ, "call:open"))
        scorer.on_signal(_make_signal(PrimitiveCap.HTTP_REQUEST, "import:requests"))

        nodes = [
            CapNode(node_type=NodeType.PRIMITIVE, label="FILE_READ",
                    confidence=0.95, source_line=1, language="python",
                    metadata={"trigger": "call:open", "high_risk": False}),
            CapNode(node_type=NodeType.PRIMITIVE, label="HTTP_REQUEST",
                    confidence=0.95, source_line=2, language="python",
                    metadata={"trigger": "import:requests", "high_risk": False}),
        ]

        graph = build_graph(nodes, intent, "python", "test code", scorer=scorer)

        # Check that functional nodes have bayesian_score metadata
        func_nodes = graph.get_nodes_by_type(NodeType.FUNCTIONAL)
        has_dynamic = any(n.bayesian_score is not None for n in func_nodes)
        assert has_dynamic, "Dynamic scorer should populate bayesian_score on nodes"


# ════════════════════════════════════════════════════════════════
# 8. History Tracking Tests
# ════════════════════════════════════════════════════════════════

class TestHistoryTracking:
    """Test optional confidence history tracking."""

    def test_history_disabled_by_default(self):
        """History should be empty when track_history=False."""
        scorer = DynamicScorer()
        scorer.init(_make_intent(), track_history=False)

        scorer.on_signal(_make_signal(PrimitiveCap.FILE_READ, "call:open"))
        scorer.on_step()

        assert scorer.get_history("FILE_READ") == []

    def test_history_enabled(self):
        """History should record snapshots when track_history=True."""
        scorer = DynamicScorer()
        scorer.init(_make_intent(), track_history=True)

        scorer.on_signal(_make_signal(PrimitiveCap.FILE_READ, "call:open"))
        scorer.on_step()
        scorer.on_step()
        scorer.on_step()

        history = scorer.get_history("FILE_READ")
        assert len(history) == 3, f"Expected 3 snapshots, got {len(history)}"
        assert all(isinstance(v, float) for v in history)


# ════════════════════════════════════════════════════════════════
# Run
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

