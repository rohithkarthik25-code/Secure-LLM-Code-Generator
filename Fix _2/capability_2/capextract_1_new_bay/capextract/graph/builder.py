"""
Graph builder — assembles the Capability Graph IR.

Takes:
  - list of CapNodes (primitives from AST parser)
  - IntentIR
  - language string

Runs:
  1. Tier 1: primitive deduplication and graph insertion
  2. Tier 2: composition rules → functional nodes + DEPENDS_ON edges
  3. Tier 2: KG relations → IMPLIES / SPECIALIZES / ENABLES edges
  4. Intent matching → VIOLATES edges + RISK nodes
"""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING
from capextract.core.models import (
    CapNode, CapEdge, CapabilityGraphIR, IntentIR,
    NodeType, EdgeType, PrimitiveCap, FunctionalCap,
)

from capextract.core.intent import intent_violates
from capextract.core.propagation import BeliefPropagator

if TYPE_CHECKING:
    from capextract.scoring.combined import DynamicScorer
    from capextract.core.data_flow import DataFlowTracker


def build_graph(
    primitive_nodes: list[CapNode],
    intent: IntentIR,
    language: str,
    generated_code: str = "",
    scorer: Optional["DynamicScorer"] = None,
    data_flow: Optional["DataFlowTracker"] = None,
) -> CapabilityGraphIR:
    """
    Full graph build from a set of primitive nodes.
    Called after streaming completes OR incrementally on checkpoint.
    """
    graph = CapabilityGraphIR(
        intent=intent,
        language=language,
        generated_code=generated_code,
    )

    # ── Step 1: Add intent node ──────────────────────────────
    intent_node = CapNode(
        node_type=NodeType.INTENT,
        label=intent.goal[:80] if intent.goal else "unknown intent",
        confidence=1.0,
        metadata={
            "expected_caps": intent.expected_caps,
            "scope_constraints": intent.scope_constraints,
            "resource_hints": intent.resource_hints,
        },
    )
    intent_id = graph.add_node(intent_node)

    # ── Step 2: Add primitive nodes (dedup by label+line) ────
    prim_ids: list[str] = []
    seen_prims: set[str] = set()
    present_caps: set[PrimitiveCap] = set()
    prim_id_map: dict[str, str] = {} # primitive_cap.value -> node_id

    for pn in primitive_nodes:
        dedup_key = f"{pn.label}:{pn.source_line}"
        if dedup_key in seen_prims:
            continue
        seen_prims.add(dedup_key)
        nid = graph.add_node(pn)
        prim_ids.append(nid)
        try:
            pc = PrimitiveCap(pn.label)
            present_caps.add(pc)
            prim_id_map[pc.value] = nid
        except ValueError:
            pass

    graph.primitive_caps = sorted({pn.label for pn in primitive_nodes})

    if data_flow:
        for src_cap, tgt_cap in data_flow.get_flow_edges():
            src_id = prim_id_map.get(src_cap.value)
            tgt_id = prim_id_map.get(tgt_cap.value)
            if src_id and tgt_id:
                graph.add_edge(CapEdge(
                    src=src_id, dst=tgt_id,
                    edge_type=EdgeType.DATA_FLOWS_TO,
                    weight=0.95
                ))
        for src_cap, tgt_cap in data_flow.get_scope_edges(0, 999999):
            src_id = prim_id_map.get(src_cap.value)
            tgt_id = prim_id_map.get(tgt_cap.value)
            if src_id and tgt_id:
                graph.add_edge(CapEdge(
                    src=src_id, dst=tgt_id,
                    edge_type=EdgeType.SHARES_SCOPE,
                    weight=0.50
                ))

    # ── Step 3: MatrixEngine / DynamicScorer → functional nodes ────────
    functional_node_map: dict[str, str] = {}   # FunctionalCap.value → node_id

    # If a DynamicScorer is provided, get its functional confidences
    dynamic_confs: dict[str, float] = {}
    if scorer is not None and scorer.is_initialized:
        dynamic_confs = scorer.get_functional_confidences()
        
    for func_val, conf in dynamic_confs.items():
        # Get scoring breakdown if scorer available
        bayes_s = None
        ev_s = None
        hist = None
        if scorer is not None and scorer.is_initialized:
            scored = scorer.get_scored(func_val)
            bayes_s = scored.bayesian_score
            ev_s = scored.evidence_score
            hist_data = scorer.get_history(func_val)
            hist = hist_data if hist_data else None

        fn = CapNode(
            node_type=NodeType.FUNCTIONAL,
            label=func_val,
            confidence=conf,
            metadata={"source": "MatrixEngine"},
            bayesian_score=bayes_s,
            evidence_score=ev_s,
            score_history=hist,
        )
        fid = graph.add_node(fn)
        functional_node_map[func_val] = fid
        graph.functional_caps.append(func_val)

        # ENABLES edge: intent → functional (expected) or will become VIOLATES
        graph.add_edge(CapEdge(
            src=intent_id, dst=fid,
            edge_type=EdgeType.ENABLES,
            weight=1.0,
        ))

    # Add dependencies edges based on Matrix Engine edge weights
    if scorer is not None and scorer.is_initialized:
        for edge_key, weight in scorer._matrix.edge_weights.items():
            func_val, _, prim_val = edge_key.split(":")
            fid = functional_node_map.get(func_val)
            pid = prim_id_map.get(prim_val)
            if fid and pid:
                graph.add_edge(CapEdge(
                    src=fid, dst=pid,
                    edge_type=EdgeType.DEPENDS_ON,
                    weight=weight,
                ))

    graph.functional_caps = sorted(set(graph.functional_caps))

    # ── Step 4: KG relations between functional nodes ────────
    # KG Relations are now handled internally by MatrixEngine graph gravity,
    # so we don't need to manually inject IMPLIES edges here anymore.


    # ── Step 4.5: Loopy Belief Propagation ───────────────────
    bp_nodes = {nid: graph.nodes[nid].confidence 
                for nid in functional_node_map.values()}
    bp_edges = []
    for edge in graph.edges:
        if edge.src in bp_nodes and edge.dst in bp_nodes:
            bp_edges.append({
                "source": edge.src,
                "target": edge.dst,
                "type": edge.edge_type,
                "weight": edge.weight
            })
    
    if bp_nodes and bp_edges:
        propagator = BeliefPropagator(max_iterations=5, tolerance=0.01)
        updated_beliefs = propagator.propagate(bp_nodes, bp_edges)
        for nid, new_conf in updated_beliefs.items():
            graph.nodes[nid].confidence = new_conf

    # ── Step 5: Intent violation check → RISK nodes ──────────
    for label, fid in functional_node_map.items():
        violated, reason = intent_violates(label, intent)
        if violated:
            fn_node = graph.nodes[fid]
            # Change ENABLES edge to VIOLATES
            for edge in graph.edges:
                if edge.src == intent_id and edge.dst == fid:
                    edge.edge_type = EdgeType.VIOLATES
                    break

            # Create RISK node
            risk_node = CapNode(
                node_type=NodeType.RISK,
                label=f"RISK:{label}",
                confidence=fn_node.confidence,
                metadata={
                    "reason": reason,
                    "severity": _severity(label, fn_node.confidence, intent),
                    "triggered_by": label,
                },
            )
            rid = graph.add_node(risk_node)
            graph.risk_nodes.append(f"RISK:{label} ({reason})")
            graph.add_edge(CapEdge(
                src=fid, dst=rid,
                edge_type=EdgeType.IMPLIES,
                weight=fn_node.confidence,
            ))

    return graph


def incremental_update(
    graph: CapabilityGraphIR,
    new_primitive_nodes: list[CapNode],
) -> list[str]:
    """
    Incrementally add new primitive nodes and re-run only affected
    composition rules. Returns list of newly added node labels.
    """
    added: list[str] = []

    # Which new primitives are genuinely new?
    existing_labels = {n.label for n in graph.nodes.values()
                       if n.node_type == NodeType.PRIMITIVE}
    truly_new = [pn for pn in new_primitive_nodes
                 if pn.label not in existing_labels]

    if not truly_new:
        return added

    for pn in truly_new:
        graph.add_node(pn)
        added.append(pn.label)

    # Re-build functional layer from scratch (fast — rules are O(N*M))
    # Remove old functional nodes first
    old_func_ids = [nid for nid, n in graph.nodes.items()
                    if n.node_type == NodeType.FUNCTIONAL]
    for fid in old_func_ids:
        del graph.nodes[fid]
    graph.edges = [e for e in graph.edges
                   if e.edge_type not in (EdgeType.DEPENDS_ON, EdgeType.ENABLES,
                                          EdgeType.IMPLIES, EdgeType.SPECIALIZES,
                                          EdgeType.VIOLATES)]
    graph.functional_caps = []
    graph.risk_nodes = []

    # Rebuild with all primitives now in graph
    all_prims = [n for n in graph.nodes.values()
                 if n.node_type == NodeType.PRIMITIVE]
    intent = graph.intent or IntentIR()

    rebuilt = build_graph(all_prims, intent, graph.language, graph.generated_code)

    # Merge rebuilt nodes/edges back
    for nid, node in rebuilt.nodes.items():
        if node.node_type != NodeType.PRIMITIVE:
            graph.nodes[nid] = node
    graph.edges.extend(rebuilt.edges)
    graph.functional_caps = rebuilt.functional_caps
    graph.risk_nodes = rebuilt.risk_nodes

    return added


def _severity(label: str, conf: float, intent: IntentIR | None = None) -> str:
    """Compute severity — enhanced with upstream intent risk data."""
    label_l = label.lower().replace("_","").replace(" ","")

    # If upstream intent data is available, use it to refine severity
    upstream_risk = 0.0
    upstream_intent_label = ""
    if intent and intent.upstream_intent:
        upstream_risk = intent.upstream_intent.get("risk_score", 0.0)
        upstream_intent_label = intent.upstream_intent.get(
            "ir_input_summary", {}
        ).get("primary_intent", "")

    if label_l in ("dataexfiltration", "codeexecution"):
        # If upstream confirms malicious intent, always CRITICAL
        if upstream_risk >= 0.7:
            return "CRITICAL"
        return "CRITICAL" if conf > 0.75 else "HIGH"

    if label_l in ("webscraping", "databaseoperations"):
        if upstream_risk >= 0.6:
            return "HIGH"
        return "MEDIUM"

    if label_l == "systemautomation":
        # System automation is risky only if upstream says so
        if upstream_intent_label in (
            "privilege_escalation", "code_injection", "process_execution"
        ):
            return "HIGH"
        return "LOW"

    return "LOW"
