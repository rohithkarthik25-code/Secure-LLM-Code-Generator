from __future__ import annotations
from dataclasses import dataclass
from typing import Set, Dict, List, Tuple

from capextract.core.models import PrimitiveCap

@dataclass(frozen=True)
class Taint:
    cap: PrimitiveCap
    line: int

class DataFlowTracker:
    """
    Lightweight Program Dependence Graph (PDG) and Taint Tracker.
    Used to track data-flow between variables to establish DATA_FLOWS_TO edges.
    """
    def __init__(self):
        # variable -> set of taints
        self.var_taints: Dict[str, Set[Taint]] = {}
        # cap -> line
        self.cap_sources: List[Tuple[PrimitiveCap, int]] = []
        # confirmed data flows: (source_cap, sink_cap)
        self.flow_edges: Set[Tuple[PrimitiveCap, PrimitiveCap]] = set()

    def bind(self, variable: str, capability: PrimitiveCap, line: int) -> None:
        """Register a variable as tainted by a source capability."""
        if variable not in self.var_taints:
            self.var_taints[variable] = set()
        self.var_taints[variable].add(Taint(capability, line))
        self.cap_sources.append((capability, line))

    def propagate(self, source_var: str, target_var: str) -> None:
        """Track variable-to-variable assignment (target = source)."""
        if source_var in self.var_taints:
            if target_var not in self.var_taints:
                self.var_taints[target_var] = set()
            self.var_taints[target_var].update(self.var_taints[source_var])

    def consume(self, variable: str, sink_capability: PrimitiveCap, line: int) -> None:
        """Record that a tainted variable is consumed by a sink capability."""
        if variable in self.var_taints:
            for taint in self.var_taints[variable]:
                self.flow_edges.add((taint.cap, sink_capability))
        self.cap_sources.append((sink_capability, line))

    def check_flow(self, source_cap: PrimitiveCap, sink_cap: PrimitiveCap) -> bool:
        """Check if a data-flow edge exists between two specific primitives."""
        return (source_cap, sink_cap) in self.flow_edges

    def get_flow_edges(self) -> List[Tuple[PrimitiveCap, PrimitiveCap]]:
        """Return all confirmed DATA_FLOWS_TO edge pairs."""
        return list(self.flow_edges)

    def get_scope_edges(self, line_start: int, line_end: int) -> List[Tuple[PrimitiveCap, PrimitiveCap]]:
        """Return SHARES_SCOPE edge pairs for primitives in the same function/block."""
        edges = []
        caps_in_scope = [
            cap for cap, line in self.cap_sources
            if line_start <= line <= line_end
        ]
        
        for i, cap1 in enumerate(caps_in_scope):
            for cap2 in caps_in_scope[i+1:]:
                if cap1 != cap2:
                    edges.append((cap1, cap2))
                    edges.append((cap2, cap1))
                    
        return list(set(edges))
