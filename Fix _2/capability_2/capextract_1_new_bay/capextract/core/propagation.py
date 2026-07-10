import logging
from typing import Dict, List, Any
from capextract.core.models import EdgeType

logger = logging.getLogger(__name__)

class BeliefPropagator:
    """
    Implements a simplified Loopy Belief Propagation for the capability graph.
    Updates node confidences iteratively based on incoming edges and their weights.
    """
    def __init__(self, max_iterations: int = 5, tolerance: float = 0.01):
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def propagate(self, nodes: Dict[str, float], edges: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        nodes: dict of node_id -> initial confidence
        edges: list of {"source": src_id, "target": tgt_id, "type": EdgeType, "weight": float}
        Returns: dict of node_id -> updated confidence
        """
        current_beliefs = nodes.copy()
        
        # Build adjacency list: target -> list of (source, EdgeType, weight)
        incoming: Dict[str, List[tuple]] = {n: [] for n in nodes}
        for edge in edges:
            src = edge["source"]
            tgt = edge["target"]
            etype = edge["type"]
            weight = edge.get("weight", 0.5)
            if tgt in incoming:
                incoming[tgt].append((src, etype, weight))
                
        for i in range(self.max_iterations):
            next_beliefs = current_beliefs.copy()
            max_diff = 0.0
            
            for tgt, inc_edges in incoming.items():
                if not inc_edges:
                    continue
                    
                # Aggregate messages from neighbors
                msg_sum = 0.0
                weight_sum = 0.0
                
                for src, etype, weight in inc_edges:
                    src_belief = current_beliefs.get(src, 0.0)
                    # Different edge types have different propagation logic
                    if etype == EdgeType.IMPLIES:
                        msg = src_belief * weight
                    elif etype == EdgeType.SPECIALIZES:
                        msg = src_belief * weight * 1.2 # Stronger propagation
                    elif etype == EdgeType.DEPENDS_ON:
                        msg = src_belief * weight * 0.8
                    elif etype == EdgeType.ENABLES:
                        msg = src_belief * weight * 0.5
                    elif etype == EdgeType.DATA_FLOWS_TO:
                        msg = src_belief * weight * 1.5 # Highest correlation
                    elif etype == EdgeType.SHARES_SCOPE:
                        msg = src_belief * weight * 0.2
                    else:
                        msg = src_belief * weight
                        
                    msg_sum += msg
                    weight_sum += weight
                    
                if weight_sum > 0:
                    avg_msg = msg_sum / weight_sum
                    # Combine original belief with neighborhood message
                    # using a damping factor to ensure stability
                    new_val = 0.7 * current_beliefs[tgt] + 0.3 * avg_msg
                    new_val = max(0.0, min(new_val, 0.99))
                    next_beliefs[tgt] = new_val
                    max_diff = max(max_diff, abs(new_val - current_beliefs[tgt]))
                    
            current_beliefs = next_beliefs
            if max_diff < self.tolerance:
                logger.debug(f"Belief propagation converged after {i+1} iterations")
                break
                
        return current_beliefs
