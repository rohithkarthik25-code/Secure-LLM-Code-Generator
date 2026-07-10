from __future__ import annotations
import math
import re
from typing import Dict, Set

from capextract.core.models import IntentIR, PrimitiveCap, FunctionalCap
from capextract.core.parser import ParsedSignal
from capextract.core.vector_mapper import VectorMapper
from sentence_transformers import util
import torch

def _camel_to_space(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1 \2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1 \2', s1).lower()

class MatrixEngine:
    """
    Replaces static tier 2 functional rules and bayesian inflation logic.
    Computes a mathematical gravity/motif match for functional capabilities
    based on the unique set of primitives seen.
    """
    def __init__(self):
        self.node_confidence: Dict[str, float] = {}
        self.edge_weights: Dict[str, float] = {}
        self.seen_prims: Set[str] = set()
        
        # DYNAMIC MOTIF GENERATION
        self.functional_motifs = {}
        self.initialized = False

    def init_priors(self, intent: IntentIR) -> None:
        """Uses upstream Intent to bias initial probabilities (Root Prior)."""
        self.node_confidence.clear()
        self.edge_weights.clear()
        self.seen_prims.clear()
        
        # 1. Initialize VectorMapper to dynamically generate Motif weights
        vm = VectorMapper.get_instance()
        if not self.functional_motifs:
            print("[MatrixEngine] Dynamically generating Functional Motifs using Semantic Projection...")
            for fc in FunctionalCap:
                if fc == FunctionalCap.UNCLASSIFIED: continue
                self.functional_motifs[fc.value] = {}
                fc_text = _camel_to_space(fc.value)
                fc_emb = vm.model.encode(fc_text, convert_to_tensor=True)
                
                for pc in PrimitiveCap:
                    if pc == PrimitiveCap.UNKNOWN: continue
                    pc_text = str(pc.value).replace('_', ' ').lower()
                    pc_emb = vm.model.encode(pc_text, convert_to_tensor=True)
                    
                    sim = util.cos_sim(fc_emb, pc_emb)[0].item()
                    if sim > 0.40: # Semantic threshold for a motif edge
                        self.functional_motifs[fc.value][pc.value] = round(sim, 2)
            print("[MatrixEngine] Motif Generation Complete.")
        
        expected_set = {e.lower().replace("_", "").replace(" ", "") for e in intent.expected_caps}
        
        # Initialize functional priors
        for fc in FunctionalCap:
            if fc == FunctionalCap.UNCLASSIFIED: continue
            
            prior = 0.05 # Base prior
            
            # Boost if upstream expected this
            if fc.value.lower().replace("_", "") in expected_set:
                prior += 0.30
                
            # Upstream high-risk bias
            if intent.upstream_intent:
                primary = intent.upstream_intent.get("ir_input_summary", {}).get("primary_intent", "")
                if fc.value in ["DataExfiltration", "CodeExecution", "SystemAutomation"]:
                    if primary in ["data_exfiltration", "code_injection", "privilege_escalation"]:
                        prior += 0.25
                    elif primary in ["legitimate_automation", "system_monitoring"]:
                        prior -= 0.02
                        
            self.node_confidence[fc.value] = max(0.01, min(prior, 0.90))
            
        self.initialized = True

    def init_edge_priors(self, kg_relations: list) -> None:
        pass

    def update(self, signal: ParsedSignal) -> None:
        """
        Takes a new primitive signal (dynamically mapped via VectorMapper)
        and mathematically recalculates the matrix.
        """
        if not self.initialized: return
        
        prim_cap = signal.capability.value
        
        # Update primitive confidence (Node weight)
        # Taking max ensures we keep the highest vector similarity score
        current_conf = self.node_confidence.get(prim_cap, 0.0)
        self.node_confidence[prim_cap] = max(current_conf, signal.confidence)
        
        # Only recalculate functional gravity if this is a NEW unique primitive
        # This fixes the naive bayes inflation bug
        if prim_cap not in self.seen_prims:
            self.seen_prims.add(prim_cap)
            self._recalculate_functional_gravity()
            
    def _recalculate_functional_gravity(self):
        """
        Calculates the posterior of functional capabilities based on structural gravity.
        """
        for func_cap, motif in self.functional_motifs.items():
            prior = self.node_confidence.get(func_cap, 0.05)
            
            # Sum the gravitational pull of all seen primitives belonging to this motif
            pull = sum(motif.get(p, 0.0) for p in self.seen_prims)
            
            # Normalize against the max possible pull for this motif
            max_pull = sum(motif.values())
            if max_pull == 0: continue
            
            # Likelihood based on motif completion (0.0 to 1.0)
            likelihood_pos = min(pull / max_pull, 1.0)
            
            # Bayes update
            l_pos = 0.50 + (likelihood_pos * 0.45) # Scale to 0.5 - 0.95
            l_neg = 1.0 - l_pos
            
            numerator = l_pos * prior
            denominator = numerator + l_neg * (1.0 - prior)
            if denominator > 1e-10:
                posterior = max(0.01, min(numerator / denominator, 0.99))
                self.node_confidence[func_cap] = posterior
                
                # Update Edge weights dynamically based on motif
                for p in self.seen_prims:
                    if p in motif:
                        edge_key = f"{func_cap}:DEPENDS_ON:{p}"
                        self.edge_weights[edge_key] = motif[p] * posterior

    def get_posterior(self, cap_label: str) -> float:
        return self.node_confidence.get(cap_label, 0.05)

    def get_all_posteriors(self) -> dict[str, float]:
        return dict(self.node_confidence)
        
    def get_edge_posterior(self, edge_key: str) -> float:
        return self.edge_weights.get(edge_key, 0.50)
