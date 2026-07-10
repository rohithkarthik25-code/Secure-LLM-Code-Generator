import json
import logging
from typing import Dict, Any, List, Set, Tuple

from capextract.core.models import IntentIR, PrimitiveCap, FunctionalCap
from capextract.llm.adapters import LLMAdapter

logger = logging.getLogger(__name__)

# The library of atomic reasoning modules for SELF-DISCOVER
REASONING_MODULES = [
    "Data Flow Tracking: Verify if sensitive data from a source flows to an external sink.",
    "Network Boundary Verification: Check if network operations cross expected boundary scopes.",
    "Privilege Check: Analyze if the operations attempt to escalate privileges or bypass access controls.",
    "Input Validation Check: Determine if user input is executed or passed to critical APIs without sanitization.",
    "Persistence Analysis: Look for modifications to registry, cron, or startup scripts.",
    "Evasion Detection: Identify code that deletes logs, disables debugging, or uses obfuscation.",
    "Cryptographic Integrity: Verify if encryption is used maliciously (e.g. ransomware) or correctly.",
    "Resource Exhaustion: Check for infinite loops or excessive memory allocation."
]

SYSTEM_PROMPT = """You are an expert security architect. Your task is to dynamically generate a JSON threat model for a given user intent using the SELF-DISCOVER methodology.
Process:
1. SELECT: Pick relevant reasoning modules from the provided library.
2. ADAPT: Rephrase the chosen modules specifically for the prompt's intent.
3. IMPLEMENT: Create a JSON threat model mapping functional capabilities to required primitive combinations and data flows.

Output ONLY valid JSON matching this schema:
{
  "selected_modules": ["<string>"],
  "adapted_modules": ["<string>"],
  "threat_model": {
    "FunctionalCapabilityName": {
      "required_primitives": ["PRIMITIVE_CAP_1", "PRIMITIVE_CAP_2"],
      "optional_primitives": ["PRIMITIVE_CAP_3"],
      "required_flows": [["PRIMITIVE_SOURCE", "PRIMITIVE_SINK"]],
      "base_confidence": 0.70
    }
  }
}"""

class SelfDiscoverEngine:
    """
    Implements SELF-DISCOVER methodology to dynamically generate threat models via LLM.
    """
    def __init__(self):
        pass

    def generate_threat_model(self, intent: IntentIR, adapter: LLMAdapter) -> Dict[str, Any]:
        """
        Makes a pre-generation LLM call to build the dynamic threat model.
        Returns the parsed JSON threat model, or an empty dict if it fails.
        """
        user_message = (
            f"Intent Goal: {intent.goal}\n"
            f"Expected Capabilities: {intent.expected_caps}\n"
            f"Resource Hints: {intent.resource_hints}\n\n"
            f"Reasoning Modules Library:\n" + "\n".join(f"- {m}" for m in REASONING_MODULES)
        )

        try:
            response = adapter.llm_call(SYSTEM_PROMPT, user_message)
            # Find JSON block in response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].strip()
            else:
                json_str = response.strip()
            
            model = json.loads(json_str)
            return model.get("threat_model", {})
        except Exception as e:
            logger.warning(f"SELF-DISCOVER generation failed, falling back to static rules: {e}")
            return {}

    def evaluate(self, 
                 threat_model: Dict[str, Any], 
                 present_primitives: Set[PrimitiveCap], 
                 flow_edges: List[Tuple[PrimitiveCap, PrimitiveCap]]) -> Dict[str, float]:
        """
        Evaluates extracted primitives and data-flow edges against the dynamic JSON threat model.
        Returns a dict of functional capabilities and their computed confidences.
        """
        results = {}
        for func_label, criteria in threat_model.items():
            req_prims = set(criteria.get("required_primitives", []))
            opt_prims = set(criteria.get("optional_primitives", []))
            req_flows = criteria.get("required_flows", [])
            base_conf = criteria.get("base_confidence", 0.50)

            # Check required primitives
            try:
                req_enum_set = {PrimitiveCap(p) for p in req_prims}
            except ValueError:
                continue # Unknown primitive in dynamic model
                
            if not req_enum_set.issubset(present_primitives):
                continue

            # Base match
            conf = base_conf

            # Check required flows
            flows_met = True
            for source, sink in req_flows:
                try:
                    s_cap, sink_cap = PrimitiveCap(source), PrimitiveCap(sink)
                    if (s_cap, sink_cap) not in flow_edges:
                        flows_met = False
                        break
                except ValueError:
                    flows_met = False
                    break
            
            if not flows_met:
                continue # Required flow missing

            # Bonus for optionals
            try:
                opt_enum_set = {PrimitiveCap(p) for p in opt_prims}
                matched_opts = len(opt_enum_set & present_primitives)
                conf += matched_opts * 0.05
            except ValueError:
                pass

            results[func_label] = min(conf, 0.99)

        return results
