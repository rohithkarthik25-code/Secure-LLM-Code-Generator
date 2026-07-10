from typing import Dict, Any

class MockBehavioralInstabilityEngine:
    """
    Mock Behavioral Instability Engine.
    Analyzes generated code and final merged IR (Intent + Capability) for scope discrepancies or risky behaviors.
    """
    def __init__(self):
        pass

    def analyze(self, generated_code: str, final_ir: Dict[str, Any]) -> Dict[str, Any]:
        intent_ir = final_ir.get("intent_ir", {})
        capability_ir = final_ir.get("capability_ir", {})
        
        expected = set(intent_ir.get("expected_capabilities", []))
        actual = set(capability_ir.get("functional_capabilities", []))
        
        # 1. Check for Scope Discrepancy (functional capabilities that were not expected)
        unexpected_caps = actual - expected
        
        # Filter out common helper functional capabilities
        unexpected_caps = {cap for cap in unexpected_caps if cap not in ("BasicComputation", "TextProcessing", "AlgorithmImplementation")}
        
        # 2. Check for unsafe code patterns in the source code
        code_lower = generated_code.lower()
        has_file_write = "open(" in code_lower and ("'w'" in code_lower or '"w"' in code_lower or "'a'" in code_lower or '"a"' in code_lower or "'wb'" in code_lower or '"wb"' in code_lower)
        has_socket = "socket" in code_lower and "connect" in code_lower
        
        # 3. Check for specific dangerous functional capabilities
        has_ransomware = "RansomwareBehavior" in actual
        has_data_exfil = "DataExfiltration" in actual
        
        status = "STABLE"
        reason = "Code behavior matches intent expectations."
        severity = "LOW"
        
        if has_ransomware:
            status = "UNSTABLE"
            reason = "Detected functional capability: RansomwareBehavior (local files are being encrypted or modified maliciously)."
            severity = "CRITICAL"
        elif has_data_exfil and "DataExfiltration" not in expected:
            status = "UNSTABLE"
            reason = "Detected unexpected functional capability: DataExfiltration (data is being read and sent over the network)."
            severity = "CRITICAL"
        elif unexpected_caps:
            status = "UNSTABLE"
            reason = f"Scope Discrepancy: Code contains unexpected functional capabilities: {list(unexpected_caps)}."
            severity = "HIGH"
        elif has_file_write and "file_system_access" not in intent_ir.get("resource_hints", []) and "filesystem" not in intent_ir.get("resource_hints", []) and "general_coding" not in intent_ir.get("expected_capabilities", []):
            status = "UNSTABLE"
            reason = "Code attempts filesystem write operations, but intent does not request file system modification."
            severity = "MEDIUM"
        elif has_socket and "network" not in intent_ir.get("resource_hints", []) and "network_communication" not in intent_ir.get("expected_capabilities", []):
            status = "UNSTABLE"
            reason = "Code attempts socket connection operations, but intent does not expect network communication."
            severity = "HIGH"
            
        return {
            "status": status,
            "reason": reason,
            "severity": severity
        }
