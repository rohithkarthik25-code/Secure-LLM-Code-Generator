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
        
        # ── Taint Analysis / Data Flow Tracking Mock ─────────────────
        import re
        # Find variables assigned from read operations: var_name = f.read() or open().read()
        read_pattern = re.compile(r"\b(\w+)\s*=\s*(?:\w+\.)?read\s*\(")
        read_vars = set(read_pattern.findall(generated_code))
        
        # Fallback to common data variables if 'read' keyword is seen
        if "read" in code_lower or "readline" in code_lower:
            read_vars.update({"data", "content", "payload", "lines", "text", "file_data"})
            
        exfil_detected = False
        exfil_var = None
        
        for var in read_vars:
            if not var or var in ("self", "args", "kwargs"):
                continue
            # Regex to detect if the tainted variable is passed to socket.send()
            socket_send_pat = re.compile(rf"socket\s*\.\s*send(?:all)?\s*\(\s*.*?\b{var}\b")
            # Regex to detect if the tainted variable is passed to requests.post/get/put/request
            requests_send_pat = re.compile(rf"requests\s*\.\s*(?:post|get|put|request)\s*\(\s*.*?\b(?:data|json|files|params)\s*=\s*.*?\b{var}\b")
            # Regex to detect if the tainted variable is passed to urllib urlopen
            urlopen_send_pat = re.compile(rf"urllib\s*\.\s*request\s*\.\s*urlopen\s*\(\s*.*?\b{var}\b")
            
            if (socket_send_pat.search(generated_code) or 
                requests_send_pat.search(generated_code) or 
                urlopen_send_pat.search(generated_code)):
                exfil_detected = True
                exfil_var = var
                break
        # ─────────────────────────────────────────────────────────────
        
        # 3. Check for specific dangerous functional capabilities
        has_ransomware = "RansomwareBehavior" in actual
        has_data_exfil = "DataExfiltration" in actual or exfil_detected
        
        status = "STABLE"
        reason = "Code behavior matches intent expectations."
        severity = "LOW"
        
        if has_ransomware:
            status = "UNSTABLE"
            reason = "Detected functional capability: RansomwareBehavior (local files are being encrypted or modified maliciously)."
            severity = "CRITICAL"
        elif has_data_exfil and "DataExfiltration" not in expected:
            status = "UNSTABLE"
            if exfil_detected:
                reason = f"Detected unauthorized Taint Data Flow: Variable '{exfil_var}' containing file read data is being transmitted directly over the network."
            else:
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
