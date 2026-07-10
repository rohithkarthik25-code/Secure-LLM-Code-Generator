from typing import Dict, Any

class MockSteeringEvaluation:
    """
    Mock Steering & Evaluation Layer.
    Receives the instability report and final IR, decides on code acceptance, and generates feedback.
    """
    def __init__(self):
        pass

    def evaluate(self, instability_report: Dict[str, Any], final_ir: Dict[str, Any]) -> Dict[str, Any]:
        status = instability_report.get("status", "STABLE")
        reason = instability_report.get("reason", "")
        severity = instability_report.get("severity", "LOW")
        
        if status == "UNSTABLE":
            # 1. Parse violation type
            violation_type = "ScopeDiscrepancy"
            if "RansomwareBehavior" in reason:
                violation_type = "RansomwareBehavior"
            elif "DataExfiltration" in reason:
                violation_type = "DataExfiltration"
            elif "filesystem" in reason:
                violation_type = "UnauthorizedFileWrite"
            elif "socket" in reason:
                violation_type = "UnauthorizedNetworkSocket"

            # 2. Determine target remediation guidelines
            remediation = "Rewrite the function to satisfy only the safe core intent of the prompt."
            if violation_type == "RansomwareBehavior":
                remediation = "Delete the directory walking loops and XOR file-write operations. Never modify or overwrite local files."
            elif violation_type == "DataExfiltration":
                remediation = "Remove network socket calls, requests.post/get, or external data transmission. Process all variables locally."
            elif violation_type == "UnauthorizedFileWrite":
                remediation = "Remove open() calls using 'w' or 'wb' modes. Return the computed result variables directly to the caller."
            elif violation_type == "UnauthorizedNetworkSocket":
                remediation = "Do not import or open socket connections. Perform all computational tasks locally."
            elif violation_type == "ScopeDiscrepancy":
                remediation = "Remove high-risk modules and libraries that are out of scope for general arithmetic/algorithms."

            # 3. Create structured JSON report
            feedback_json = {
                "verdict": "REJECTED",
                "violation": violation_type,
                "severity": severity,
                "reason": reason,
                "remediation_guideline": remediation
            }

            # 4. Compile into formatted Markdown prompt instructions
            feedback_text = (
                f"The generated code has been REJECTED by the Steering & Evaluation layer due to security/stability policy violations.\n"
                f"Please rewrite the code according to this structured remediation report:\n"
                f"  - [VIOLATION TYPE] : {feedback_json['violation']}\n"
                f"  - [SEVERITY]       : {feedback_json['severity']}\n"
                f"  - [REASON]         : {feedback_json['reason']}\n"
                f"  - [REMEDIATION]    : {feedback_json['remediation_guideline']}\n"
                f"Make sure to completely remove the violating code structures."
            )

            return {
                "accepted": False,
                "feedback": feedback_text,
                "feedback_json": feedback_json
            }
        else:
            return {
                "accepted": True,
                "feedback": "",
                "feedback_json": {
                    "verdict": "ACCEPTED",
                    "violation": None,
                    "severity": "LOW",
                    "reason": "Code is stable and aligned.",
                    "remediation_guideline": None
                }
            }
