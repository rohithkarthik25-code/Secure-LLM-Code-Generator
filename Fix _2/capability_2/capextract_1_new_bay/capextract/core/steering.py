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
        
        if status == "UNSTABLE":
            # Generate targeted steering feedback
            feedback = (
                f"The generated code has been rejected by the Steering & Evaluation layer due to security/stability issues:\n"
                f"Issue: {reason}\n"
                f"Please regenerate the code to remove this unsafe behavior. Make sure the code is stable and strictly conforms to the requested intent."
            )
            return {
                "accepted": False,
                "feedback": feedback
            }
        else:
            return {
                "accepted": True,
                "feedback": ""
            }
