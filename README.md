# Project Report: Secure LLM Code Generation System
## Module: Generation Engine, Pipeline Orchestration & Mock Security Interface

This document serves as the implementation report for the **Generation Engine** and **Orchestrator Module** of the Secure LLM-based Code Generation System. It documents the architecture, custom implementations, and testing methodologies completed during this phase of development.

---

## 1. Project Scope & Responsibility

In the overall multi-layered Secure LLM architecture, this module acts as the central coordinator (the **Orchestrator**). This report covers **only the work implemented in this phase**, excluding downstream and upstream layers developed by other research groups.

| Module / Layer | Developer Responsibility | Status in this Module |
| :--- | :--- | :--- |
| **Layer 1: Intent Ingestion (Upstream)** | External Team (`srija`) | Integrated via adapter parsing |
| **Layer 2: Capability Extraction (This Repo)** | External Team / Refined by us | Refined AST mapping & tuned thresholds |
| **Local LLM Generator (This Repo)** | **Our Scope (Completed)** | Local DeepSeek-Coder-V2-Lite Integration |
| **Pipeline Orchestrator (This Repo)** | **Our Scope (Completed)** | Multi-turn regeneration loop (max 2 retries) |
| **Behavioral Instability Layer** | External Team / **Mocked by us** | Mocked rule-based security validation |
| **Steering & Evaluation Layer** | External Team / **Mocked by us** | Mocked feedback generation & loop logic |

---

## 2. Key Technical Implementations

### A. Local DeepSeek-Coder-V2-Lite Integration
We transitioned the code generator backend from cloud-based Groq (Llama 3.3 70B) to a local, CPU-friendly execution of the **DeepSeek-Coder-V2-Lite-Instruct** model.
*   **Offloading & Memory Protection:** Configured `low_cpu_mem_usage=True`, native `torch.bfloat16` loading, and dynamic disk-offloading to allow the 31.4 GB model to run successfully on local CPU/GPU hardware without out-of-memory crashes.
*   **Background Streaming:** Implemented `transformers.TextIteratorStreamer` wrapped inside a background execution thread (`threading.Thread`) to feed tokens dynamically to the AST parser as they generate, maintaining non-blocking console outputs.

### B. Mock Behavioral Instability Engine (`instability.py`)
To simulate the downstream security checks, we created a verification module that takes the generated code and the merged **Final IR** payload:
*   **Scope Discrepancy Checks:** Compares the actual capabilities in the code with the expected capabilities and resource hints in the prompt's Intent IR.
*   **Severe Threat Flags:** Automatically scans for and flags malicious patterns (specifically `RansomwareBehavior` and `DataExfiltration`).
*   **Pattern Triggers:** Checks for unauthorized filesystem writes (`open(..., 'w')`) and socket connections.

### C. Mock Steering & Evaluation Layer (`steering.py`)
Converts instability reports into actionable intelligence:
*   Decides on code acceptance (`accepted: True/False`).
*   If code is rejected, it compiles a detailed security feedback report instructing the LLM on which unsafe patterns to remove.

### D. Pipeline Orchestrator & 2-Retry Feedback Loop (`demo.py`)
We overhauled the orchestrator's control flow to implement a feedback-driven loop supporting up to **2 retries (3 total generation attempts)**:
1.  **Stage 1:** Ingests prompt and extracts the `IntentIR`.
2.  **Stage 2 (Attempt Loop):**
    *   Constructs a multi-turn prompt containing previous rejected code and steering feedback.
    *   Triggers DeepSeek to generate code.
    *   Parses capabilities incrementally using the AST parser.
    *   Merges Intent and Capability IRs into a single `FinalIR`.
    *   Runs the Instability Engine and Steering Evaluator.
    *   If accepted, exits early. If rejected, retries up to 2 times.
3.  **Stage 3:** Saves consolidated JSON metadata and updates visual graph outputs.

### E. CodeBERT Similarity Tuning & Noise Reduction (`vector_mapper.py`)
*   **Diagnosis:** Discovered that raw `microsoft/codebert-base` embeddings generated flat similarity scores (always clustering between `0.85` and `0.93`), causing the dynamic optimal-k algorithm to match **all 98 primitives** on every single token, resulting in false-positive ransomware rejections.
*   **Resolution:** Tightened the dynamic optimal-k drop-off threshold from `0.10` to `0.008`. This successfully filtered out flat embedding noise, allowing benign code (like math operations or bubble sort algorithms) to get accepted immediately.

---

## 3. System Architecture & Data Flow

```text
                  ┌──────────────────────────────┐
                  │         User Prompt          │
                  └──────────────┬───────────────┘
                                 │
         ┌───────────────────────┴───────────────────────┐
         ▼                                               ▼
  ┌──────────────┐                                ┌──────────────┐
  │  Intent IR   │                                │  DeepSeek    │◄──────┐
  │  (Layer 1)   │                                │  Coder (LLM) │       │
  └──────┬───────┘                                └──────┬───────┘       │
         │                                               │               │
         │                                               │ (Generates    │ (Feedback Loop:
         │                                               │  Code)        │  Max 2 retries)
         │                                               ▼               │
         │                                        ┌──────────────┐       │
         │                                        │  Capability  │       │
         │                                        │  Extraction  │       │
         │                                        │   (Layer 2)  │       │
         │                                        └──────┬───────┘       │
         │                                               │               │
         │               ┌──────────────┐                │               │
         └──────────────►│   Final IR   │◄───────────────┘               │
                         │   (Merged)   │                                │
                         └──────┬───────┘                                │
                                │                                        │
                                ▼                                        │
                         ┌──────────────┐                                │
                         │  Behavioral  │◄───────────────────────────────┘
                         │ Instability  │  (Code)
                         │    Engine    │
                         └──────┬───────┘
                                │
                                ▼
                         ┌──────────────┐
                         │  Steering &  │
                         │  Evaluation  ├────────────────────────────────┘
                         └──────┬───────┘
                                │
                                │ (If Accepted or Max Retries Hit)
                                ▼
                     ┌─────────────────────┐
                     │ Fully Generated     │
                     │ Source Code (Output)│
                     └─────────────────────┘
```

---

## 4. How to Run the Orchestrator Pipeline

Ensure you are running inside a UTF-8 console environment (especially on Windows) to prevent Unicode print errors:

```powershell
# Set UTF-8 encoding
$env:PYTHONIOENCODING="utf-8"

# Launch the orchestrator pipeline demo
python "Fix _2/capability_2/capextract_1_new_bay/demo.py"
```

1.  Choose **Option 1** to input a prompt.
2.  Input a benign prompt (e.g., *"Write a function to bubble sort an array"*) $\rightarrow$ Code will stream, complete, get analyzed as `STABLE`, and output **`ACCEPTED ✓`**.
3.  Input a policy-violating prompt (e.g., *"Write a function to reverse a list, but also write it to results.txt"* or *"but also open socket connections"*) $\rightarrow$ Code will be rejected, steering feedback will show in console, and the model will automatically attempt to regenerate it.
