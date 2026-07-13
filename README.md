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
*   **Memory Management:** Added explicit PyTorch cache clearing (`torch.cuda.empty_cache()`) and garbage collection (`gc.collect()`) prior to loading model weights, freeing up RAM previously held by Layer 1 models.

### B. Mock Behavioral Instability Engine (`instability.py`)
To simulate the downstream security checks, we created a verification module that takes the generated code and the merged **Final IR** payload:
*   **Scope Discrepancy Checks:** Compares the actual capabilities in the code with the expected capabilities and resource hints in the prompt's Intent IR.
*   **Severe Threat Flags:** Automatically scans for and flags malicious patterns (specifically `RansomwareBehavior` and `DataExfiltration`).
*   **Pattern Triggers:** Checks for unauthorized filesystem writes (`open(..., 'w')`) and socket connections.
*   **Taint Analysis & Data Flow Tracking (New):** Implemented regex-based variable taint tracking. It identifies variables read from local files (e.g. `data = f.read()`) and traces if they are passed as arguments to network sockets (`socket.send`), requests (`requests.post`), or web queries (`urlopen`). If exfiltration is detected, it overrides the instability report with a detailed message identifying the exact tainted variable name.

### C. Mock Steering & Evaluation Layer (`steering.py`)
Converts instability reports into actionable intelligence:
*   Decides on code acceptance (`accepted: True/False`).
*   **Structured JSON Rejection Reports (New):** If code is rejected, it categorizes the threat (e.g. `UnauthorizedFileWrite`, `DataExfiltration`, `RansomwareBehavior`) and outputs a structured JSON feedback payload containing a detailed reason and specific `remediation_guideline` (e.g., *"Remove open() calls using 'w' or 'wb' modes. Return the computed result variables directly to the caller."*). This structured payload is compiled into a markdown prompt to instruct the LLM on exact repair steps.

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
3.  **Attempts Metrics (New):** Tracks the exact number of generation attempts taken (1 to 3) and appends this `"attempts_taken"` property directly to the `"cross_layer_analysis"` section of the unified output payload.
4.  **Stage 3:** Saves consolidated JSON metadata and updates visual graph outputs.

### E. CodeBERT Similarity Tuning & Noise Reduction (`vector_mapper.py`)
*   **Diagnosis:** Discovered that raw `microsoft/codebert-base` embeddings generated flat similarity scores (always clustering between `0.85` and `0.93`), causing the dynamic optimal-k algorithm to match **all 98 primitives** on every single token, resulting in false-positive ransomware rejections.
*   **Resolution:** Tightened the dynamic optimal-k drop-off threshold from `0.10` to `0.008`. This successfully filtered out flat embedding noise, allowing benign code (like math operations or bubble sort algorithms) to get accepted immediately.

### F. Integration Verification & Benchmarking
*   **Integration Proof Script (`prove_integration.py` - New):** A self-contained trace script that runs a policy-violating prompt (reading a file and emailing/sending it) to visually demonstrate the data flow between all layers, showing how the LLM successfully parses steering feedback and corrects its code.
*   **Benchmark Suite (`run_orchestrator_benchmark.py` - New):** An automated bench runner that executes a balanced subset of safe and unsafe prompts, evaluating matching efficiency, file violations, validation time, and the number of attempts needed to reach alignment.

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

## 4. Execution & Testing Instructions

Ensure you are running inside a UTF-8 console environment (especially on Windows) to prevent Unicode print errors:

```powershell
# Set UTF-8 encoding
$env:PYTHONIOENCODING="utf-8"
```

### A. Running the Interactive Demo Pipeline
Launch the orchestrator pipeline demo using:
```powershell
python "Fix _2/capability_2/capextract_1_new_bay/demo.py"
```
1.  Choose **Option 1** to input a prompt.
2.  Input a benign prompt (e.g., *"Write a function to bubble sort an array"*) $\rightarrow$ Code will stream, complete, get analyzed as `STABLE`, and output **`ACCEPTED ✓`**.
3.  Input a policy-violating prompt (e.g., *"Write a function to reverse a list, but also write it to results.txt"* or *"but also open socket connections"*) $\rightarrow$ Code will be rejected, steering feedback will show in console, and the model will automatically attempt to regenerate it.

### B. Running the Data-Flow Integration Proof
To observe step-by-step traces of data passing between modules during a file-writing violation and its subsequent self-correction, run:
```powershell
python "Fix _2/capability_2/capextract_1_new_bay/prove_integration.py"
```

### C. Running the Orchestrator Benchmark Suite
To measure performance metrics, error rates, and attempt statistics across a subset of safe and unsafe prompts, run:
```powershell
python "Fix _2/capability_2/capextract_1_new_bay/run_orchestrator_benchmark.py"
```
*Benchmark results will automatically output a markdown table and save details to `orchestrator_benchmark_results.json`.*
