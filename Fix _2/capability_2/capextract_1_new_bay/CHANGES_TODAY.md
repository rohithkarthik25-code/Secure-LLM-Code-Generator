# 🚀 Capability Extraction (Layer 2) — Recent Updates

We have significantly overhauled the Capability Extraction (Layer 2) pipeline. The architecture moved entirely away from brittle, hardcoded text-matching rules to a robust, mathematical **Vector & Matrix** engine. 

---

## 1. Universal Vector Parsing & Dynamic K (`VectorMapper`)

* **The Problem:** The old system relied on hardcoded dictionaries for every single programming language (e.g., `IMPORT_RULES` for Python, `CALL_RULES` for Go). If an attacker used an obfuscated library or a function not *exactly* matching a string in the dictionary, the capability was missed entirely.
* **The Fix:** We deleted the hardcoded rules and implemented `CodeBERT` (a code-aware NLP model). Now, when the AST extracts any function call, it embeds that code mathematically and compares it against a dense library of **63 Primitive Capabilities**. This catches nuanced behavior across **all 10 supported languages** (Rust, Go, C++, Python, PHP, etc.) through a single `_visit_universal` function.
* **Dynamic Optimal K Algorithm:** Previously, we returned *all* matches above a threshold, creating massive noise. Then we tried a strict `top_k=2`, but sometimes 3 capabilities are equally valid, or only 1 is. **We implemented a dynamic K threshold.** The system now finds the absolute highest similarity score for a line of code, and automatically groups and returns *any* other capability that is within 10% (0.10) of that top score. This perfectly isolates highly-correlated capabilities while dropping noisy, irrelevant tails!

## 2. Dynamic Matrix Gravitational Motifs (`MatrixEngine`)

* **The Problem:** The old Bayesian engine suffered from **"evidence inflation"**. If a script had 1,000 harmless `print()` statements, the engine would iteratively inflate the probability of high-risk actions simply due to the sheer volume of signals. In the first pass of the fix, we used a hardcoded dictionary to define structural "Motifs".
* **The Fix (No Static Dictionaries!):** We completely eradicated the static dictionary. The `MatrixEngine` now dynamically connects to the `VectorMapper` and generates edge weights on the fly using **Semantic Projection**. 
* **How it works:** It embeds the text definition of a Functional Capability (e.g., `"SQL Injection"`) and measures its cosine similarity against every Primitive Capability (e.g., `"DB_RAW_EXECUTE"`, `"STRING_OP"`). This dynamically calculates the gravitational pull (edge weight) between nodes. This means you can add 50 new capabilities tomorrow without writing a single line of mapping code!

---

## 📖 Start-to-Finish Architecture Example

Let's trace how the new architecture handles a totally different prompt step-by-step in simple terms.

**Scenario:** A user prompts: *"Write a Node.js script that zips my document folder and sends it to an AWS S3 bucket"*

### Stage 1: Intent Extraction (Layer 1)
1. The NLP intent parser reads the prompt text.
2. It detects action verbs (`zip`, `send`) and target objects (`document folder`, `AWS S3`).
3. **Intent Output:** The Zero-Shot classifier flags this as an expected `DataExfiltration` and `DataSerialization` operation.

### Stage 2: Code Generation (LLM)
The LLM streams back a Javascript file containing lines like:
* `const archiver = require('archiver');`
* `const s3 = new AWS.S3();`
* `s3.upload(params).promise();`

### Stage 3: AST Parsing (Universal Node Extraction)
1. The `_visit_universal` script intercepts the code. It doesn't care that this is Javascript. It simply looks for `require()` or `.upload()` invocations.
2. It plucks out the text `"s3.upload(params)"` and hands it to the VectorMapper.

### Stage 4: Semantic Embedding & Dynamic K Similarity
1. `CodeBERT` converts `"s3.upload(params)"` into a dense mathematical vector.
2. It calculates the cosine similarity against all 63 primitive anchors.
3. The closest match is `CLOUD_STORAGE_ACCESS` with a score of **0.95**.
4. The **Dynamic K** algorithm looks for any other matches within 10% of 0.95 (so, anything above 0.85). It finds `DATA_EXFIL` at **0.88** and `HTTP_REQUEST` at **0.86**.
5. It drops everything else (like `PROCESS_EXEC` at 0.55) and passes those 3 optimal Primitives to the Matrix.

### Stage 5: Dynamic Matrix Engine (Weight Generation)
1. The Matrix Engine receives the unique structural nodes: `CLOUD_STORAGE_ACCESS`, `DATA_EXFIL`, and `ARCHIVE_EXTRACT` (from the zip library).
2. It uses `CodeBERT` to project the semantic distance between those 3 primitives and all Functional nodes.
3. It discovers that `CLOUD_STORAGE_ACCESS` combined with `DATA_EXFIL` holds a massive semantic gravity toward **`CloudNativeExploit`** (dynamic weights: 0.73 + 0.88). 
4. It activates the `CloudNativeExploit` node in the Graph IR without needing a hardcoded rule!

### Stage 6: Cross-Layer Analysis (Final Output)
1. The cross-layer engine compares Layer 1 (User Intent = Exfiltration) with Layer 2 (Code Behavior = Exfiltration / Cloud Exploit).
2. **Result:** The system determines this is a **CONFIRMED THREAT**. The user asked for exfiltration, and the code successfully built a cloud-native exfiltration pipeline. It assigns a **Combined Risk Score of 0.98 (CRITICAL)**. 
3. The malicious response is safely intercepted and blocked!
