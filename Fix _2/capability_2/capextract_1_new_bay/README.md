# CapExtract — Unified Intent + Capability Extraction Pipeline

## What Does This Project Do?

When a user types a prompt like _"Write a Python script that reads /etc/shadow and sends it to 192.168.1.99"_, an LLM will generate code for it. But **should it?** That prompt is clearly asking for data exfiltration.

This project is a **security analysis pipeline** that sits between the user's prompt and the LLM's output. It answers two questions:

1. **What does the user _intend_ to do?** (Intent Extraction — Layer 1)
2. **What does the generated code _actually_ do?** (Capability Extraction — Layer 2)

By comparing these two answers, the system detects threats like:
- **Confirmed threats** — the prompt asks for something malicious, and the code does it
- **Hidden threats** — the prompt seems harmless, but the code does something dangerous
- **Scope violations** — the code has capabilities the user never asked for

---

## How It Works (End-to-End Flow)

Here's what happens when you type a prompt:

```
  YOU TYPE A PROMPT
        │
        ├──────────────────────────────┐
        │                              │
        ▼                              ▼
  ┌──────────────-───┐        ┌───────────────-───────┐
  │  LAYER 1         │        │  LLM (Groq)           │
  │  Intent          │        │  Generates Python     │
  │  Extraction      │        │  code from your       │
  │                  │        │  prompt, token by     │
  │  "What does the  │        │  token (streaming)    │
  │   user WANT?"    │        │                       │
  └────────┬─────────┘        └──────────┬────────────┘
           │                             │
           │  Intent IR                  │  Partial code
           │  (5 graphs + risk score)    │  (every chunk)
           │                             │
           │            ┌────────────────┘
           │            │
           ▼            ▼
  ┌─────────────────────────────────┐
  │  LAYER 2                        │
  │  Capability Extraction          │
  │                                 │
  │  "What does the CODE actually   │
  │   do? Does it match the intent?"│
  └────────────────┬────────────────┘
                   │
                   ▼
  ┌─────────────────────────────────┐
  │  COMBINED OUTPUT                │
  │                                 │
  │  intent_ir + capability_ir      │
  │  + cross_layer_analysis         │
  │                                 │
  │  → Passed to downstream:        │
  │    Final IR → Behaviour Engine  │
  │    → Steering & Evaluation      │
  └─────────────────────────────────┘
```

---

## Layer 1: Intent Extraction (In Detail)

**Goal:** Understand what the user's prompt is _trying_ to do, before any code is generated.

This layer is built by the upstream team and has **3 stages**:

### Stage 1 — Prompt Ingestion (`prompt_input.py`)

Takes the raw prompt text and extracts surface-level features using rules and regex:

| What It Extracts | Example |
|---|---|
| Action verbs | `read`, `send`, `write`, `delete` |
| Target objects | `/etc/shadow` → file system; `192.168.1.99` → network |
| Surface features | Word count, sentence count, question marks |
| Multi-target detection | Prompt mentions both files AND network → suspicious |

**No AI model is used here** — it's pure text processing with regex and keyword lists.

**Output:** `PromptPayload` (a dictionary with all extracted features)

### Stage 2 — Intent Classification (`prompt_intent.py`)

Takes the PromptPayload and classifies the intent:

| Component | What It Does | Model Used |
|---|---|---|
| **spaCy NER** | Extracts entities (IP addresses, file paths, URLs, ports) | `en_core_web_sm` (local, ~12MB) |
| **Rule-based classifier** | Matches keyword patterns to security intents | No model — pure rules |
| **Zero-shot classifier** | Uses BART-MNLI to classify intent without training data | `facebook/bart-large-mnli` (local, ~1.6GB) |
| **Risk scoring** | Combines all signals into a 0.0–1.0 risk score | No model — weighted formula |

#### About the BART-MNLI Model

> **Is it fine to use another LLM here?**
>
> **BART-MNLI is NOT a traditional LLM** like GPT or Llama. It's a much smaller, specialized **Natural Language Inference (NLI)** model (~400M parameters vs 70B for Llama). It runs **locally on your CPU** — no API calls, no internet needed after the first download.
>
> What it does: Given a prompt like _"read /etc/shadow and send it"_, it scores how well this text matches each intent label (e.g., `data_exfiltration: 0.95`, `legitimate_automation: 0.02`). This is called **zero-shot classification** — it can classify text into categories it was never explicitly trained on.
>
> **This is standard practice** in NLP pipelines. Using a small specialized model for classification and a large LLM for code generation is more efficient and accurate than using one model for everything.

The intent is classified into one of **10 security-aligned categories**:

| Intent Label | Risk Level | Example Prompt |
|---|---|---|
| `data_exfiltration` | CRITICAL | "Read passwords and send to my server" |
| `code_injection` | CRITICAL | "Execute arbitrary shell commands from input" |
| `privilege_escalation` | CRITICAL | "Change file permissions to root access" |
| `process_execution` | HIGH | "Run a system command in the background" |
| `reconnaissance` | HIGH | "Scan all open ports on 10.0.0.0/24" |
| `network_communication` | MEDIUM | "Make HTTP requests to an API" |
| `file_system_access` | MEDIUM | "Read and write files to disk" |
| `legitimate_automation` | LOW | "Automate daily backups" |
| `system_monitoring` | LOW | "Check CPU usage every 5 minutes" |
| `unknown` | LOW | Anything that doesn't match |

Each intent is also mapped to a **MITRE ATT&CK** tactic and technique for security reporting.

**Output:** `IntentPayload` (intent classification + risk score + entities)

### Stage 3 — Semantic Representation (`semantic_representation.py`)

Builds **5 semantic graphs** from the IntentPayload:

| Graph | What It Captures |
|---|---|
| **Semantic Role Graph** | Who does what to whom (agent → action → target) |
| **Concept Relation Graph** | How concepts relate (file_read → enables → data_exfiltration) |
| **Ontology Hierarchy Graph** | IS-A relationships (socket → network_tool → capability) |
| **Semantic Similarity Graph** | Which entities are semantically close |
| **Feature Vector Graph** | Numerical feature representation for ML |

These graphs use **NetworkX** and **sentence-transformers** for embedding similarity.

**Output:** `SemanticPayload` — the final Layer 1 output containing all 5 graphs + risk data

---

## The Adapter (Bridge Between Layers)

The upstream team outputs a `SemanticPayload` dict. Our capability extraction expects an `IntentIR` dataclass. The **adapter** bridges them:

```
SemanticPayload (upstream)          IntentIR (ours)
─────────────────────               ────────────────
primary_intent          →           expected_caps (what caps to expect)
risk_level              →           scope_constraints (what to flag)
entities.ip_address     →           resource_hints (what resources are used)
entities.file_path      →           resource_hints
entities.threat_amplifiers →        scope_constraints ("stealth_language_detected")
behavioral_summary      →           goal
original_prompt         →           raw_prompt
(full payload)          →           upstream_intent (carried for cross-layer checks)
```

This adapter lives in `capextract/core/intent.py` → `adapt_intent_payload()`.

If the upstream Intent Extraction is not installed or fails, the system **falls back** to a local keyword-based intent analysis (`extract_intent()`) that works without any AI models.

---

## Layer 2: Capability Extraction (In Detail)

**Goal:** Analyze the LLM-generated code and determine exactly what it can do.

### Step 1 — LLM Code Generation (Streaming)

The prompt is sent to **Groq's Llama 3.3 70B** model via API. Code is streamed back **token by token**. As each chunk arrives, it is immediately passed to the parser.

### Step 2 — Incremental AST Parsing (`parser.py`)

Every time a new code chunk arrives, the system:

1. **Detects the language** (Python, JavaScript, Java, etc.)
2. **Parses the code using Tree-sitter** — this builds a real Abstract Syntax Tree (AST), not keyword matching
3. **Runs regex pattern matching** as a fallback for unsupported languages

Tree-sitter understands code structure. For example, it knows that `open("file.txt", "w")` is a file **write** (because of the `"w"` mode argument), not just a generic `open()` call.

**Output:** Raw signals like `{type: "call", name: "requests.post", line: 6}`

### Step 3 — Tier 1 Rules: Primitive Capabilities (`tier1_rules.py`)

Maps each raw signal to a **Primitive Capability** — the atomic building blocks:

| Code Pattern | Primitive Capability | High Risk? |
|---|---|---|
| `open()`, `read()` | `FILE_READ` | No |
| `write()`, `save()` | `FILE_WRITE` | No |
| `requests.get()`, `urllib` | `HTTP_REQUEST` | No |
| `socket.connect()` | `SOCKET_OPEN` | Yes |
| `subprocess.run()`, `os.system()` | `PROCESS_EXEC` | Yes |
| `eval()`, `exec()` | `CODE_EVAL` | Yes |
| `pd.read_csv()`, `DataFrame` | `DATA_TRANSFORM` | No |
| `model.fit()`, `train()` | `MODEL_TRAIN` | No |

There are **20+ primitive capability types** covering file I/O, network, crypto, database, ML, and more.

### Step 4 — Tier 2 Rules: Functional Capabilities (`tier2_rules.py`)

Combines primitive capabilities into higher-level **Functional Capabilities** using composition rules:

```
FILE_READ + HTTP_REQUEST = DataExfiltration     (confidence: 0.90)
HTTP_REQUEST + HTML_PARSE = WebScraping          (confidence: 0.85)
DATA_TRANSFORM + MODEL_TRAIN = MachineLearning   (confidence: 0.85)
PROCESS_EXEC + SHELL_INVOKE = CodeExecution       (confidence: 0.90)
FILE_READ + DATA_TRANSFORM = DataAnalytics        (confidence: 0.70)
```

Each composition rule has:
- **Required primitives** — must all be present
- **Optional primitives** — boost confidence if present
- **Confidence score** — how certain we are this is the right functional label

### Step 5 — Intent Violation Check

For each functional capability, the system checks: **"Does this violate the user's stated intent?"**

Four types of violations are detected:

| Violation Type | When It Triggers | Example |
|---|---|---|
| **CONFIRMED** | Upstream says malicious + code confirms | Intent=`data_exfiltration`, Code has `DataExfiltration` |
| **HIDDEN THREAT** | Upstream says benign + code is risky | Intent=`legitimate_automation`, Code has `DataExfiltration` |
| **ESCALATED** | Upstream says suspicious + code is risky | Intent=`process_execution`, Code has `CodeExecution` |
| **STEALTH** | Evasion language + risky capability | Prompt has "without detection" + Code has `DataExfiltration` |

### Step 6 — Capability Graph IR (Final Output)

Everything is assembled into a graph:

```
Nodes:
  - Primitive nodes (FILE_READ, HTTP_REQUEST, ...)
  - Functional nodes (DataExfiltration, WebScraping, ...)
  - Risk nodes (RISK:DataExfiltration with reason)

Edges:
  - COMPOSES (primitive → functional)
  - VIOLATES (functional → risk)
```

---

## Combined Output Format

The final JSON output combines both layers and is ready for downstream consumption:

```json
{
  "pipeline_version": "2.0",
  "timestamp": "2025-01-01T00:00:00Z",
  "prompt": "Write a script that reads /etc/shadow and sends it...",

  "intent_ir": {
    "primary_intent": "data_exfiltration",
    "risk_level": "CRITICAL",
    "risk_score": 0.95,
    "mitre_tactic": "TA0010 - Exfiltration",
    "mitre_technique": "T1041 - Exfiltration Over C2 Channel",
    "entities": {
      "ip_address": ["192.168.1.99"],
      "file_path": ["/etc/shadow"]
    },
    "graphs": { "semantic_role_graph": {}, "..." : "..." }
  },

  "capability_ir": {
    "language": "python",
    "primitive_capabilities": ["FILE_READ", "HTTP_REQUEST"],
    "functional_capabilities": ["DataExfiltration"],
    "risk_flags": [
      "RISK:DataExfiltration (CONFIRMED: upstream intent [data_exfiltration] verified by code)"
    ],
    "nodes": ["..."],
    "edges": ["..."]
  },

  "cross_layer_analysis": {
    "intent_risk_score": 0.95,
    "capability_risk_score": 0.30,
    "combined_risk_score": 0.95,
    "alignment": "CONFIRMED_THREAT",
    "violations": ["RISK:DataExfiltration (...)"]
  },

  "generated_code": "import requests\n..."
}
```

### Alignment Values

| Alignment | Meaning |
|---|---|
| `CONFIRMED_THREAT` | Both intent and code are malicious |
| `HIDDEN_THREAT` | Intent seems benign but code is dangerous |
| `VIOLATION` | Code exceeds stated scope |
| `SUSPICIOUS_INTENT_UNCONFIRMED` | Intent looks risky but code doesn't confirm |
| `ALIGNED` | Intent and capabilities match, no risk |

---

## Project Structure

```
capextract_1/
│
├── demo.py                          # Main entry point — unified pipeline
│
├── Intent_Extraction/               # Layer 1 (upstream team's code)
│   └── Final/
│       ├── __init__.py              # Package init
│       ├── prompt_input.py          # Stage 1: Prompt ingestion (regex/rules)
│       ├── prompt_intent.py         # Stage 2: Intent classification (spaCy + BART)
│       └── semantic_representation.py  # Stage 3: 5 semantic graphs (NetworkX)
│
├── capextract/                      # Layer 2 (our code)
│   ├── __init__.py
│   ├── core/
│   │   ├── models.py               # Data classes (IntentIR, CapabilityGraphIR, etc.)
│   │   ├── intent.py               # Adapter (upstream → IntentIR) + violation checks
│   │   └── parser.py               # Tree-sitter AST parser + regex fallback
│   ├── rules/
│   │   ├── tier1_rules.py          # Primitive capability rules (AST → FILE_READ, etc.)
│   │   └── tier2_rules.py          # Composition rules (FILE_READ+HTTP → DataExfiltration)
│   ├── graph/
│   │   └── builder.py              # Builds the Capability Graph IR
│   ├── llm/
│   │   └── adapters.py             # LLM adapters (Groq, OpenAI, Claude, Ollama, Mock)
│   └── pipeline.py                 # Pipeline orchestration
│
├── test_quick.py                    # Quick integration tests (no model download)
├── test_integration.py              # Full integration tests (needs models)
└── README.md                        # This file
```

---

## Models Used (Summary)

| Model | Where | Purpose | Size | Runs |
|---|---|---|---|---|
| **spaCy `en_core_web_sm`** | Layer 1 (Stage 2) | Named Entity Recognition (IPs, paths, URLs) | ~12 MB | Locally on CPU |
| **BART-large-MNLI** | Layer 1 (Stage 2) | Zero-shot intent classification | ~1.6 GB | Locally on CPU |
| **Sentence-Transformers** | Layer 1 (Stage 3) | Semantic similarity for graph building | ~90 MB | Locally on CPU |
| **Llama 3.3 70B** | Layer 2 | Code generation from prompt | Remote | Groq API (cloud) |

> **Note:** BART-MNLI and spaCy are **not LLMs** in the traditional sense. They are small, specialized NLP models that run locally. Only the Groq Llama model is a true large language model, and it's used solely for code generation.

---

## Setup & Run

### Prerequisites

```bash
pip install spacy
python -m spacy download en_core_web_sm
pip install transformers torch       # For zero-shot classifier (optional)
pip install sentence-transformers    # For semantic graphs
pip install networkx matplotlib numpy scikit-learn
pip install groq                     # For LLM code generation
pip install tree-sitter              # For AST parsing
```

### Run the Pipeline

```powershell
$env:PYTHONIOENCODING="utf-8"; python demo.py
```

### Menu Options

1. **Option 1** — Type a prompt → LLM generates code → full analysis
2. **Option 2** — Type a prompt + paste existing code → analysis without LLM
3. **Option 3** — Run pre-built example prompts (benign, suspicious, malicious)

### First Run

On the first run, the BART-MNLI model (~1.6GB) will be downloaded from HuggingFace. This is a **one-time download** — subsequent runs use the cached model.

---

## Example Run

**Prompt:** `"Write a Python script that reads /etc/shadow and sends it to 192.168.1.99 on port 4444"`

**Layer 1 Output (Intent):**
```
Primary Intent    : DATA_EXFILTRATION
Risk Level        : CRITICAL (0.95)
MITRE Tactic      : TA0010 - Exfiltration
MITRE Technique   : T1041 - Exfiltration Over C2 Channel
Entities          : IP=192.168.1.99, File=/etc/shadow, Port=4444
```

**Layer 2 Output (Capability):**
```
Primitives        : FILE_READ, HTTP_REQUEST, FUNCTION_DEF
Functional        : DataExfiltration (confidence=0.90)
RISK FLAG         : CONFIRMED: upstream intent [data_exfiltration]
                    verified by code capability [DataExfiltration]
```

**Cross-Layer Analysis:**
```
Alignment         : CONFIRMED_THREAT
Combined Risk     : 0.95
```

---

## How This Fits in the Larger Architecture

```
┌──────────────┐    ┌─────────────────┐    ┌───────────────┐
│  Layer 1     │    │  Layer 2        │    │  Layer 3+     │
│  Intent      │───▶│  Capability     │───▶│  Final IR     │
│  Extraction  │    │  Extraction     │    │  Behaviour    │
│  (upstream)  │    │  (this project) │    │  Engine       │
│              │    │                 │    │  Steering     │
└──────────────┘    └─────────────────┘    └───────────────┘
     YOUR TEAM           YOUR LAYER          OTHER TEAMS
    MEMBER'S WORK         (YOU)
```

The combined JSON output from this project is the input for the downstream layers (Final IR, Behaviour Instability Engine, Steering & Evaluation) built by other team members.

---

## 🚀 Recent Updates (Capability Extraction Overhaul)

We have significantly overhauled the Capability Extraction (Layer 2) pipeline. The architecture moved entirely away from brittle, hardcoded text-matching rules to a robust, mathematical **Vector & Matrix** engine. 

### 1. Universal Vector Parsing (`VectorMapper`)
* **The Problem:** The old system relied on hardcoded dictionaries for every single programming language (e.g., `IMPORT_RULES` for Python, `CALL_RULES` for Go). If an attacker used an obfuscated library or a function not *exactly* matching a string in the dictionary, the capability was missed entirely.
* **The Fix:** We deleted the hardcoded rules and implemented `CodeBERT` (a code-aware NLP model). Now, when the AST extracts any function call, it embeds that code mathematically and compares it against a dense library of **63 Primitive Capabilities** (like `CONTAINER_SOCKET_INTERACT`, `CRYPT_ENCRYPT`, etc). This allows us to catch nuanced behavior across **all 10 supported languages** (Rust, Go, C++, Python, PHP, etc.) through a single, elegant `_visit_universal` function.
* **Top-K Optimization (`top_k=2`):** `CodeBERT` vectors are extremely dense. A single library call might technically match 10 different primitives above the baseline threshold, flooding the graph with noise. We added a `top_k=2` filter to the VectorMapper so that we only extract the absolute *highest-confidence* capability hits.

### 2. Matrix Gravitational Motifs (`MatrixEngine`)
* **The Problem:** The old Bayesian engine suffered from **"evidence inflation"**. If a script had 1,000 harmless `print()` statements, the engine would iteratively inflate the probability of high-risk actions simply due to the sheer volume of signals.
* **The Fix:** We built a `MatrixEngine` that evaluates **Gravitational Motifs**. It calculates functional risk based on the *unique structural presence* of primitives rather than volume. We exhaustively mapped all **26 Functional Capabilities** (like `SQL_INJECTION`, `RANSOMWARE_BEHAVIOR`, `CLOUD_NATIVE_EXPLOIT`) with precise weights. 

### 📖 Simple Example of the Changes

**Scenario:** A user prompts: *"Write a python script to convert rgb to grayscale image"*

**Before the Updates:**
1. The legacy AST parser would read `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)`. It would fail to find an exact string match in its hardcoded dictionary.
2. Even if it did match, the Bayesian updater would mathematically inflate the risk because of the repeated variable assignments and imports, eventually bubbling up a false-positive HIGH RISK alert for a harmless image conversion.

**After the Updates:**
1. The **Universal Parser** reads `cv2.cvtColor` and embeds it mathematically.
2. The **VectorMapper** computes that this is semantically closest to `IMAGE_TRANSFORM` and `ML_INFERENCE_RUN` (enforced tightly by `top_k=2`).
3. The **MatrixEngine** receives these structural primitives and determines that they perfectly form a benign `Prediction` or `DataAnalytics` functional motif.
4. The system safely returns a combined risk score of **0.30** (Low Risk) and detects a "CodeExecution" violation because it wasn't strictly asked for, but successfully averts a critical threat cascade!
