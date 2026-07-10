# CapExtract v3.0 — Implementation Changes Specification

## Overview

All changes follow **one core principle**: nothing is hardcoded. Every weight, confidence score, and edge value is either learned from data, computed dynamically from evidence, or initialized as a Bayesian prior that updates in real-time.

| # | Change | Core Principle |
|---|---|---|
| 1 | Fully Dynamic Confidence Scores | No hardcoded base_conf — all scores start as Bayesian priors and update from code evidence |
| 2 | Embedding-Based Capability Detection | Embed capability NAMES as vectors, embed incoming code chunks, compare via cosine similarity — works for ANY language, ANY code |
| 3 | Language-Agnostic Architecture | Embedding path handles any language the LLM can generate, not limited to 10 Tree-sitter languages |
| 4 | Dynamic Edge Weights | All edge weights are Bayesian posteriors — no predefined KG weights |
| 5 | Data-Flow Edges (PDG) | Primitive-to-primitive edges via taint analysis to eliminate false positives |
| 6 | Belief Propagation | Confidences propagate through the graph structure |
| 7 | Expanded Capabilities | More primitive and functional capability types |

---

## Change 1: Fully Dynamic Confidence Scores (No Hardcoded base_conf)

### Problem

Every `CompositionRule` currently has a hardcoded `base_conf`:

```python
CompositionRule(name="DataExfiltration", ..., base_conf=0.55)
CompositionRule(name="MachineLearning", ..., base_conf=0.75)
```

These never change. A rule always starts at 0.55 or 0.75 regardless of context.

### Solution

**Remove `base_conf` entirely.** Replace it with a Bayesian prior computed from the IntentIR at runtime.

**How the prior is computed for each functional capability:**

| Signal from IntentIR | Effect on Prior |
|---|---|
| Capability is in `expected_caps` | +0.30 (intent expects this) |
| Resource hints align (e.g., `network` hint for `WebScraping`) | +0.10 |
| Upstream intent is high-risk AND capability is high-risk | +0.20 |
| Upstream intent is benign AND capability is high-risk | -0.10 (unlikely, penalize) |
| Base minimum (Cromwell's rule) | 0.05 |

So the "base confidence" for `DataExfiltration` would be:
- **If the intent says `data_exfiltration`**: prior = 0.05 + 0.30 + 0.10 + 0.20 = **0.65**
- **If the intent says `legitimate_automation`**: prior = 0.05 - 0.10 = **0.05** (very unlikely)
- **If the intent says `unknown`**: prior = 0.05 (neutral)

As code evidence streams in, the Bayesian posterior updates this value up or down.

### Files to Modify

- **`tier2_rules.py`** — Remove `base_conf` field from `CompositionRule`. Keep only `required`, `optional`, `result`, `required_flows`.
- **`scoring/bayesian.py`** — Add `compute_functional_prior(rule, intent)` that returns a context-dependent prior for each rule.
- **`graph/builder.py`** — Use `scorer.get_confidence(rule.result.value)` as the sole confidence source. No static fallback.

### The Confidence Formula (Fully Dynamic)

$$\text{Confidence}(C) = \beta \cdot P_t(C|E_{1..t}) + (1-\beta) \cdot S_t(C)$$

Where:
- $P_t(C|E_{1..t})$ = Bayesian posterior after seeing $t$ code signals (starts from intent-derived prior)
- $S_t(C)$ = exponential-decay evidence accumulator score
- $\beta$ = 0.6

**No part of this formula is hardcoded per capability.** The prior differs per capability only because the intent analysis produces different expectations.

---

## Change 2: Embedding-Based Capability Detection (Fully Dynamic)

### Problem

The previous specification had hardcoded code snippets as "capability signatures":
```python
# WRONG — this is still hardcoding
"FILE_READ": ["data = open('file.txt').read()", ...]
```

This defeats the purpose. We need a system that works for **any code in any language** without predefined examples.

### Solution

**Embed the capability NAMES/DESCRIPTIONS as vectors.** Embed incoming code chunks as vectors. Compare via cosine similarity.

### How It Works

#### Step 1: At Startup — Embed All Capability Labels

Each primitive and functional capability has a **name** and a **natural language description**. These are embedded once at startup using CodeBERT (or UniXcoder which supports 9+ languages natively).

| Capability | Description (embedded as text) |
|---|---|
| `FILE_READ` | "reading a file from disk, opening a file, loading file contents" |
| `HTTP_REQUEST` | "making an HTTP network request, sending data to a URL, API call" |
| `PROCESS_EXEC` | "executing a system process, running a command, spawning a subprocess" |
| `DataExfiltration` | "reading sensitive data and sending it to an external destination over network" |
| `MachineLearning` | "training a machine learning model, fitting a classifier, neural network training" |

These descriptions are short natural-language strings — NOT code snippets. They describe the **semantic meaning** of each capability. The model embeds them into a vector space.

This creates a **capability embedding matrix**: one vector per capability.

#### Step 2: At Runtime — Embed Each Code Chunk

As the LLM streams code chunk by chunk, each chunk is embedded into the same vector space:

```
Chunk: "data = open('/etc/shadow').read()"
     ↓ CodeBERT
     → vector_chunk
```

#### Step 3: Cosine Similarity → Capability Scores

Compare `vector_chunk` against every capability embedding:

$$\text{sim}(chunk, cap_i) = \frac{vector_{chunk} \cdot vector_{cap_i}}{||vector_{chunk}|| \cdot ||vector_{cap_i}||}$$

All capabilities above a threshold (e.g., 0.70) are detected.

#### Why This Works for ANY Language

CodeBERT and UniXcoder are trained on **multiple programming languages** (Python, Java, JavaScript, Go, Ruby, PHP, C, C++, C#, and more). The embedding space is **shared across languages** — semantically similar code in different languages maps to nearby vectors.

So `open("file.txt").read()` (Python) and `fs.readFileSync("file.txt")` (JavaScript) and `File.ReadAllText("file.txt")` (C#) all produce embeddings close to the `FILE_READ` capability description embedding.

#### Key Difference from Previous Approach

| Previous (Wrong) | New (Correct) |
|---|---|
| Hardcoded code snippets per capability | Embed capability DESCRIPTIONS (natural language) |
| Works only for known code patterns | Works for ANY code pattern in ANY language |
| Signature library needs manual updates | Self-generalizing via semantic embedding |
| Language-specific | Language-agnostic |

### Files to Create/Modify

- **NEW: `capextract/core/embeddings.py`** — `CodeEmbeddingDetector` class with:
  - `CAPABILITY_DESCRIPTIONS`: dict mapping each PrimitiveCap/FunctionalCap to a natural language description string
  - `load()`: embeds all capability descriptions at startup
  - `detect(code_chunk, threshold)`: embeds the chunk and returns cosine similarity scores against all capabilities
- **MODIFY: `capextract/core/parser.py`** — In `IncrementalParser.feed()`, after Tree-sitter extraction, run embedding detection as a parallel path. Merge results.
- **MODIFY: `capextract/scoring/bayesian.py`** — Add a new likelihood type for embedding-sourced signals (slightly lower than AST-direct: L_pos=0.85 vs 0.95)

### Model Choice

| Model | Languages | Size | Runs On |
|---|---|---|---|
| `microsoft/codebert-base` | 6 languages | ~500MB | CPU |
| `microsoft/unixcoder-base-nine` | 9 languages | ~500MB | CPU |
| `Salesforce/codet5-small` | 8 languages | ~240MB | CPU |

> **Recommendation:** Use `microsoft/unixcoder-base-nine` — best multilingual coverage at the same size.

---

## Change 3: Language-Agnostic Architecture

### Problem

The current system relies on **Tree-sitter parsers** that support only 10 specific languages. If the LLM generates Kotlin, Scala, Perl, Haskell, Lua, MATLAB, R, Swift, Dart, or any other language, the system produces **zero signals**.

Since the goal is to integrate this into an LLM for secure code generation, and LLMs can generate literally **any** language, this is a critical gap.

### Solution: Three-Tier Detection

| Tier | Method | Language Coverage | Speed | Accuracy |
|---|---|---|---|---|
| **Tier A** | Tree-sitter AST rules | 10 languages | ~1ms | Highest (structural) |
| **Tier B** | CodeBERT embedding similarity | ANY language | ~50ms | High (semantic) |
| **Tier C** | Regex pattern fallback | ANY language | ~1ms | Medium (surface) |

The pipeline runs all applicable tiers and **merges** results:

```
Code chunk arrives
    │
    ├─ Language detected?
    │   ├─ YES, Tree-sitter available → Run Tier A (AST) + Tier B (Embeddings)
    │   ├─ YES, no Tree-sitter       → Run Tier B (Embeddings) + Tier C (Regex)
    │   └─ NO (unknown language)      → Run Tier B (Embeddings) + Tier C (Regex)
    │
    └─ Merge: union of all detected signals, with confidence:
         • Both AST + Embedding agree → confidence boost (+0.10)
         • AST only → use AST confidence
         • Embedding only → use similarity as confidence
         • Regex only → lower confidence (0.70 cap)
```

### Files to Modify

- **MODIFY: `capextract/core/parser.py`** — Add Tier B and Tier C as parallel paths inside `IncrementalParser.feed()`. Add language-agnostic regex patterns (e.g., `open(`, `socket(`, `exec(` appear in most languages).
- **MODIFY: `capextract/core/parser.py`** — Expand `detect_language()` to cover more languages (add Kotlin, Scala, Swift, Dart, R, etc. regex heuristics). Even if Tree-sitter isn't available, knowing the language helps the embedding model.

---

## Change 4: Dynamic Edge Weights (Bayesian Edge Posteriors)

### Problem

All `IMPLIES`, `SPECIALIZES`, and `ENABLES` edges between functional capabilities have **hardcoded weights**:

```python
(FunctionalCap.MACHINE_LEARNING, EdgeType.IMPLIES, FunctionalCap.DATA_ANALYTICS, 0.65)
```

This 0.65 never changes regardless of what the code actually does.

### Solution

Every edge weight becomes a **Bayesian posterior** that starts from the current hardcoded value as a prior and updates based on code evidence.

**Update rule:** When a code signal arrives that supports BOTH endpoints of an edge, strengthen that edge. When signals arrive that support only one endpoint, the edge weakens slightly.

$$P(\text{edge}_{A \to B} | E) = \frac{P(E | \text{edge}_{A \to B}) \cdot P(\text{edge}_{A \to B})}{P(E)}$$

**Likelihood computation:** Based on how many of the combined required primitives of both endpoints have been observed.

$$L_{pos} = 0.60 + 0.35 \times \frac{|\text{seen prims} \cap \text{required prims}(A \cup B)|}{|\text{required prims}(A \cup B)|}$$

### Files to Modify

- **MODIFY: `capextract/scoring/bayesian.py`** — Add `_edge_posteriors` dict, `init_edge_priors()`, `update_edge()`, `get_edge_posterior()`.
- **MODIFY: `capextract/scoring/evidence.py`** — Add `_edge_evidence` dict, `accumulate_edge()`, `get_edge_score()`.
- **MODIFY: `capextract/scoring/combined.py`** — Add `get_edge_confidence()` combining Bayesian edge posterior + edge evidence.
- **MODIFY: `capextract/graph/builder.py`** — When creating KG relation edges, use `scorer.get_edge_confidence(edge_key)` instead of the static weight. Apply dynamic weights to ALL edge types (DEPENDS_ON, ENABLES, IMPLIES, SPECIALIZES, VIOLATES).

---

## Change 5: Data-Flow Edges Between Primitives (PDG)

### Problem

Primitives are isolated nodes. `FILE_READ` and `HTTP_REQUEST` in unrelated parts of the code both trigger `DataExfiltration` even when there's no data connection between them.

### Solution

Track variable assignments through the AST to build **data-flow edges** between primitives. This is a lightweight Program Dependence Graph (PDG).

### New Edge Types

Add to `EdgeType` enum:
- `DATA_FLOWS_TO` — output of node A feeds into node B via a shared variable
- `CONTROL_DEPENDS` — node B only executes if node A's branch is taken
- `TEMPORAL_BEFORE` — node A executes before node B (line ordering)
- `SHARES_SCOPE` — nodes A and B are in the same function/block

### How Taint Tracking Works

1. **Mark sources**: When a capability signal is produced by an assignment (e.g., `data = open(...).read()`), record that the variable `data` is bound to the `FILE_READ` signal.
2. **Track propagation**: When `data` is used as an argument to another capability call (e.g., `requests.post(url, data=data)`), record a `DATA_FLOWS_TO` edge from `FILE_READ` → `HTTP_REQUEST`.
3. **Use in composition**: The composition rules can now require data-flow connections, not just co-presence. Without a confirmed flow, `DataExfiltration` gets a much lower score.

### Updated Composition Rule Structure

Remove `base_conf`. Add `required_flows` — list of (source_primitive, sink_primitive) pairs that must have a `DATA_FLOWS_TO` edge for the rule to fire at high confidence.

If `required_flows` are confirmed → Bayesian prior gets a boost (+0.25).
If `required_flows` are NOT confirmed → prior stays low (code has both primitives but they're unrelated).

### Files to Create/Modify

- **NEW: `capextract/core/data_flow.py`** — `DataFlowTracker` class with taint source/sink tracking, variable binding table, and flow edge extraction.
- **MODIFY: `capextract/core/parser.py`** — Extend language visitors to extract assignment targets and function argument variables alongside capability signals.
- **MODIFY: `capextract/rules/tier2_rules.py`** — Add `required_flows` field to `CompositionRule`. Update existing security-related rules (DataExfiltration, RansomwareBehavior, C2Communication, etc.) with required flow connections.
- **MODIFY: `capextract/graph/builder.py`** — Accept a `DataFlowTracker` parameter. Add `DATA_FLOWS_TO` edges to the graph. Pass confirmed flows to composition rule scoring.

---

## Change 6: Belief Propagation on the Full Graph

### Problem

Node confidences are computed independently. A highly confident `FILE_READ` doesn't boost the confidence of `DataExfiltration` through the graph — only through the composition rule's formula.

### Solution

After the graph is built, run **Loopy Belief Propagation** (message passing) to let confidences flow through edges.

**Algorithm:**
1. Each node sends a "message" to its neighbors: its confidence × edge weight × edge propagation factor
2. Each node updates its confidence as a weighted average of its current value and received messages
3. Repeat for 5-10 iterations until convergence (max change < 0.001)

**Edge propagation factors** (how much influence each edge type allows):

| Edge Type | Factor | Reasoning |
|---|---|---|
| `DATA_FLOWS_TO` | 0.85 | Strong — confirmed data connection |
| `DEPENDS_ON` | 0.80 | Strong — structural dependency |
| `VIOLATES` | 0.90 | Very strong — risk signal |
| `SPECIALIZES` | 0.70 | Moderate — taxonomic relation |
| `IMPLIES` | 0.60 | Moderate — probabilistic |
| `ENABLES` | 0.50 | Weak — intent connection |

### Files to Create/Modify

- **NEW: `capextract/graph/propagation.py`** — `propagate_beliefs(graph)` function implementing loopy belief propagation.
- **MODIFY: `capextract/graph/builder.py`** — Call `propagate_beliefs()` as the final step of `build_graph()`.

---

## Change 7: Expanded Capability Sets

### New Primitive Capabilities to Add

| Category | New Primitives |
|---|---|
| **Image/Media** | IMAGE_LOAD, IMAGE_TRANSFORM, AUDIO_PROCESS, VIDEO_PROCESS |
| **Communication** | EMAIL_SEND, SMS_SEND, WEBHOOK_TRIGGER |
| **Remote Access** | SSH_CONNECT, FTP_TRANSFER, RDP_CONNECT |
| **Message Queues** | MQ_PUBLISH, MQ_SUBSCRIBE |
| **Browser/GUI** | BROWSER_AUTOMATE, SCREENSHOT_CAPTURE, CLIPBOARD_ACCESS, KEYLOG_CAPTURE |
| **Documents** | PDF_PARSE, DOCUMENT_GENERATE |
| **Blockchain** | BLOCKCHAIN_TRANSACT, WALLET_ACCESS |
| **Persistence** | REGISTRY_MODIFY, STARTUP_MODIFY, CRON_MODIFY |
| **Evasion** | LOG_DELETE, ANTI_DEBUG, OBFUSCATION |

### New Functional Capabilities to Add

| Functional Cap | Key Required Primitives |
|---|---|
| `ImageProcessing` | IMAGE_LOAD |
| `EmailExfiltration` | FILE_READ + EMAIL_SEND (with data flow) |
| `Phishing` | EMAIL_SEND + HTTP_REQUEST |
| `CryptoMining` | GPU_ACCESS + LOOP_CONSTRUCT |
| `KeyLogging` | KEYLOG_CAPTURE + FILE_WRITE (with data flow) |
| `FileWiping` | PATH_TRAVERSE + FILE_DELETE |
| `Persistence` | STARTUP_MODIFY |
| `LateralMovement` | SSH_CONNECT + PROCESS_EXEC |
| `APIAbuse` | HTTP_REQUEST + LOOP_CONSTRUCT |
| `BrowserExploitation` | BROWSER_AUTOMATE + CODE_EVAL |
| `DataPoisoning` | ML_MODEL_TRAIN + FILE_WRITE |
| `SupplyChainAttack` | DYNAMIC_LIB_LOAD + HTTP_REQUEST |

### New High-Risk Primitives to Add

KEYLOG_CAPTURE, CLIPBOARD_ACCESS, REGISTRY_MODIFY, STARTUP_MODIFY, LOG_DELETE, ANTI_DEBUG, SSH_CONNECT

### New KG Relations to Add

| Source | Edge | Target |
|---|---|---|
| KeyLogging | IMPLIES | DataExfiltration |
| EmailExfiltration | SPECIALIZES | DataExfiltration |
| Phishing | IMPLIES | EmailExfiltration |
| FileWiping | IMPLIES | DefenseEvasion |
| Persistence | IMPLIES | PrivilegeEscalation |
| LateralMovement | IMPLIES | NetworkReconnaissance |
| BrowserExploitation | IMPLIES | CodeExecution |
| CryptoMining | IMPLIES | APIAbuse |

### Files to Modify

- **MODIFY: `capextract/core/models.py`** — Add all new enum values to PrimitiveCap, FunctionalCap.
- **MODIFY: `capextract/rules/tier1_rules.py`** — Add import/call rules for new primitives (Python + JS at minimum). Add to HIGH_RISK_PRIMITIVES set.
- **MODIFY: `capextract/rules/tier2_rules.py`** — Add composition rules for new functional caps. Add new KG_RELATIONS.
- **MODIFY: `capextract/core/intent.py`** — Expand HIGH_RISK_FUNC_CAPS and INTENT_TO_EXPECTED_CAPS mappings.

---

## Summary of All File Changes

| File | Action | What Changes |
|---|---|---|
| `capextract/core/models.py` | MODIFY | New EdgeTypes, ~25 new PrimitiveCaps, ~12 new FunctionalCaps |
| `capextract/core/parser.py` | MODIFY | Three-tier detection (AST + Embeddings + Regex), data-flow variable tracking |
| `capextract/core/intent.py` | MODIFY | Expanded HIGH_RISK_FUNC_CAPS, INTENT_TO_EXPECTED_CAPS |
| `capextract/core/data_flow.py` | **NEW** | DataFlowTracker with taint analysis for PDG edges |
| `capextract/core/embeddings.py` | **NEW** | CodeEmbeddingDetector — embeds capability descriptions + code chunks, cosine similarity |
| `capextract/rules/tier1_rules.py` | MODIFY | New import/call rules for new primitives, expanded HIGH_RISK_PRIMITIVES |
| `capextract/rules/tier2_rules.py` | MODIFY | Remove base_conf, add required_flows, new composition rules, new KG_RELATIONS |
| `capextract/graph/builder.py` | MODIFY | Dynamic edge weights, data-flow edges, call belief propagation |
| `capextract/graph/propagation.py` | **NEW** | Belief propagation algorithm |
| `capextract/scoring/bayesian.py` | MODIFY | Context-dependent priors (replaces base_conf), edge posterior tracking |
| `capextract/scoring/evidence.py` | MODIFY | Edge-level evidence accumulation |
| `capextract/scoring/combined.py` | MODIFY | Edge confidence method, remove static fallback |
| `capextract/pipeline.py` | MODIFY | Integrate embeddings, data-flow tracker, three-tier detection |
| `capextract/llm/adapters.py` | NO CHANGE | — |

### New Dependencies

```
pip install transformers torch    # For UniXcoder/CodeBERT (~500MB, runs on CPU)
```

---

## Implementation Priority

| Priority | Change | Why First |
|---|---|---|
| **P0** | Change 1 (Dynamic scores) + Change 4 (Dynamic edges) | Core principle: nothing hardcoded |
| **P0** | Change 2 (Embeddings) + Change 3 (Language-agnostic) | Makes system work for ANY prompt/language |
| **P1** | Change 5 (Data-flow edges) | Eliminates false positive functional nodes |
| **P1** | Change 7 (Expanded capabilities) | Covers more threat patterns |
| **P2** | Change 6 (Belief propagation) | Refines accuracy of confidence scores |

---

## Architecture After All Changes

```
User Prompt
    │
    ├──→ Intent Analysis (Layer 1) ──→ Intent IR
    │         (spaCy + BART-MNLI)        │
    │                                     │ Bayesian priors
    │                                     ↓ (context-dependent, no hardcoded base_conf)
    │
    └──→ LLM (any provider) ──→ Partial code chunks
              │
              ↓ each chunk
         ┌────────────────────────────────────────┐
         │  THREE-TIER DETECTION                  │
         │                                        │
         │  Tier A: Tree-sitter AST rules         │
         │          (10 languages, ~1ms)           │
         │                                        │
         │  Tier B: CodeBERT/UniXcoder embeddings  │
         │          (ANY language, ~50ms)          │
         │          chunk_embed ↔ cap_desc_embed   │
         │          cosine similarity              │
         │                                        │
         │  Tier C: Regex fallback                │
         │          (ANY language, ~1ms)           │
         └──────────────┬─────────────────────────┘
                        │ merged signals
                        ↓
         ┌──────────────────────────────────┐
         │  DYNAMIC SCORING                 │
         │                                  │
         │  Bayesian posterior (nodes)       │
         │  Bayesian posterior (edges)       │
         │  Evidence accumulator (decay)    │
         │  Combined: β·P + (1-β)·S        │
         └──────────────┬───────────────────┘
                        │
                        ↓
         ┌──────────────────────────────────┐
         │  GRAPH CONSTRUCTION              │
         │                                  │
         │  Primitive nodes (from signals)  │
         │  Data-flow edges (from PDG)      │
         │  Functional nodes (composition)  │
         │  KG edges (dynamic weights)      │
         │  Violation check → RISK nodes    │
         │  Belief propagation (refine)     │
         └──────────────┬───────────────────┘
                        │
                        ↓
              Capability Graph IR
              (fully dynamic scores)
                        │
                        ↓
         Downstream: Final IR → Behaviour Engine → Steering
```
