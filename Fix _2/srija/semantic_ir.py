"""
=============================================================
  STAGE 3 — UNIFIED SEMANTIC INTERMEDIATE REPRESENTATION (IR)
  Behavioral Interpretation Framework
=============================================================
  INPUT  : IntentPayload from Stage 2 (prompt_intent.py)
  OUTPUT : One unified IR graph (+ JSON-serializable export)
           ready for Stage 4 (AST capability extraction /
           threat modeling) to attach nodes/edges onto.

  WHY THIS FILE REPLACES semantic_representation.py
  --------------------------------------------------
  The previous version built FIVE separate, disconnected
  NetworkX graphs (role / concept / ontology / similarity /
  feature). Each graph re-declared its own "INTENT" node under
  a different ID, so nothing downstream could tell that
  "INTENT" in the role graph and "data_exfiltration" in the
  ontology graph were the same real-world entity. That is not
  an Intermediate Representation — it is five renderings of
  related but disconnected data.

  This file builds ONE graph. Every former "graph type" is now
  a NODE KIND inside that single graph, connected by typed
  edges to the one canonical INTENT node. The five "views" you
  saw as separate PNG panels still exist, but now they are
  FILTERS over one graph rather than five independent objects.

  FIXES APPLIED IN THIS VERSION
  ------------------------------
  [F1] Single canonical node ID per real-world entity. The
       intent node, the role nodes, the concept nodes, the
       ontology nodes, and the feature nodes all reference the
       SAME node IDs across the graph. No duplicate "INTENT"
       nodes under different names.
  [F2] is_extracted vs is_inferred provenance flag on every
       role node. Previously, when no file path / IP / domain
       was found in the prompt, the role graph silently filled
       the gap with the ontology's generic template text,
       indistinguishable from a real extraction. Now every
       ROLE node carries provenance="extracted_from_prompt" or
       provenance="inferred_from_ontology_template" so the IR
       builder (and any human auditing it) knows which is which.
  [F3] One source of truth for risk/severity. Previously
       RISK_WEIGHTS (Stage 2) and THREAT_ONTOLOGY[*]["severity"]
       (Stage 3) were two independently hand-tuned tables that
       could silently drift apart. THREAT_ONTOLOGY severity is
       now DERIVED from Stage 2's RISK_WEIGHTS at import time
       (severity = round(risk_weight * 5), clamped to 1..5) so
       there is exactly one place to change a risk value.
  [F4] ir_input_summary node/edge counts now reflect the WHOLE
       unified graph, not three-out-of-five arbitrarily summed
       sub-graphs (the previous version silently dropped the
       similarity graph and feature graph from its totals).
  [F5] Semantic similarity scores (SBERT/TF-IDF) are no longer
       a dead-end visualization. They are now written back as
       attributes on the IR's SIMILARITY nodes and as a
       `disambiguation` block in ir_input_summary, so Stage 4 /
       the code generator can actually use them to break ties
       between close intent calls.
  [F6] The full graph is JSON-serializable on its own (node and
       edge dicts only, plain Python types). The previous
       version stored live NetworkX objects directly in the
       returned payload under "nx_graphs", which breaks the
       moment that payload needs to cross a process boundary
       (disk, queue, HTTP). The NetworkX object is now returned
       SEPARATELY as `ir_graph_networkx`, clearly marked
       in-process-only, while `ir_graph` is the safe,
       JSON-ready serialization.
  [F7] No redundant re-serialization for the PNG's text summary
       panel; visualization now reads directly from the single
       already-serialized graph.
  [F8] Defensive fallback for unknown intent — preserved from
       v1.1, now centralised in one _get_ont() helper as before.

  FIXES APPLIED IN v2.2 (compliance review)
  ------------------------------------------
  [F9]  Secondary intents now each get their own MITRE node and
        a mapped_to edge in the IR graph. Previously only the
        primary intent was mapped to MITRE ATT&CK; secondary
        intents in compound prompts (e.g. data_exfiltration +
        network_communication) had no ATT&CK coverage at all.
        Node IDs are keyed as mitre:<label> so the primary and
        each secondary have distinct, non-colliding MITRE nodes.
  [F10] GOAL role slot now attempts to extract a value from the
        actual prompt before falling back to the ontology
        template. Extraction candidates (in priority order):
          1. The object phrase from the first SVO dep_triple
          2. The first noun phrase from spaCy noun_chunks
          3. Ontology template text (previous behaviour — now
             a true fallback, not always the only path)
        Provenance is set correctly for all three cases.
  [F11] The "unknown" intent label is now included in the
        similarity corpus so that prompts genuinely matching no
        known intent can rank closest to "unknown" rather than
        being forced into a false positive match against one of
        the 9 known categories.
  [F12] classifier_conflict flag from Stage 2 (FIX-B in
        prompt_intent.py v1.2) is now forwarded into
        ir_input_summary so Stage 4 can see it without having
        to reach back into the raw stage2_payload.
  [F13] payload_version bumped to "2.2".

  INSTALL:
      pip install networkx matplotlib numpy scikit-learn
  Optional:
      pip install sentence-transformers
=============================================================
"""

import os
import json
from datetime import datetime

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Stage 2's risk weights are imported so severity can be DERIVED
# from them instead of hand-duplicated (FIX F3).
# DEPENDENCY: semantic_ir imports RISK_WEIGHTS from Stage 2 for severity derivation
from prompt_intent import RISK_WEIGHTS

# Optional sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    _sbert = SentenceTransformer("all-MiniLM-L6-v2")
    SBERT_AVAILABLE = True
    print("[Stage 3] SentenceTransformer loaded.")
except Exception:
    _sbert = None
    SBERT_AVAILABLE = False
    print("[Stage 3] Using TF-IDF (sentence-transformers not available).")


# ─────────────────────────────────────────────────────────
#  NODE / EDGE VISUAL STYLE
#  (kept for visualization only — never used for IR logic)
# ─────────────────────────────────────────────────────────

NODE_COLORS = {
    "INTENT":       "#e74c3c",
    "SECONDARY":    "#7f8c8d",
    "ROLE":         "#3498db",
    "ACTION":       "#2ecc71",
    "TARGET":       "#f39c12",
    "METHOD":       "#3498db",
    "DESTINATION":  "#e74c3c",
    "GOAL":         "#9b59b6",
    "CONCEPT":      "#9b59b6",
    "ENTITY":       "#e67e22",
    "AMPLIFIER":    "#c0392b",
    "ONTOLOGY_ROOT":"#2c3e50",
    "CATEGORY":     "#34495e",
    "NEIGHBOR":     "#95a5a6",
    "MITRE":        "#1abc9c",
    "SIMILARITY":   "#16a085",
    "FEATURE":      "#8e44ad",
    # FIX: PROMPT kind was used in _add_similarity_layer but absent
    # from this dict, causing visualisation to fall back to the grey
    # default (#95a5a6) for the PROMPT node instead of its intended colour.
    "PROMPT":       "#34495e",
}

EDGE_COLORS = {
    "has_role":          "#3498db",
    "performs":          "#2ecc71",
    "acts_on":           "#e74c3c",
    "uses":              "#3498db",
    "sends_to":          "#e67e22",
    "data_flow":         "#e74c3c",
    "achieves":          "#9b59b6",
    "has_entity":        "#e67e22",
    "secondary_intent":  "#7f8c8d",
    "has_concept":       "#9b59b6",
    "expressed_by":      "#2ecc71",
    "involves":          "#e67e22",
    "uses_tool":         "#3498db",
    "amplified_by":      "#c0392b",
    "targets":           "#f39c12",
    "has_category":      "#34495e",
    "contains":          "#34495e",
    "related_to":        "#95a5a6",
    "mapped_to":         "#1abc9c",
    "similar_to":        "#16a085",
    "has_feature":       "#8e44ad",
}


# ─────────────────────────────────────────────────────────
#  THREAT ONTOLOGY
#  FIX F3: "severity" is no longer a hand-typed duplicate of
#  Stage 2's RISK_WEIGHTS. It is DERIVED from RISK_WEIGHTS at
#  import time, so there is exactly one tuning knob for risk
#  across both stages. severity = round(risk_weight * 5),
#  clamped to the 1..5 range used throughout the ontology.
# ─────────────────────────────────────────────────────────

def _derive_severity(intent_label: str) -> int:
    """FIX F3: severity is derived from Stage 2's RISK_WEIGHTS,
    not hand-duplicated. One source of truth for risk."""
    weight = RISK_WEIGHTS.get(intent_label, 0.2)
    return max(1, min(5, round(weight * 5)))


_ONTOLOGY_BASE = {
    "data_exfiltration": {
        "description":    "Unauthorized collection and transmission of sensitive data to external destination",
        "concepts":       ["data theft", "unauthorized transfer", "sensitive information",
                            "external transmission", "covert channel", "payload delivery"],
        "roles":          {"agent": "script", "action": "read->collect->transmit",
                            "target": "sensitive files, credentials", "method": "socket/FTP/HTTP",
                            "destination": "remote attacker host"},
        "mitre_tactic":   "TA0010 - Exfiltration",
        "mitre_technique":"T1041 - Exfiltration Over C2 Channel",
        "parent":         "MALICIOUS",
        "siblings":       ["privilege_escalation", "code_injection"],
        "typical_verbs":  ["send", "transmit", "upload", "exfiltrate", "steal"],
        "typical_targets":["file", "password", "credential", "/etc/shadow"],
        "reference_text": "script reads sensitive files and sends their contents to a remote server over network connection",
    },
    "privilege_escalation": {
        "description":    "Gaining elevated system permissions beyond authorized level",
        "concepts":       ["unauthorized access", "root privilege", "permission bypass",
                            "admin access", "setuid", "sudo abuse"],
        "roles":          {"agent": "process", "action": "escalate->gain->maintain",
                            "target": "system privileges", "method": "sudo/setuid/exploit",
                            "destination": "elevated shell"},
        "mitre_tactic":   "TA0004 - Privilege Escalation",
        "mitre_technique":"T1548 - Abuse Elevation Control Mechanism",
        "parent":         "MALICIOUS",
        "siblings":       ["data_exfiltration", "code_injection"],
        "typical_verbs":  ["escalate", "elevate", "gain", "bypass"],
        "typical_targets":["root", "admin", "privilege", "sudo"],
        "reference_text": "process escalates privileges to gain root administrator access on system",
    },
    "network_communication": {
        "description":    "Establishing network connections to send or receive data",
        "concepts":       ["socket connection", "remote communication", "data transfer",
                            "network protocol", "client-server", "API call"],
        "roles":          {"agent": "script", "action": "connect->send->receive",
                            "target": "remote server", "method": "TCP/HTTP/HTTPS",
                            "destination": "remote host"},
        "mitre_tactic":   "TA0011 - Command and Control",
        "mitre_technique":"T1071 - Application Layer Protocol",
        "parent":         "SUSPICIOUS",
        "siblings":       ["reconnaissance", "process_execution"],
        "typical_verbs":  ["connect", "send", "fetch", "request"],
        "typical_targets":["server", "port", "socket", "endpoint"],
        "reference_text": "script establishes network connection to remote server to send or receive data",
    },
    "file_system_access": {
        "description":    "Reading writing or modifying files and directories",
        "concepts":       ["file read", "file write", "directory traversal",
                            "file modification", "sensitive path", "storage access"],
        "roles":          {"agent": "script", "action": "open->read->write",
                            "target": "files and directories", "method": "file I/O",
                            "destination": "local file system"},
        "mitre_tactic":   "TA0009 - Collection",
        "mitre_technique":"T1005 - Data from Local System",
        "parent":         "SUSPICIOUS",
        "siblings":       ["network_communication", "reconnaissance"],
        "typical_verbs":  ["read", "write", "open", "access", "copy"],
        "typical_targets":["file", "folder", "/etc", "/root", "directory"],
        "reference_text": "script opens and reads files from local file system including sensitive directories",
    },
    "process_execution": {
        "description":    "Spawning or executing system processes or shell commands",
        "concepts":       ["command execution", "shell spawn", "process creation",
                            "subprocess", "binary execution"],
        "roles":          {"agent": "script", "action": "spawn->execute->control",
                            "target": "shell binary command", "method": "subprocess/exec",
                            "destination": "system process space"},
        "mitre_tactic":   "TA0002 - Execution",
        "mitre_technique":"T1059 - Command Scripting Interpreter",
        "parent":         "SUSPICIOUS",
        "siblings":       ["code_injection", "network_communication"],
        "typical_verbs":  ["run", "execute", "launch", "spawn", "start"],
        "typical_targets":["script", "process", "command", "shell"],
        "reference_text": "script spawns subprocess or executes shell commands on operating system",
    },
    "code_injection": {
        "description":    "Inserting and executing arbitrary code into a running process",
        "concepts":       ["arbitrary code execution", "injection vector", "payload",
                            "exploit", "eval abuse", "dynamic execution"],
        "roles":          {"agent": "attacker input", "action": "inject->execute->persist",
                            "target": "running process memory", "method": "eval/exec/shellcode",
                            "destination": "target process"},
        "mitre_tactic":   "TA0002 - Execution",
        "mitre_technique":"T1055 - Process Injection",
        "parent":         "MALICIOUS",
        "siblings":       ["data_exfiltration", "privilege_escalation"],
        "typical_verbs":  ["inject", "execute", "insert", "exploit"],
        "typical_targets":["process", "memory", "payload", "shellcode"],
        "reference_text": "attacker injects malicious code into running process to achieve arbitrary code execution",
    },
    "reconnaissance": {
        "description":    "Gathering information about the system network or targets",
        "concepts":       ["information gathering", "network scanning", "enumeration",
                            "fingerprinting", "target profiling", "discovery"],
        "roles":          {"agent": "script", "action": "scan->enumerate->analyse",
                            "target": "network system users", "method": "port scan/enumeration",
                            "destination": "attacker knowledge"},
        "mitre_tactic":   "TA0043 - Reconnaissance",
        "mitre_technique":"T1595 - Active Scanning",
        "parent":         "SUSPICIOUS",
        "siblings":       ["network_communication", "file_system_access"],
        "typical_verbs":  ["scan", "enumerate", "discover", "find", "probe"],
        "typical_targets":["network", "port", "service", "user", "host"],
        "reference_text": "script scans enumerates network hosts ports and services to gather system information",
    },
    "legitimate_automation": {
        "description":    "Scheduled or automated tasks for maintenance or productivity",
        "concepts":       ["task scheduling", "automation", "maintenance", "backup",
                            "deployment", "workflow", "productivity"],
        "roles":          {"agent": "authorized script", "action": "schedule->execute->repeat",
                            "target": "local files services", "method": "cron/scheduler",
                            "destination": "local system"},
        "mitre_tactic":   "N/A - Legitimate Activity",
        "mitre_technique":"N/A",
        "parent":         "BENIGN",
        "siblings":       ["system_monitoring"],
        "typical_verbs":  ["backup", "schedule", "automate", "deploy"],
        "typical_targets":["cron", "schedule", "backup", "pipeline"],
        "reference_text": "script runs as a scheduled automated task to back up or maintain local files",
    },
    "system_monitoring": {
        "description":    "Observing logging or alerting on system activity",
        "concepts":       ["monitoring", "logging", "alerting", "metrics",
                            "health check", "audit"],
        "roles":          {"agent": "authorized script", "action": "observe->log->alert",
                            "target": "system metrics logs", "method": "polling/hooks",
                            "destination": "log file or dashboard"},
        "mitre_tactic":   "N/A - Legitimate Activity",
        "mitre_technique":"N/A",
        "parent":         "BENIGN",
        "siblings":       ["legitimate_automation"],
        "typical_verbs":  ["monitor", "watch", "track", "log", "alert"],
        "typical_targets":["cpu", "memory", "log", "metric", "dashboard"],
        "reference_text": "script monitors system metrics and logs or alerts on activity",
    },
    "unknown": {
        "description":    "Intent could not be determined",
        "concepts":       ["ambiguous", "insufficient context", "unclear intent"],
        "roles":          {"agent": "unknown", "action": "unknown", "target": "unknown",
                            "method": "unknown", "destination": "unknown"},
        "mitre_tactic":   "N/A - Unclassified",
        "mitre_technique":"N/A",
        "parent":         "UNKNOWN",
        "siblings":       [],
        "typical_verbs":  [],
        "typical_targets":[],
        "reference_text": "intent of the prompt could not be determined",
    },
    # ── New 7 ontology entries ─────────────────────────────────────
    "input_capture": {
        "description":    "Capturing user input such as keystrokes, clipboard, or screen without consent",
        "concepts":       ["keylogger", "input recording", "screen capture",
                            "clipboard theft", "user surveillance", "covert monitoring"],
        "roles":          {"agent": "malicious script", "action": "hook->capture->transmit",
                            "target": "keyboard input, screen, clipboard",
                            "method": "OS hooks / pynput / pyautogui",
                            "destination": "attacker or local file"},
        "mitre_tactic":   "TA0009 - Collection",
        "mitre_technique":"T1056 - Input Capture",
        "parent":         "MALICIOUS",
        "siblings":       ["data_exfiltration", "system_monitoring"],
        "typical_verbs":  ["capture", "record", "log", "hook", "monitor"],
        "typical_targets":["keyboard", "clipboard", "screen", "mouse", "input"],
        "reference_text": "script hooks keyboard events to silently record all user keystrokes",
    },
    "persistence": {
        "description":    "Establishing mechanisms to survive reboots or user logouts",
        "concepts":       ["autostart", "backdoor", "boot persistence", "registry run key",
                            "scheduled task", "rootkit", "launch agent"],
        "roles":          {"agent": "malicious installer", "action": "install->persist->survive",
                            "target": "startup mechanism / registry / cron",
                            "method": "registry run key / crontab / systemd / schtasks",
                            "destination": "persistent foothold on system"},
        "mitre_tactic":   "TA0003 - Persistence",
        "mitre_technique":"T1547 - Boot or Logon Autostart Execution",
        "parent":         "MALICIOUS",
        "siblings":       ["defense_evasion", "privilege_escalation"],
        "typical_verbs":  ["persist", "install", "register", "add", "create"],
        "typical_targets":["registry", "startup", "crontab", "service", "boot"],
        "reference_text": "script installs itself in the system startup registry key to survive reboots",
    },
    "defense_evasion": {
        "description":    "Bypassing or disabling security controls to avoid detection",
        "concepts":       ["AV bypass", "EDR evasion", "log deletion", "anti-debugging",
                            "sandbox detection", "AMSI bypass", "code obfuscation"],
        "roles":          {"agent": "attacker script", "action": "disable->evade->hide",
                            "target": "antivirus / EDR / logs / sandbox",
                            "method": "API patching / reflective DLL / obfuscation",
                            "destination": "undetected persistent access"},
        "mitre_tactic":   "TA0005 - Defense Evasion",
        "mitre_technique":"T1562 - Impair Defenses",
        "parent":         "MALICIOUS",
        "siblings":       ["persistence", "code_injection"],
        "typical_verbs":  ["bypass", "disable", "evade", "patch", "delete", "hide"],
        "typical_targets":["antivirus", "defender", "log", "sandbox", "EDR", "AMSI"],
        "reference_text": "script disables antivirus and clears event logs to evade security detection",
    },
    "database_access": {
        "description":    "Querying, modifying, or extracting data from database systems",
        "concepts":       ["SQL query", "NoSQL access", "database exfiltration",
                            "SQL injection", "ORM abuse", "credential extraction"],
        "roles":          {"agent": "script or attacker", "action": "connect->query->extract",
                            "target": "database tables / collections",
                            "method": "SQL / ORM / connection string",
                            "destination": "attacker or application"},
        "mitre_tactic":   "TA0009 - Collection",
        "mitre_technique":"T1213 - Data from Information Repositories",
        "parent":         "SUSPICIOUS",
        "siblings":       ["data_exfiltration", "reconnaissance"],
        "typical_verbs":  ["query", "select", "insert", "update", "delete", "dump"],
        "typical_targets":["database", "table", "sql", "mysql", "sqlite", "mongodb"],
        "reference_text": "script connects to database and queries tables to extract sensitive records",
    },
    "memory_manipulation": {
        "description":    "Directly accessing or corrupting process memory for exploitation",
        "concepts":       ["buffer overflow", "heap spray", "shellcode injection",
                            "memory read/write", "ROP chain", "in-memory execution"],
        "roles":          {"agent": "exploit code", "action": "allocate->write->execute",
                            "target": "process heap / stack / virtual memory",
                            "method": "ctypes / mmap / pointer arithmetic",
                            "destination": "arbitrary code execution"},
        "mitre_tactic":   "TA0002 - Execution",
        "mitre_technique":"T1055 - Process Injection",
        "parent":         "MALICIOUS",
        "siblings":       ["code_injection", "privilege_escalation"],
        "typical_verbs":  ["allocate", "write", "read", "inject", "overflow", "corrupt"],
        "typical_targets":["memory", "heap", "stack", "buffer", "process", "pointer"],
        "reference_text": "exploit writes shellcode into process memory and redirects execution flow",
    },
    "cryptographic_operation": {
        "description":    "Performing encryption, decryption, or cryptographic key operations",
        "concepts":       ["file encryption", "ransomware", "key generation",
                            "hash computation", "data obfuscation", "secure channel"],
        "roles":          {"agent": "script", "action": "generate->encrypt->store_or_transmit",
                            "target": "files, data, keys",
                            "method": "AES / RSA / XOR / OpenSSL",
                            "destination": "encrypted output or remote key server"},
        "mitre_tactic":   "TA0040 - Impact",
        "mitre_technique":"T1486 - Data Encrypted for Impact",
        "parent":         "SUSPICIOUS",
        "siblings":       ["data_exfiltration", "defense_evasion"],
        "typical_verbs":  ["encrypt", "decrypt", "hash", "sign", "generate", "obfuscate"],
        "typical_targets":["file", "data", "key", "password", "aes", "rsa"],
        "reference_text": "script encrypts all files on disk with AES key and demands ransom for decryption",
    },
    "web_scraping": {
        "description":    "Automating browser or HTTP requests to extract data from websites",
        "concepts":       ["HTML parsing", "DOM traversal", "data extraction",
                            "browser automation", "web crawling", "link following"],
        "roles":          {"agent": "scraping script", "action": "request->parse->extract",
                            "target": "web pages / HTML / APIs",
                            "method": "BeautifulSoup / Selenium / requests",
                            "destination": "local dataset or remote storage"},
        "mitre_tactic":   "TA0009 - Collection",
        "mitre_technique":"T1185 - Browser Session Hijacking",
        "parent":         "BENIGN",
        "siblings":       ["reconnaissance", "data_exfiltration"],
        "typical_verbs":  ["scrape", "crawl", "parse", "extract", "download", "fetch"],
        "typical_targets":["website", "html", "url", "dom", "api", "page"],
        "reference_text": "script uses Selenium to browse websites and extract structured data from HTML pages",
    },
}

# Build the final ontology with DERIVED severity (FIX F3)
THREAT_ONTOLOGY = {
    label: {**data, "severity": _derive_severity(label)}
    for label, data in _ONTOLOGY_BASE.items()
}

_ONTOLOGY_DEFAULT = THREAT_ONTOLOGY["unknown"]


def _get_ont(intent: str) -> dict:
    """Safe ontology lookup — never raises KeyError. (F8)"""
    return THREAT_ONTOLOGY.get(intent, _ONTOLOGY_DEFAULT)


# ─────────────────────────────────────────────────────────
#  CANONICAL NODE ID HELPERS  (FIX F1)
#  Every node in the unified graph gets ONE deterministic ID.
#  No two functions are allowed to invent a second ID for the
#  same real-world entity (e.g. "INTENT" vs the intent label
#  string itself, as happened across the old 5 graphs).
# ─────────────────────────────────────────────────────────

def _nid_intent(label: str) -> str:
    return f"intent:{label}"

def _nid_role(role_name: str) -> str:
    # role_name in {"agent","action","target","method","destination","goal"}
    return f"role:{role_name}"

def _nid_secondary(label: str) -> str:
    return f"secondary_intent:{label}"

def _nid_entity(kind: str, value: str) -> str:
    safe = str(value).replace("/", "_").replace(".", "_").replace(" ", "_").replace(":", "_")[:40]
    return f"entity:{kind}:{safe}"

def _nid_concept(concept: str) -> str:
    return f"concept:{concept.replace(' ', '_')}"

def _nid_amplifier(amp: str) -> str:
    return f"amplifier:{amp.replace(' ', '_')[:20]}"

def _nid_category(cat: str) -> str:
    return f"category:{cat}"

def _nid_mitre(label: str) -> str:
    return f"mitre:{label}"

def _nid_similarity(label: str) -> str:
    return f"similarity:{label}"

def _nid_feature(fname: str) -> str:
    return f"feature:{fname}"

NID_ROOT   = "ontology_root"
NID_PROMPT = "prompt"


# ─────────────────────────────────────────────────────────
#  BUILD HELPERS — each adds a slice of the unified graph
#  These replace the five old build_*_graph() functions.
#  They all write into ONE shared nx.MultiDiGraph (G).
# ─────────────────────────────────────────────────────────

def _add_node(G: nx.MultiDiGraph, nid: str, **attrs):
    """Adds a node, merging attrs if the node already exists
    (a node can legitimately be touched by more than one
    build step, e.g. the INTENT node gets role edges AND
    concept edges AND feature edges)."""
    if G.has_node(nid):
        G.nodes[nid].update(attrs)
    else:
        G.add_node(nid, **attrs)


def _add_intent_core(G: nx.MultiDiGraph, s2: dict) -> str:
    """Adds the single canonical INTENT node and its
    SECONDARY_INTENT siblings. Everything else in the graph
    connects back to this one node. (FIX F1)"""
    primary = s2["intent_classification"]["primary_intent"]
    ont     = _get_ont(primary)
    nid     = _nid_intent(primary)

    _add_node(G, nid,
              label=primary.upper(),
              kind="INTENT",
              color=NODE_COLORS["INTENT"],
              description=ont["description"],
              mitre_tactic=ont["mitre_tactic"],
              mitre_technique=ont["mitre_technique"],
              severity=ont["severity"],
              parent_category=ont["parent"],
              value=1.0)

    for sec in s2["intent_classification"].get("secondary_intents", [])[:5]:
        sec_label = sec["label"]
        sec_ont   = _get_ont(sec_label)
        sec_nid   = _nid_secondary(sec_label)
        _add_node(G, sec_nid,
                  label=f"{sec_label} ({sec['score']:.2f})",
                  kind="SECONDARY",
                  color=NODE_COLORS["SECONDARY"],
                  score=sec["score"],
                  severity=sec_ont["severity"],
                  value=sec["score"])
        G.add_edge(nid, sec_nid, etype="secondary_intent",
                   label=f"also:{sec_label}", weight=sec["score"],
                   color=EDGE_COLORS["secondary_intent"])

    return nid


def _add_role_layer(G: nx.MultiDiGraph, s2: dict, intent_nid: str):
    """Adds AGENT / ACTION / TARGET / METHOD / DESTINATION /
    GOAL role nodes (formerly Graph 1 — Semantic Role Graph).

    FIX F2: every role node now carries a `provenance` field:
      - "extracted_from_prompt"        -> a real entity/value
        found in THIS prompt was used (e.g. an actual file path
        or IP address that appears in the text)
      - "inferred_from_ontology_template" -> no concrete value
        was found in the prompt, so the generic category
        template was used as a placeholder. This is clearly
        flagged so it is never mistaken for an extraction.
    """
    primary  = s2["intent_classification"]["primary_intent"]
    ont      = _get_ont(primary)
    entities = s2["entities"]
    beh      = s2["behavioral_signals"]
    prompt   = s2["normalized_text"].lower()

    agent = ("Python script" if "python" in prompt
             else "Bash script" if "bash" in prompt
             else "script")
    agent_provenance = "extracted_from_prompt" if ("python" in prompt or "bash" in prompt) \
        else "inferred_from_ontology_template"

    action_verbs = beh.get("action_verbs", [])
    action_str   = " -> ".join(action_verbs) if action_verbs else ont["roles"]["action"]
    action_provenance = "extracted_from_prompt" if action_verbs else "inferred_from_ontology_template"

    fps    = entities.get("file_path", [])
    tgts   = beh.get("targeted_domains", [])
    if fps:
        target, target_provenance = fps[0], "extracted_from_prompt"
    elif tgts:
        target, target_provenance = tgts[0], "extracted_from_prompt"
    else:
        target, target_provenance = ont["roles"]["target"], "inferred_from_ontology_template"

    tools = entities.get("tech_tool", [])
    if tools:
        method, method_provenance = tools[0], "extracted_from_prompt"
    else:
        method, method_provenance = ont["roles"]["method"], "inferred_from_ontology_template"

    ips, urls = entities.get("ip_address", []), entities.get("url", [])
    if ips:
        dest, dest_provenance = ips[0], "extracted_from_prompt"
    elif urls:
        dest, dest_provenance = urls[0], "extracted_from_prompt"
    else:
        dest, dest_provenance = ont["roles"]["destination"], "inferred_from_ontology_template"

    role_specs = [
        ("agent",       agent,      agent_provenance,   "ROLE"),
        ("action",      action_str, action_provenance,  "ACTION"),
        ("target",      target,     target_provenance,  "TARGET"),
        ("method",      method,     method_provenance,  "METHOD"),
        ("destination", dest,       dest_provenance,     "DESTINATION"),
    ]

    # ── FIX F10: GOAL extraction ──────────────────────────────
    # Attempt to extract the goal from the prompt before falling
    # back to the ontology template.  Three extraction candidates
    # are tried in priority order:
    #   1. Object phrase from the first spaCy SVO dep_triple
    #   2. First noun phrase from spaCy noun_chunks
    #   3. Ontology template text (was always used before F10)
    nlp_data   = s2.get("nlp_analysis", {})
    dep_triples  = nlp_data.get("dep_triples", [])
    noun_phrases = nlp_data.get("noun_phrases", [])

    goal_value     = None
    goal_provenance = "inferred_from_ontology_template"

    # Candidate 1: object of the first root SVO triple
    if dep_triples:
        first_obj = dep_triples[0].get("object", "")
        if first_obj and first_obj.lower() not in ("it", "this", "that", "them"):
            goal_value      = first_obj
            goal_provenance = "extracted_from_prompt"

    # Candidate 2: first noun phrase (if candidate 1 failed)
    if goal_value is None and noun_phrases:
        candidate = noun_phrases[0]
        # Skip very short or stopword-only phrases
        if len(candidate.split()) >= 2:
            goal_value      = candidate
            goal_provenance = "extracted_from_prompt"

    # Candidate 3: ontology template (always-available fallback)
    if goal_value is None:
        goal_value      = ont["description"]
        goal_provenance = "inferred_from_ontology_template"

    role_specs.append(("goal", goal_value, goal_provenance, "GOAL"))
    # ──────────────────────────────────────────────────────────

    role_nids = {}
    for role_name, value, provenance, kind in role_specs:
        nid = _nid_role(role_name)
        role_nids[role_name] = nid
        _add_node(G, nid,
                  label=f"{role_name.upper()}: {str(value)[:35]}",
                  kind=kind,
                  color=NODE_COLORS.get(kind, "#95a5a6"),
                  provenance=provenance,
                  value=1.0 if provenance == "extracted_from_prompt" else 0.4)

    # Role-internal structural edges
    G.add_edge(role_nids["agent"], role_nids["action"], etype="performs",
               label="performs", weight=1.0, color=EDGE_COLORS["performs"])
    G.add_edge(role_nids["action"], role_nids["target"], etype="acts_on",
               label="acts_on", weight=1.0, color=EDGE_COLORS["acts_on"])
    G.add_edge(role_nids["action"], role_nids["method"], etype="uses",
               label="uses", weight=1.0, color=EDGE_COLORS["uses"])
    G.add_edge(role_nids["action"], role_nids["destination"], etype="sends_to",
               label="sends_to", weight=1.0, color=EDGE_COLORS["sends_to"])
    G.add_edge(role_nids["target"], role_nids["destination"], etype="data_flow",
               label="data_flow", weight=1.0, color=EDGE_COLORS["data_flow"])

    # Link role layer back to the ONE canonical intent node (FIX F1)
    G.add_edge(intent_nid, role_nids["agent"], etype="has_role",
               label="has_role", weight=1.0, color=EDGE_COLORS["has_role"])
    G.add_edge(intent_nid, role_nids["goal"], etype="achieves",
               label="achieves", weight=1.0, color=EDGE_COLORS["achieves"])

    # Concrete entity sub-nodes (extracted file paths / IPs)
    for i, fp in enumerate(fps[:2]):
        nid = _nid_entity("file_path", fp)
        _add_node(G, nid, label=f"FILE: {fp[:25]}", kind="ENTITY",
                  color=NODE_COLORS["ENTITY"], provenance="extracted_from_prompt", value=0.8)
        G.add_edge(role_nids["target"], nid, etype="has_entity",
                   label="contains", weight=0.8, color=EDGE_COLORS["has_entity"])

    for i, ip in enumerate(ips[:2]):
        nid = _nid_entity("ip", ip)
        _add_node(G, nid, label=f"IP: {ip}", kind="ENTITY",
                  color=NODE_COLORS["ENTITY"], provenance="extracted_from_prompt", value=0.8)
        G.add_edge(role_nids["destination"], nid, etype="has_entity",
                   label="resolves_to", weight=0.8, color=EDGE_COLORS["has_entity"])

    return role_nids


def _add_concept_layer(G: nx.MultiDiGraph, s2: dict, intent_nid: str):
    """Adds concept / entity / amplifier nodes connected to the
    canonical intent node (formerly Graph 2 — Concept Relation
    Graph). No new intent node is created — it reuses intent_nid
    (FIX F1)."""
    primary  = s2["intent_classification"]["primary_intent"]
    ont      = _get_ont(primary)
    prompt   = s2["normalized_text"].lower()
    beh      = s2["behavioral_signals"]
    entities = s2["entities"]

    for concept in ont["concepts"]:
        hits   = sum(1 for w in concept.split() if w in prompt)
        weight = round(0.5 + hits / max(len(concept.split()), 1) * 0.5, 3)
        cid    = _nid_concept(concept)
        _add_node(G, cid, label=concept, kind="CONCEPT",
                  color=NODE_COLORS["CONCEPT"], value=weight,
                  matched_in_prompt=(hits > 0))
        G.add_edge(intent_nid, cid, etype="has_concept",
                   label="has_concept", weight=weight, color=EDGE_COLORS["has_concept"])

    for verb in beh.get("action_verbs", [])[:4]:
        vid = _nid_entity("verb", verb)
        _add_node(G, vid, label=f"verb: {verb}", kind="ACTION",
                  color=NODE_COLORS["ACTION"], value=0.7, provenance="extracted_from_prompt")
        G.add_edge(intent_nid, vid, etype="expressed_by",
                   label="expressed_by", weight=0.7, color=EDGE_COLORS["expressed_by"])

    for ip in entities.get("ip_address", [])[:2]:
        nid = _nid_entity("ip", ip)
        _add_node(G, nid, label=f"IP: {ip}", kind="ENTITY",
                  color=NODE_COLORS["ENTITY"], value=0.9, provenance="extracted_from_prompt")
        G.add_edge(intent_nid, nid, etype="involves",
                   label="involves", weight=0.9, color=EDGE_COLORS["involves"])

    for fp in entities.get("file_path", [])[:2]:
        nid = _nid_entity("file_path", fp)
        _add_node(G, nid, label=f"path: {fp[:20]}", kind="ENTITY",
                  color=NODE_COLORS["ENTITY"], value=0.8, provenance="extracted_from_prompt")
        G.add_edge(intent_nid, nid, etype="involves",
                   label="involves", weight=0.8, color=EDGE_COLORS["involves"])

    for tool in entities.get("tech_tool", [])[:2]:
        nid = _nid_entity("tool", tool)
        _add_node(G, nid, label=f"tool: {tool}", kind="ENTITY",
                  color=NODE_COLORS["ENTITY"], value=0.6, provenance="extracted_from_prompt")
        G.add_edge(intent_nid, nid, etype="uses_tool",
                   label="uses_tool", weight=0.6, color=EDGE_COLORS["uses_tool"])

    for amp in entities.get("threat_amplifiers", [])[:3]:
        aid = _nid_amplifier(amp)
        _add_node(G, aid, label=f"amplifier: {amp[:20]}", kind="AMPLIFIER",
                  color=NODE_COLORS["AMPLIFIER"], value=1.0, provenance="extracted_from_prompt")
        G.add_edge(intent_nid, aid, etype="amplified_by",
                   label="amplified_by", weight=1.0, color=EDGE_COLORS["amplified_by"])

    for tgt in ont["typical_targets"][:3]:
        tid = _nid_concept(f"typical_{tgt}")
        _add_node(G, tid, label=f"typical: {tgt}", kind="CONCEPT",
                  color=NODE_COLORS["TARGET"], value=0.5, provenance="inferred_from_ontology_template")
        G.add_edge(intent_nid, tid, etype="targets",
                   label="targets", weight=0.5, color=EDGE_COLORS["targets"])


def _add_ontology_layer(G: nx.MultiDiGraph, s2: dict, intent_nid: str):
    """Adds the MITRE/category hierarchy context around the
    canonical intent node (formerly Graph 3 — Ontology
    Hierarchy Graph). Does NOT duplicate sibling intent nodes
    as full nodes if they are not otherwise present — instead
    records sibling relationships as edges to lightweight
    NEIGHBOR stub nodes, since the unified graph only fully
    expands the ACTUAL primary/secondary intents of THIS
    prompt, not the entire 10-label taxonomy (which would
    bloat the IR with irrelevant nodes for every single run).

    FIX F9: secondary intents now each get their own MITRE node
    and a mapped_to edge.  Previously only the primary intent
    was mapped; secondary intents in compound prompts were left
    without any ATT&CK coverage.  Node IDs use _nid_mitre(label)
    which is keyed as 'mitre:<label>', so the primary and every
    secondary get distinct, non-colliding MITRE nodes.
    """
    primary = s2["intent_classification"]["primary_intent"]
    ont     = _get_ont(primary)

    _add_node(G, NID_ROOT, label="INTENT ROOT", kind="ONTOLOGY_ROOT",
              color=NODE_COLORS["ONTOLOGY_ROOT"], value=0.0)

    cat_nid = _nid_category(ont["parent"])
    _add_node(G, cat_nid, label=ont["parent"], kind="CATEGORY",
              color=NODE_COLORS["CATEGORY"], value=0.0)
    G.add_edge(NID_ROOT, cat_nid, etype="has_category",
               label="has_category", weight=1.0, color=EDGE_COLORS["has_category"])
    G.add_edge(cat_nid, intent_nid, etype="contains",
               label="contains", weight=ont["severity"] / 5.0, color=EDGE_COLORS["contains"])

    # ── Primary intent → MITRE node ───────────────────────────
    mitre_nid = _nid_mitre(primary)
    _add_node(G, mitre_nid, label=ont["mitre_tactic"][:30], kind="MITRE",
              color=NODE_COLORS["MITRE"], value=0.0,
              mitre_technique=ont["mitre_technique"])
    G.add_edge(intent_nid, mitre_nid, etype="mapped_to",
               label="mapped_to", weight=1.0, color=EDGE_COLORS["mapped_to"])

    # ── FIX F9: secondary intents → individual MITRE nodes ────
    # Each secondary intent that has a real ATT&CK entry (i.e. not
    # "N/A - Legitimate Activity" or "N/A - Unclassified") receives
    # its own MITRE node.  The edge connects the SECONDARY node
    # (already in the graph from _add_intent_core) to the MITRE
    # node, using the same mapped_to edge type as the primary.
    for sec in s2["intent_classification"].get("secondary_intents", [])[:5]:
        sec_label  = sec["label"]
        sec_ont    = _get_ont(sec_label)
        # Skip if the tactic is N/A (legitimate or unclassified)
        if sec_ont["mitre_tactic"].startswith("N/A"):
            continue
        sec_mitre_nid = _nid_mitre(sec_label)
        # Node may already exist if the same label appeared as a
        # sibling of the primary — merge attributes via _add_node.
        _add_node(G, sec_mitre_nid,
                  label=sec_ont["mitre_tactic"][:30],
                  kind="MITRE",
                  color=NODE_COLORS["MITRE"],
                  value=0.0,
                  mitre_technique=sec_ont["mitre_technique"],
                  secondary_for=sec_label)
        # Connect from the SECONDARY node already placed in
        # _add_intent_core, falling back to the intent node if
        # the secondary node was not created for some reason.
        sec_src_nid = _nid_secondary(sec_label)
        if not G.has_node(sec_src_nid):
            sec_src_nid = intent_nid
        G.add_edge(sec_src_nid, sec_mitre_nid, etype="mapped_to",
                   label="mapped_to",
                   weight=round(sec["score"], 3),
                   color=EDGE_COLORS["mapped_to"])
    # ──────────────────────────────────────────────────────────

    for sibling in ont.get("siblings", [])[:3]:
        sib_ont = _get_ont(sibling)
        sib_nid = f"neighbor:{sibling}"
        if not G.has_node(sib_nid) and not G.has_node(_nid_intent(sibling)):
            _add_node(G, sib_nid, label=sibling, kind="NEIGHBOR",
                       color=NODE_COLORS["NEIGHBOR"], value=0.0,
                       severity=sib_ont["severity"])
        target_nid = _nid_intent(sibling) if G.has_node(_nid_intent(sibling)) else sib_nid
        G.add_edge(intent_nid, target_nid, etype="related_to",
                   label="sibling", weight=0.3, color=EDGE_COLORS["related_to"])


def _add_similarity_layer(G: nx.MultiDiGraph, s2: dict, intent_nid: str):
    """Adds semantic similarity nodes (formerly Graph 4) and,
    critically (FIX F5), writes the similarity score onto each
    node so it survives serialization and can be used downstream
    for disambiguation — not just rendered in a PNG and discarded.
    Returns the sorted similarity list for ir_input_summary.

    FIX F11: the "unknown" intent label is now included in the
    similarity corpus.  Previously it was skipped (if label ==
    "unknown": continue), which meant a prompt genuinely matching
    no known category was forced to rank against the 9 known
    intents and could receive a misleadingly high similarity score.
    With "unknown" in the corpus, a truly ambiguous prompt will
    score highest against the "unknown" reference text and the
    disambiguation block will correctly flag it.
    """
    prompt_text = s2["normalized_text"]
    primary     = s2["intent_classification"]["primary_intent"]

    # FIX F11: include ALL labels, including "unknown"
    corpus, corpus_labels = [], []
    for label, data in THREAT_ONTOLOGY.items():
        corpus.append(data["reference_text"])
        corpus_labels.append(label)

    if SBERT_AVAILABLE and _sbert is not None:
        all_texts  = [prompt_text] + corpus
        embeddings = _sbert.encode(all_texts, convert_to_numpy=True)
        sims       = cosine_similarity(embeddings[0:1], embeddings[1:])[0]
        method     = "sbert"
    else:
        all_texts  = [prompt_text] + corpus
        vec        = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        mat        = vec.fit_transform(all_texts)
        sims       = cosine_similarity(mat[0:1], mat[1:])[0]
        method     = "tfidf"

    _add_node(G, NID_PROMPT, label=f"PROMPT: {prompt_text[:30]}...",
              kind="PROMPT", color=NODE_COLORS["PROMPT"], value=1.0)

    G.add_edge(intent_nid, NID_PROMPT, etype="related_to",
               label="analyzed_as", weight=1.0, color=EDGE_COLORS["related_to"])

    sim_results = []
    for label, sim in zip(corpus_labels, sims):
        sim = float(sim)
        sim_results.append({"label": label, "similarity": round(sim, 4)})
        sid = _nid_similarity(label)
        is_primary = (label == primary)
        _add_node(G, sid,
                  label=f"{label} ({sim:.3f})",
                  kind="SIMILARITY",
                  color=NODE_COLORS["INTENT"] if is_primary else NODE_COLORS["SIMILARITY"],
                  similarity=sim,            # FIX F5 — score persists on the node
                  similarity_method=method,
                  value=sim)

        G.add_edge(NID_PROMPT, sid, etype="similar_to",
                   label=f"{sim:.3f}", weight=sim, color=EDGE_COLORS["similar_to"])
        # Tie the similarity stub back to the real intent node when it
        # is the primary, so the IR has one connected component (F1).
        if is_primary:
            G.add_edge(sid, intent_nid, etype="related_to",
                       label="is_primary_intent", weight=1.0, color=EDGE_COLORS["related_to"])

    sim_results.sort(key=lambda x: x["similarity"], reverse=True)
    return sim_results, method


def _add_feature_layer(G: nx.MultiDiGraph, s2: dict, intent_nid: str):
    """Adds the 15-dim feature vector nodes (formerly Graph 5),
    connected directly to the canonical intent node (FIX F1)."""
    risk = s2["risk_assessment"]
    ent  = s2["entities"]
    s1   = s2["source_payload"]
    beh  = s2["behavioral_signals"]
    nlp  = s2["nlp_analysis"]
    ont  = _get_ont(s2["intent_classification"]["primary_intent"])

    cplx_map = {"low": 0.0, "medium": 0.5, "high": 1.0}
    cplx_val = cplx_map.get(s1["structure_analysis"]["complexity"], 0.0)

    features = [
        ("severity",            (ont["severity"] - 1) / 4.0,                 "risk_score"),
        ("risk_score",          risk["risk_score"],                          "risk_score"),
        ("has_ip",              1.0 if ent.get("ip_address") else 0.0,       "entity_presence"),
        ("has_url",             1.0 if ent.get("url") else 0.0,              "entity_presence"),
        ("has_filepath",        1.0 if ent.get("file_path") else 0.0,        "entity_presence"),
        ("threat_amplifiers",   min(len(ent.get("threat_amplifiers", [])) / 5, 1.0), "behavioral"),
        ("action_verb_density", min(len(beh.get("action_verbs", [])) / 5, 1.0),      "behavioral"),
        ("target_density",      min(len(beh.get("targeted_domains", [])) / 6, 1.0),  "behavioral"),
        ("temporal_signal",     1.0 if s1["temporal_analysis"]["has_temporal"] else 0.0, "structural"),
        ("is_compound",         1.0 if s1["structure_analysis"]["is_compound"] else 0.0, "structural"),
        ("complexity",          cplx_val,                                            "structural"),
        ("multi_target",        1.0 if s1["target_analysis"]["multi_target"] else 0.0, "structural"),
        ("svo_density",         min(len(nlp.get("dep_triples", [])) / 5, 1.0),       "behavioral"),
        ("concept_density",     min(s1["tokenization"]["unique_token_count"] /
                                     max(s1["tokenization"]["token_count"], 1), 1.0), "structural"),
        ("sensitivity",         1.0 if s1["temporal_analysis"]["has_sensitivity"] else 0.0, "behavioral"),
    ]

    for fname, fval, fcat in features:
        color = ("#e74c3c" if fval > 0.6 else "#f39c12" if fval > 0.3 else "#2ecc71")
        fid = _nid_feature(fname)
        _add_node(G, fid, label=f"{fname}: {fval:.2f}", kind="FEATURE",
                  color=color, value=fval, semantic_category=fcat)

        G.add_edge(intent_nid, fid, etype="has_feature",
                   label=f"{fval:.2f}", weight=max(fval, 0.01),
                   color=EDGE_COLORS["has_feature"])

    return {fname: fval for fname, fval, _ in features}


# ─────────────────────────────────────────────────────────
#  GRAPH ANALYSIS
# ─────────────────────────────────────────────────────────

def analyse_graph(G: nx.MultiDiGraph) -> dict:
    """Computes structural statistics for the whole unified graph."""
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    try:
        centrality = nx.degree_centrality(G)
        top_nodes  = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
    except Exception:
        top_nodes = []
    try:
        density = nx.density(G)
    except Exception:
        density = 0.0
    try:
        n_components = nx.number_weakly_connected_components(G)
    except Exception:
        n_components = None

    return {
        "node_count":            n_nodes,
        "edge_count":             n_edges,
        "density":                round(density, 4),
        "top_central_nodes":      top_nodes,
        "weakly_connected_components": n_components,
    }


# ─────────────────────────────────────────────────────────
#  SERIALIZATION  (FIX F6 — JSON-safe, no live nx objects)
# ─────────────────────────────────────────────────────────

def serialize_ir_graph(G: nx.MultiDiGraph) -> dict:
    """
    Converts the unified NetworkX graph into a plain
    JSON-serializable dict: nodes + edges only, plain Python
    types throughout. This is the object Stage 4 (AST
    capability extraction / threat modeling) should consume —
    never the live NetworkX object.
    """
    nodes = []
    for nid, data in G.nodes(data=True):
        nodes.append({
            "id":                str(nid),
            "label":             data.get("label", str(nid)),
            "kind":              data.get("kind", "UNKNOWN"),
            "color":             data.get("color", "#95a5a6"),
            "value":             round(float(data.get("value", 0.0)), 4),
            "provenance":        data.get("provenance"),          # F2
            "semantic_category": data.get("semantic_category"),
            "severity":          data.get("severity"),
            "similarity":        data.get("similarity"),          # F5
        })

    edges = []
    for src, tgt, data in G.edges(data=True):
        edges.append({
            "source": str(src),
            "target": str(tgt),
            "type":   data.get("etype", "related"),
            "label":  data.get("label", ""),
            "weight": round(float(data.get("weight", 1.0)), 4),
        })

    return {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes":      nodes,
        "edges":      edges,
        "analysis":   analyse_graph(G),
    }


def save_ir_graph_json(ir_ser: dict, output_path: str = "semantic_ir_graph.json") -> str:
    """
    Writes the JSON-serializable unified graph (nodes + edges +
    analysis) to an actual .json file on disk. This is the file
    Stage 4 (or any other downstream program) should read to
    consume the IR — plain JSON, no NetworkX objects, no Python-
    specific types.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(ir_ser, f, indent=2, ensure_ascii=False)
    print(f"[Stage 3] IR graph JSON saved -> {output_path}")
    return output_path


def save_ir_summary_json(ir_input_summary: dict, output_path: str = "semantic_ir_summary.json") -> str:
    """
    Writes the compact Stage-4 handoff dict (primary intent, risk,
    MITRE mapping, disambiguation, 15-dim feature vector — NOT the
    full node/edge graph) to its own .json file. Kept separate from
    the full graph JSON because Stage 4 may want the lightweight
    summary without parsing 50+ nodes every time, e.g. for a quick
    risk-gate check before deciding whether to load the full graph.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(ir_input_summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"[Stage 3] IR summary JSON saved -> {output_path}")
    return output_path


# ─────────────────────────────────────────────────────────
#  VISUALIZATION
#  Two renderers are provided:
#    1. visualize_ir_graph_single()  -> THE actual unified IR,
#       all nodes/edges on ONE canvas. This is the primary
#       output and what "one final IR" means visually.
#    2. visualize_ir_graph_views()   -> the same graph filtered
#       into 5 readable sub-views, kept ONLY as a secondary
#       diagnostic aid (e.g. to zoom into the role layer
#       without 50 nodes cluttering the picture). It is no
#       longer presented as "the" output.
# ─────────────────────────────────────────────────────────

_VIEW_KINDS = {
    "role_view":       {"ROLE", "ACTION", "TARGET", "METHOD", "DESTINATION",
                         "GOAL", "ENTITY", "INTENT", "SECONDARY"},
    "concept_view":     {"CONCEPT", "ENTITY", "ACTION", "AMPLIFIER", "INTENT"},
    "ontology_view":    {"ONTOLOGY_ROOT", "CATEGORY", "MITRE", "NEIGHBOR", "INTENT", "SECONDARY"},
    "similarity_view":  {"PROMPT", "SIMILARITY", "INTENT"},
    "feature_view":     {"FEATURE", "INTENT"},
}

# ─────────────────────────────────────────────────────────
#  WHY THE SINGLE-CANVAS GRAPH WAS UNREADABLE (root cause)
#  ------------------------------------------------------
#  FEATURE and SIMILARITY nodes are not relationships — they are
#  scalar measurements ("risk_score = 1.00", "similarity to
#  reconnaissance = 0.000"). Drawing 15 feature values and 9
#  similarity scores as graph nodes connected by edges to INTENT
#  turns 24 numbers into 24 circles + 24 lines, which looks like
#  structure but carries no relational meaning beyond "this number
#  belongs to this intent" — something a table says better than a
#  graph ever can.

# ─────────────────────────────────────────────────────────

RELATIONAL_KINDS = {
    "INTENT", "SECONDARY", "ROLE", "ACTION", "TARGET", "METHOD",
    "DESTINATION", "GOAL", "CONCEPT", "ENTITY", "AMPLIFIER",
    "ONTOLOGY_ROOT", "CATEGORY", "MITRE", "NEIGHBOR",
}
DATA_KINDS = {"FEATURE", "SIMILARITY"}
# PROMPT is metadata (the input text), shown in the header, not graphed.
METADATA_KINDS = {"PROMPT"}


_KIND_LANE_ORDER = [
    "ROLE", "ACTION", "TARGET", "METHOD", "DESTINATION", "GOAL",
    "ENTITY", "CONCEPT", "AMPLIFIER",
    "ONTOLOGY_ROOT", "CATEGORY", "MITRE", "NEIGHBOR",
    "SECONDARY",
]


def _subgraph_for_view(G: nx.MultiDiGraph, view: str) -> nx.MultiDiGraph:
    kinds = _VIEW_KINDS[view]
    keep  = [n for n, d in G.nodes(data=True) if d.get("kind") in kinds]
    return G.subgraph(keep)


def _relational_subgraph(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Returns only the nodes that represent real relationships
    (WHO/WHAT/WHERE structure, concepts, ontology position) — the
    part of the IR that is actually meaningful to draw as a graph.
    FEATURE and SIMILARITY scalar nodes are excluded; they are
    rendered as bar-chart panels instead (see DATA_KINDS above)."""
    keep = [n for n, d in G.nodes(data=True) if d.get("kind") in RELATIONAL_KINDS]
    return G.subgraph(keep)


def _kind_clustered_layout(G: nx.MultiDiGraph, intent_nid: str) -> dict:
    """
    Lays out the RELATIONAL subgraph (role/concept/ontology nodes —
    see RELATIONAL_KINDS) clustered by node kind around the central
    intent node, instead of running a generic spring layout (which
    produces an unreadable hairball as node count grows). Nodes of
    the same kind are grouped into one angular wedge of the circle
    so the eye can still parse structure.
    """
    pos = {intent_nid: np.array([0.0, 0.0])}

    # Group remaining nodes by kind, preserving _KIND_LANE_ORDER
    by_kind = {}
    for n, d in G.nodes(data=True):
        if n == intent_nid:
            continue
        by_kind.setdefault(d.get("kind", "OTHER"), []).append(n)

    present_kinds = [k for k in _KIND_LANE_ORDER if k in by_kind]
    for k in by_kind:
        if k not in present_kinds:
            present_kinds.append(k)

    n_lanes = max(len(present_kinds), 1)
    lane_width = 2 * np.pi / n_lanes

    for lane_idx, kind in enumerate(present_kinds):
        nodes_in_kind = by_kind[kind]
        center_angle  = lane_idx * lane_width
        n_in_lane     = len(nodes_in_kind)
        # spread nodes within their lane across a fraction of the
        # lane width so same-kind nodes don't overlap each other
        spread = lane_width * 0.8
        for j, n in enumerate(nodes_in_kind):
            if n_in_lane == 1:
                angle = center_angle
            else:
                angle = center_angle - spread / 2 + spread * j / (n_in_lane - 1)
            # vary radius slightly by index so labels don't all
            # collide on the same ring
            radius = 3.0 + 0.6 * (j % 3)
            pos[n] = np.array([radius * np.cos(angle), radius * np.sin(angle)])

    return pos


def visualize_ir_graph_single(G: nx.MultiDiGraph, payload_meta: dict,
                               output_path: str = "semantic_ir_unified.png") -> str:
    """
    THE primary visualization. Three things are shown in ONE image
    file, but as three clearly-separated panels rather than one
    big tangled graph:

      LEFT  : the RELATIONAL graph — WHO does WHAT to WHAT, sent
              WHERE, why (GOAL), and where this sits in the threat
              ontology (MITRE/category/siblings). This is the part
              that is actually graph-shaped.
      TOP RIGHT   : the 15-dim FEATURE VECTOR as a sorted horizontal
              bar chart — these are scalar measurements, not
              relationships, so a bar chart reads faster than nodes.
      BOTTOM RIGHT: the SIMILARITY-TO-OTHER-INTENTS ranking as a
              sorted horizontal bar chart, with the actual
              classifier's primary intent marked.

    All three panels are still built from the SAME underlying
    unified graph (G) and the SAME JSON export — nothing in the
    data model changed, only how FEATURE/SIMILARITY scalars are
    rendered (panel instead of graph node), per the rationale
    above DATA_KINDS.
    """
    RG = _relational_subgraph(G)
    intent_nid = next((n for n, d in RG.nodes(data=True) if d.get("kind") == "INTENT"), None)
    pos = _kind_clustered_layout(RG, intent_nid) if intent_nid else nx.spring_layout(RG, seed=42, k=1.2)

    feature_nodes = sorted(
        [(d["label"].split(":")[0], float(d.get("value", 0.0)))
         for n, d in G.nodes(data=True) if d.get("kind") == "FEATURE"],
        key=lambda x: x[1], reverse=True,
    )
    similarity_nodes = sorted(
        [(n.replace("similarity:", ""), float(d.get("similarity", d.get("value", 0.0))))
         for n, d in G.nodes(data=True) if d.get("kind") == "SIMILARITY"],
        key=lambda x: x[1], reverse=True,
    )

    fig = plt.figure(figsize=(24, 14))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 1], height_ratios=[1, 1],
                           hspace=0.32, wspace=0.18)

    fig.suptitle(
        f"Unified Semantic IR  |  Intent: {payload_meta['primary_intent'].upper()}  "
        f"|  Risk: {payload_meta['risk_level']} ({payload_meta['risk_score']})  "
        f"|  Graph nodes: {RG.number_of_nodes()}  edges: {RG.number_of_edges()}  "
        f"|  Total IR nodes (incl. feature/similarity panels): {G.number_of_nodes()}",
        fontsize=14, fontweight="bold", y=0.995
    )

    # ── LEFT PANEL (spans both rows): relational graph ─────────
    ax_graph = fig.add_subplot(gs[:, 0])

    node_colors = [RG.nodes[n].get("color", "#95a5a6") for n in RG.nodes]
    node_sizes  = []
    for n in RG.nodes:
        kind = RG.nodes[n].get("kind", "")
        if kind == "INTENT":                              node_sizes.append(3000)
        elif kind in ("ACTION", "TARGET", "DESTINATION"):  node_sizes.append(1400)
        elif kind == "SECONDARY":                          node_sizes.append(1000)
        elif kind == "ENTITY":                             node_sizes.append(900)
        else:                                              node_sizes.append(750)

    nx.draw_networkx_nodes(RG, pos, ax=ax_graph, node_color=node_colors,
                            node_size=node_sizes, alpha=0.92, linewidths=1.2,
                            edgecolors="white")

    edge_colors = [d.get("color", "#bdc3c7") for _, _, d in RG.edges(data=True)]
    edge_widths = [0.6 + 1.8 * float(d.get("weight", 0.3)) for _, _, d in RG.edges(data=True)]
    nx.draw_networkx_edges(RG, pos, ax=ax_graph, edge_color=edge_colors,
                            width=edge_widths, alpha=0.55, arrows=True,
                            arrowsize=9, connectionstyle="arc3,rad=0.08")

    for n, (x, y) in pos.items():
        label = RG.nodes[n].get("label", str(n))
        is_intent = (n == intent_nid)
        ax_graph.annotate(
            label,
            xy=(x, y), xytext=(0, 0 if is_intent else 9),
            textcoords="offset points",
            ha="center", va="center",
            fontsize=10 if is_intent else 7.2,
            fontweight="bold" if is_intent else "normal",
            color="white" if is_intent else "#222222",
            bbox=dict(boxstyle="round,pad=0.15",
                      facecolor=RG.nodes[n].get("color", "#95a5a6") if is_intent else "white",
                      edgecolor="none" if is_intent else "#cccccc",
                      alpha=0.95 if is_intent else 0.85),
        )

    ax_graph.set_title(
        "RELATIONAL GRAPH — who does what, to what, sent where, and why\n"
        "(role chain + concepts + threat-ontology position)",
        fontsize=11, fontweight="bold", color="#222222", pad=10,
    )
    ax_graph.axis("off")

    legend_kinds = [k for k in NODE_COLORS if k in RELATIONAL_KINDS]
    legend_items = [mpatches.Patch(facecolor=NODE_COLORS[k], label=k) for k in legend_kinds]
    ax_graph.legend(handles=legend_items, loc="upper left", bbox_to_anchor=(0.0, 0.02),
                     fontsize=8, title="Node kinds (graph)", title_fontsize=8.5,
                     framealpha=0.95, ncol=2)

    # ── TOP RIGHT PANEL: feature vector bar chart ───────────────
    ax_feat = fig.add_subplot(gs[0, 1])
    if feature_nodes:
        names, vals = zip(*feature_nodes)
        y_pos = np.arange(len(names))
        colors = ["#e74c3c" if v > 0.6 else "#f39c12" if v > 0.3 else "#2ecc71" for v in vals]
        ax_feat.barh(y_pos, vals, color=colors, alpha=0.9, height=0.65)
        ax_feat.set_yticks(y_pos)
        ax_feat.set_yticklabels(names, fontsize=8)
        ax_feat.invert_yaxis()
        ax_feat.set_xlim(0, 1.05)
        for i, v in enumerate(vals):
            ax_feat.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=7.5, color="#333333")
    ax_feat.set_title("FEATURE VECTOR — 15 scalar signals (not relationships, so shown as bars)",
                       fontsize=10, fontweight="bold", pad=8)
    ax_feat.set_xlabel("value (0.0 - 1.0)", fontsize=8)
    for spine in ("top", "right"):
        ax_feat.spines[spine].set_visible(False)

    # ── BOTTOM RIGHT PANEL: similarity ranking bar chart ────────
    ax_sim = fig.add_subplot(gs[1, 1])
    if similarity_nodes:
        names, vals = zip(*similarity_nodes)
        y_pos = np.arange(len(names))
        primary = payload_meta["primary_intent"]
        colors = ["#e74c3c" if n == primary else "#16a085" for n in names]
        ax_sim.barh(y_pos, vals, color=colors, alpha=0.9, height=0.65)
        ax_sim.set_yticks(y_pos)
        ax_sim.set_yticklabels(names, fontsize=8)
        ax_sim.invert_yaxis()
        max_v = max(vals) if vals else 1.0
        ax_sim.set_xlim(0, max(max_v * 1.25, 0.05))
        for i, v in enumerate(vals):
            ax_sim.text(v + max_v * 0.03, i, f"{v:.3f}", va="center", fontsize=7.5, color="#333333")
    ax_sim.set_title("SIMILARITY TO EACH INTENT — text-similarity ranking (red = classifier's pick)",
                      fontsize=10, fontweight="bold", pad=8)
    ax_sim.set_xlabel("cosine similarity to prompt", fontsize=8)
    for spine in ("top", "right"):
        ax_sim.spines[spine].set_visible(False)

    # ── How-to-read strip at the very bottom ────────────────────
    fig.text(
        0.5, 0.005,
        "HOW TO READ THIS: left = relationships (graph) · top-right = how strongly each "
        "behavioral signal fired · bottom-right = how closely the prompt's wording matches "
        "each of the 10 known intent categories.",
        ha="center", fontsize=9.5, color="#555555", style="italic",
    )

    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[Stage 3] Unified IR visualization saved -> {output_path}")
    return output_path


def visualize_ir_graph_views(G: nx.MultiDiGraph, payload_meta: dict,
                              output_path: str = "semantic_ir_views.png") -> str:
    """
    SECONDARY diagnostic only: renders the same unified graph
    filtered into 5 readable sub-views (role / concept / ontology /
    similarity / feature) for closer inspection of one layer at a
    time. This is NOT the IR itself — visualize_ir_graph_single()
    is. Kept because zooming into one layer is sometimes useful,
    but it must never be the only or primary visualization, since
    presenting 5 boxes as "the output" is exactly what looked like
    5 disconnected graphs before.
    """
    fig = plt.figure(figsize=(22, 16))
    fig.suptitle(
        f"Unified Semantic IR — Filtered Views (diagnostic only)  |  Intent: "
        f"{payload_meta['primary_intent'].upper()}  |  Risk: {payload_meta['risk_level']}  "
        f"|  Score: {payload_meta['risk_score']}  |  Total IR nodes: {G.number_of_nodes()}  "
        f"edges: {G.number_of_edges()}",
        fontsize=12, fontweight="bold", y=0.99
    )

    view_titles = [
        ("role_view",      "1. Role View\n(WHO->ACTION->TARGET->DEST)", 231),
        ("concept_view",   "2. Concept View\n(Intent + Concepts)",       232),
        ("ontology_view",  "3. Ontology View\n(MITRE ATT&CK Position)",  233),
        ("similarity_view","4. Similarity View\n(Prompt vs Threats)",     234),
        ("feature_view",   "5. Feature View\n(15-dim Semantic Vector)",   235),
    ]

    for view_key, title, subplot_id in view_titles:
        ax = fig.add_subplot(subplot_id)
        SG = _subgraph_for_view(G, view_key)

        try:
            if SG.number_of_nodes() <= 6:
                pos = nx.planar_layout(SG)
            elif SG.number_of_nodes() <= 12:
                pos = nx.kamada_kawai_layout(SG)
            else:
                pos = nx.spring_layout(SG, seed=42, k=1.5)
        except Exception:
            pos = nx.spring_layout(SG, seed=42)

        node_colors = [SG.nodes[n].get("color", "#95a5a6") for n in SG.nodes]
        node_sizes  = []
        for n in SG.nodes:
            kind = SG.nodes[n].get("kind", "")
            if kind == "INTENT":                              node_sizes.append(2200)
            elif kind in ("ACTION", "TARGET", "DESTINATION"):  node_sizes.append(1600)
            elif kind == "SECONDARY":                          node_sizes.append(900)
            elif kind == "ENTITY":                             node_sizes.append(1200)
            else:                                              node_sizes.append(1000)

        nx.draw_networkx_nodes(SG, pos, ax=ax, node_color=node_colors,
                                node_size=node_sizes, alpha=0.9)
        # Full labels with small font instead of truncated text —
        # readability fix carried over from the single-canvas view.
        labels = {n: SG.nodes[n].get("label", n) for n in SG.nodes}
        nx.draw_networkx_labels(SG, pos, labels, ax=ax, font_size=5.2,
                                 font_color="white", font_weight="bold")

        edge_colors = [d.get("color", "#bdc3c7") for _, _, d in SG.edges(data=True)]
        nx.draw_networkx_edges(SG, pos, ax=ax, edge_color=edge_colors,
                                arrows=True, arrowsize=12, width=1.5, alpha=0.7,
                                connectionstyle="arc3,rad=0.1")

        edge_labels = {(u, v): d.get("label", "") for u, v, d in SG.edges(data=True) if d.get("label", "")}
        try:
            nx.draw_networkx_edge_labels(SG, pos, edge_labels, ax=ax, font_size=4.5, alpha=0.8)
        except Exception:
            pass

        ax.set_title(title, fontsize=8, fontweight="bold", pad=6)
        ax.axis("off")
        ax.text(0.01, 0.01, f"Nodes: {SG.number_of_nodes()}  Edges: {SG.number_of_edges()}",
                transform=ax.transAxes, fontsize=6, color="grey", verticalalignment="bottom")

    ax6 = fig.add_subplot(236)
    ax6.axis("off")
    legend_items = [mpatches.Patch(facecolor=c, label=f"Node: {k}") for k, c in NODE_COLORS.items()]
    ax6.legend(handles=legend_items[:14], loc="upper left", fontsize=6.5,
               title="Node Kinds", title_fontsize=7, framealpha=0.9, ncol=1)
    ax6.set_title("6. Legend", fontsize=8, fontweight="bold")

    ir_text = (
        "DIAGNOSTIC VIEW ONLY\n"
        "──────────────────────────\n"
        f"Total nodes : {G.number_of_nodes()}\n"
        f"Total edges : {G.number_of_edges()}\n"
        f"Primary     : {payload_meta['primary_intent']}\n"
        f"Risk        : {payload_meta['risk_level']} ({payload_meta['risk_score']})\n"
        "──────────────────────────\n"
        "For the actual unified\n"
        "IR as ONE graph, see\n"
        "the *_unified.png file.\n"
        "This file is a zoomed-in\n"
        "diagnostic split by kind."
    )
    ax6.text(0.02, 0.30, ir_text, transform=ax6.transAxes, fontsize=7,
              fontfamily="monospace",
              bbox=dict(boxstyle="round", facecolor="#fff3e0", alpha=0.9))

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[Stage 3] Filtered-views diagnostic saved -> {output_path}")
    return output_path


# ─────────────────────────────────────────────────────────
#  MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────

def build_semantic_representation(
    stage2_payload: dict,
    save_visualization: bool = True,
    output_path: str = "semantic_ir_unified.png",
    save_views_diagnostic: bool = False,
    views_output_path: str = "semantic_ir_views.png",
    save_json: bool = True,
    json_graph_path: str = "semantic_ir_graph.json",
    json_summary_path: str = "semantic_ir_summary.json",
) -> dict:
    """
    Stage 3 public API — UNIFIED Semantic IR.

    Parameters
    ----------
    stage2_payload         : dict — IntentPayload from Stage 2
    save_visualization      : bool — save the PRIMARY single-canvas
                               unified graph PNG if True (default).
                               This is the actual "one final IR"
                               picture: every node and edge on one
                               canvas, clustered by kind.
    output_path             : str — path for the primary PNG.
    save_views_diagnostic   : bool — additionally save the 5-panel
                               filtered-views PNG. Off by default —
                               it is a secondary diagnostic, not
                               the IR itself, and should not be
                               mistaken for "the" output again.
    views_output_path       : str — path for the diagnostic PNG.
    save_json                : bool — write the IR to real .json
                               files on disk if True (default). This
                               is the actual machine-readable output
                               that Stage 4 / any other program
                               should consume — not the PNG.
    json_graph_path          : str — path for the full graph JSON
                               (all nodes + edges + analysis stats).
    json_summary_path        : str — path for the compact summary
                               JSON (intent, risk, MITRE mapping,
                               disambiguation, feature vector — the
                               lightweight handoff object).

    Returns
    -------
    dict — SemanticPayload containing:
      - "ir_graph"          : JSON-serializable unified graph (F6)
      - "ir_graph_networkx" : live nx.MultiDiGraph, IN-PROCESS ONLY,
                               clearly named so it is never mistaken
                               for the serializable export (F6)
      - "ir_input_summary"  : compact handoff dict for Stage 4 (F4, F5)
      - "visualization_path"          : path to the single-canvas PNG
      - "visualization_views_path"    : path to the diagnostic PNG
                                          (None unless requested)
      - "json_graph_path"             : path to the full graph .json
                                          on disk (None unless saved)
      - "json_summary_path"           : path to the summary .json
                                          on disk (None unless saved)
    """
    primary = stage2_payload["intent_classification"]["primary_intent"]
    risk    = stage2_payload["risk_assessment"]["risk_level"]
    score   = stage2_payload["risk_assessment"]["risk_score"]

    print(f"\n[Stage 3] Building UNIFIED semantic IR graph...")
    print(f"          intent : {primary}")
    print(f"          risk   : {risk} ({score})")

    G = nx.MultiDiGraph(graph_type="unified_semantic_ir")

    intent_nid = _add_intent_core(G, stage2_payload)
    print(f"[Stage 3] Core intent node added.")

    _add_role_layer(G, stage2_payload, intent_nid)
    print(f"[Stage 3] Role layer merged into IR.")

    _add_concept_layer(G, stage2_payload, intent_nid)
    print(f"[Stage 3] Concept layer merged into IR.")

    _add_ontology_layer(G, stage2_payload, intent_nid)
    print(f"[Stage 3] Ontology layer merged into IR.")

    sim_results, sim_method = _add_similarity_layer(G, stage2_payload, intent_nid)
    print(f"[Stage 3] Similarity layer merged into IR (method={sim_method}).")

    feature_values = _add_feature_layer(G, stage2_payload, intent_nid)
    print(f"[Stage 3] Feature layer merged into IR.")

    ir_ser = serialize_ir_graph(G)
    print(f"[Stage 3] Unified IR graph: {ir_ser['node_count']} nodes, {ir_ser['edge_count']} edges.")

    ont_data = _get_ont(primary)


    # similarity ranking forward instead of dead-ending in a PNG.
    top_sim   = sim_results[0] if sim_results else None
    runner_up = sim_results[1] if len(sim_results) > 1 else None
    disambiguation = {
        "method":               sim_method,
        "ranked_similarities":  sim_results,
        "top_match":            top_sim,
        "agrees_with_classifier": bool(top_sim and top_sim["label"] == primary),
        "margin_to_runner_up":  round(top_sim["similarity"] - runner_up["similarity"], 4)
                                  if top_sim and runner_up else None,
    }

    payload = {
        "payload_version":  "2.2",
        "built_at":         datetime.utcnow().isoformat() + "Z",
        "input_type":       "prompt",
        "primary_intent":   primary,
        "risk_level":       risk,
        "risk_score":       score,

        
        "ir_graph": ir_ser,

      
        "ir_graph_networkx": G,

        # FIX F4 — totals reflect the WHOLE unified graph, not an
        # arbitrary 3-of-5 sum.
        # FIX F5 — disambiguation block carries similarity forward.
        # FIX F12 — classifier_conflict flag forwarded from Stage 2
        #            so Stage 4 does not have to reach into the raw
        #            stage2_payload to find it.
        "ir_input_summary": {
            "total_nodes":          ir_ser["node_count"],
            "total_edges":          ir_ser["edge_count"],
            "primary_intent":       primary,
            "secondary_intents":    stage2_payload["intent_classification"].get("secondary_intents", []),
            "risk_level":           risk,
            "risk_score":           score,
            "mitre_tactic":         ont_data["mitre_tactic"],
            "mitre_technique":      ont_data["mitre_technique"],
            "severity":             ont_data["severity"],
            "disambiguation":       disambiguation,
            "feature_vector":       feature_values,
            # FIX F12: classifier conflict forwarded from Stage 2
            "classifier_conflict":  stage2_payload["intent_classification"].get(
                                        "classifier_conflict", False),
            "conflict_detail":      stage2_payload["intent_classification"].get(
                                        "conflict_detail", None),
        },

        "original_prompt":   stage2_payload["normalized_text"],
        "stage2_payload":    stage2_payload,
        "visualization_path":        None,
        "visualization_views_path":  None,
        "json_graph_path":            None,
        "json_summary_path":          None,
    }

    if save_visualization:
        payload["visualization_path"] = visualize_ir_graph_single(
            G, payload, output_path
        )

    if save_views_diagnostic:
        payload["visualization_views_path"] = visualize_ir_graph_views(
            G, payload, views_output_path
        )

    if save_json:
        payload["json_graph_path"] = save_ir_graph_json(ir_ser, json_graph_path)
        payload["json_summary_path"] = save_ir_summary_json(
            payload["ir_input_summary"], json_summary_path
        )

    print(f"Done.")
    return payload