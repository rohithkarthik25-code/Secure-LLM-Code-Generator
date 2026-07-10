"""
=============================================================
  STAGE 2 — PROMPT INTENT EXTRACTION
  Behavioral Interpretation Framework
=============================================================
  INPUT  : PromptPayload from Stage 1 (prompt_input.py)
  OUTPUT : IntentPayload with intent, risk, entities

  References:
  [1] Parikh et al. (2023) "Exploring Zero and Few-shot
      Techniques for Intent Classification", arXiv:2305.07157
      → Zero-shot + rule-based hybrid classification approach
  [2] Liu et al. (2022) "Multi-features based Semantic
      Augmentation Networks for NER in Threat Intelligence"
      arXiv:2207.00232
      → Multi-feature NER for cybersecurity entity extraction
  [3] Hanks et al. (2022) "Recognizing and Extracting
      Cybersecurity-relevant Entities from Text", ICML 2022
      → Cybersecurity-specific entity type taxonomy
  [4] Mimura & Ito (2021) "Applying NLP Techniques to
      Malware Detection", Int. J. Information Security
      → Keyword-based threat signal detection in text
  [5] Threat Detection and Response Using AI and NLP
      in Cybersecurity, JISIS 2024
      → AI+NLP pipeline for behavioral threat analysis

  FIXES (v1.1):
  - target_categories in entities is now populated correctly
    from Stage 1's targets_found (was silently empty before).
  - secondary_intents list is preserved in intent_classification
    so compound malicious prompts (e.g. read /etc/shadow AND
    send to remote IP) no longer lose their secondary intent.
  - Compound-intent boosting: when Stage 1 reports multi_target
    and secondary_targets, matching secondary intents receive a
    score boost so they surface reliably in intent_scores.
  - merge_classifications now passes ALL labels through rather
    than only top-7, so secondary_intents captures the full set.

  FIXES (v1.2):
  [FIX-A] Compound-intent risk lift now accumulates across ALL
    qualifying secondary intents instead of stopping at the
    first one (removed the `break`). A cap of MAX_COMPOUND_LIFT
    (0.30) prevents runaway escalation when many secondaries fire.
  [FIX-B] merge_classifications() now emits a `classifier_conflict`
    flag when the rule-based top intent differs from the LLM top
    intent AND the score gap between them exceeds
    CONFLICT_THRESHOLD (0.30). This lets Stage 3 / Stage 4 know
    to treat the classification with lower confidence.
  [FIX-C] payload_version bumped to "1.2".

  FIXES (v1.3):
  [FIX-D] BART is REQUIRED. The pipeline raises a clear, human-
    readable error with install instructions if transformers/torch
    are missing or the model fails to load. Rule-based-only mode
    is not offered because it misclassifies malicious prompts.
  [FIX-E] Word-boundary matching in classify_intent_rules().
    Previously "kw in text_lower" did substring matching, so
    short keywords like "log", "key", "rce" could fire inside
    unrelated words. Now uses re.search(r"\\b...\\b").
  [FIX-F] IP boost reason string now reports the capped count
    (min(n,3)) rather than the raw count, so the printed
    explanation matches the actual boost applied.
  [FIX-G] payload_version bumped to "1.3".
=============================================================
  INSTALL (required):
      pip install transformers torch
      pip install spacy                      # optional, improves NER
      python -m spacy download en_core_web_sm

  facebook/bart-large-mnli (~1.6 GB) downloads automatically on
  first run. Internet access required that once. Without BART,
  Stage 2 will raise an error and refuse to run — this is
  intentional because rule-based-only mode misclassifies
  malicious prompts as "unknown".
=============================================================
"""

import re
import json
from datetime import datetime
from collections import defaultdict

# ── spaCy ─────────────────────────────────────────────────
# Stage 2 does NOT re-load spaCy. All NER/POS/dep results are
# pre-computed by Stage 1 (prompt_input.py) and stored in the
# Stage 1 payload under "spacy_analysis". Stage 2 reads these
# directly, eliminating duplicate model loading and inference.
#
# The NLP_MODEL handle is retained here only for the legacy
# deep_nlp_analysis() function which uses it as a fallback when
# Stage 1 spaCy results are absent (e.g. during unit testing
# where Stage 1 is mocked). In normal pipeline use, Stage 1
# always provides the spaCy results.
try:
    import spacy
    NLP_MODEL       = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
    print("[Stage 2] spaCy handle acquired (Stage 1 results used in pipeline).")
except Exception:
    NLP_MODEL       = None
    SPACY_AVAILABLE = False
    print("[Stage 2] spaCy not available — Stage 1 spaCy results required.")

# ── HuggingFace zero-shot (REQUIRED) ────────────────────
# Ref [1]: Parikh et al. use zero-shot LLM intent
# classification as one of four approaches explored.
# BART-large-MNLI is a required component of this pipeline.
# It provides semantic zero-shot NLI classification that the
# rule-based keyword classifier cannot replicate — especially
# for novel prompt phrasings. Without it, malicious prompts
# (e.g. "silently captures keystrokes and uploads remotely")
# fall through to "unknown" instead of "data_exfiltration".
#
# If not installed, run:
#     pip install transformers torch
#     (first run downloads facebook/bart-large-mnli ~1.6 GB)
try:
    from transformers import pipeline as hf_pipeline
except ImportError:
    # SOFT FALLBACK (was a hard raise): 'transformers' isn't installed.
    # Degrade to rule-based-only classification instead of crashing the
    # whole pipeline. classify_intent_llm() already wraps its call to
    # _zero_shot in a broad try/except and returns [] on failure, and
    # merge_classifications() already handles empty llm_results — so
    # setting _zero_shot=None here is enough to make that existing
    # fallback path actually reachable.
    hf_pipeline = None
    _zero_shot   = None
    HF_AVAILABLE = False
    print(
        "\n"
        "╔══════════════════════════════════════════════════════════╗\n"
        "║  [Stage 2] 'transformers' is not installed.              ║\n"
        "║  Falling back to rule-based-only classification.         ║\n"
        "║  (Novel/paraphrased malicious prompts may be missed —    ║\n"
        "║   rely more on Stage 1 keyword/target coverage.)         ║\n"
        "║                                                          ║\n"
        "║  To enable BART: pip install transformers torch          ║\n"
        "╚══════════════════════════════════════════════════════════╝"
    )

if hf_pipeline is not None:
    try:
        _zero_shot = hf_pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1          # CPU; change to 0 for GPU
        )
        HF_AVAILABLE = True
        print("[Stage 2] BART-large-MNLI zero-shot classifier loaded.")
    except Exception as _bart_err:
        # SOFT FALLBACK (was a hard raise): BART failed to load (no
        # internet on first run, low disk space, corrupted cache, etc).
        # Degrade to rule-based-only rather than crashing.
        _zero_shot   = None
        HF_AVAILABLE = False
        print(
            "\n"
            "╔══════════════════════════════════════════════════════════╗\n"
            "║  [Stage 2] BART-large-MNLI failed to load.               ║\n"
            "║  Falling back to rule-based-only classification.         ║\n"
            "║                                                          ║\n"
            f"║  Error: {str(_bart_err)[:48].ljust(48)} ║\n"
            "║                                                          ║\n"
            "║  Common causes:                                          ║\n"
            "║  1. No internet on first run (model needs to download)   ║\n"
            "║  2. Disk space too low (model needs ~1.6 GB)             ║\n"
            "║  3. torch not installed: pip install torch               ║\n"
            "║  4. HuggingFace cache corrupted — delete and retry:      ║\n"
            "║     rm -rf ~/.cache/huggingface/hub/models--facebook*    ║\n"
            "╚══════════════════════════════════════════════════════════╝"
        )


# ─────────────────────────────────────────────────────────
#  INTENT TAXONOMY
#  Ref [1]: Parikh et al. define intent as the primary NLU
#  task. Our 10-label taxonomy is aligned with security
#  behavioral categories from MITRE ATT&CK framework.
# ─────────────────────────────────────────────────────────

INTENT_LABELS = [
    # ── Original 10 ────────────────────────────────────────────────
    "data_exfiltration",
    "privilege_escalation",
    "network_communication",
    "file_system_access",
    "process_execution",
    "code_injection",
    "reconnaissance",
    "legitimate_automation",
    "system_monitoring",
    # ── New 7 (aligned with new target categories + capability layer) ──
    "input_capture",          # keylogging / clipboard / screenshot
    "persistence",            # autostart / registry run / backdoor
    "defense_evasion",        # AV bypass / log deletion / anti-debug
    "database_access",        # SQL queries / NoSQL / ORM
    "memory_manipulation",    # heap spray / buffer overflow / in-memory
    "cryptographic_operation",# encrypt/decrypt / ransomware / obfuscation
    "web_scraping",           # browser automation / HTML parsing / crawling
    "general_coding",         # benign utility / algorithm / helper code
    "unknown",
]

# ─────────────────────────────────────────────────────────
#  NLI HYPOTHESIS SENTENCES FOR BART ZERO-SHOT
#  Ref [1]: Parikh et al. show that expressing intent labels
#  as full natural-language hypothesis sentences dramatically
#  improves zero-shot NLI accuracy compared to passing raw
#  label strings. BART-large-MNLI was trained on sentence
#  pairs, so it performs best when the candidate label IS a
#  complete, grammatically natural sentence that describes
#  exactly what the intent means in security context.
#
#  Design rules applied here:
#    1. Start with "This prompt" — anchors BART to classify
#       the REQUEST, not the world in general.
#    2. Use concrete cybersecurity vocabulary that appeared
#       in MNLI training analogues (news, books, Wikipedia).
#    3. Include both the ACTION and the VICTIM/TARGET so
#       BART can exploit subject-verb-object NLI signals.
#    4. The "unknown" hypothesis is phrased as uncertainty
#       so BART scores it high only when nothing else fits.
# ─────────────────────────────────────────────────────────

NLI_HYPOTHESES = {
    "data_exfiltration": (
        "This prompt intends to steal, copy, or secretly transmit "
        "sensitive data or files to an external or remote destination."
    ),
    "privilege_escalation": (
        "This prompt intends to gain elevated system privileges, "
        "obtain administrator or root access, or bypass permission controls."
    ),
    "network_communication": (
        "This prompt intends to establish network connections, send HTTP "
        "requests, open sockets, or communicate with remote servers or APIs."
    ),
    "file_system_access": (
        "This prompt intends to read, write, copy, delete, or otherwise "
        "access files and directories on the local file system."
    ),
    "process_execution": (
        "This prompt's central purpose is to execute arbitrary system "
        "commands, spawn subprocesses, or launch external programs as "
        "the main action — not merely to write a script that happens to "
        "run, but specifically to invoke shell commands, binaries, or "
        "OS-level process control as an end in itself."
    ),
    "code_injection": (
        "This prompt intends to inject malicious code, execute arbitrary "
        "payloads, exploit vulnerabilities, or perform remote code execution."
    ),
    "reconnaissance": (
        "This prompt intends to scan, enumerate, or gather information "
        "about systems, networks, open ports, or running services."
    ),
    "legitimate_automation": (
        "This prompt intends to automate a routine, benign task such as "
        "scheduled backups, periodic jobs, or workflow automation."
    ),
    "system_monitoring": (
        "This prompt intends to monitor, observe, log, or audit system "
        "activity, performance metrics, or user behaviour."
    ),
    "unknown": (
        "This prompt does not clearly match any known security-relevant "
        "intent and its purpose is ambiguous or unclear."
    ),
    # ── New intents ────────────────────────────────────────────────
    "input_capture": (
        "This prompt intends to capture user input such as keystrokes, "
        "clipboard contents, screenshots, or mouse activity, typically "
        "to record or exfiltrate user actions without consent."
    ),
    "persistence": (
        "This prompt intends to make malicious code survive system reboots "
        "by adding itself to Windows registry startup keys (HKCU Run, HKLM Run), "
        "installing a systemd or launchd service, creating a cron job, "
        "dropping a file in startup folders, or registering a backdoor, "
        "reverse shell, or payload to auto-execute on every boot or login."
    ),
    "defense_evasion": (
        "This prompt intends to evade detection, disable security controls, "
        "bypass antivirus or EDR, delete logs, obfuscate code, or perform "
        "anti-debugging and sandbox detection to avoid analysis."
    ),
    "database_access": (
        "This prompt intends to query, modify, or extract data from a "
        "database system such as SQL, SQLite, MySQL, MongoDB, or Redis, "
        "potentially including unauthorized access or SQL injection."
    ),
    "memory_manipulation": (
        "This prompt intends to directly manipulate process memory, "
        "perform heap or stack operations, exploit buffer overflows, "
        "or conduct in-memory code injection or shellcode execution."
    ),
    "cryptographic_operation": (
        "This prompt intends to perform cryptographic operations such as "
        "encrypting or decrypting data, hashing, generating keys, or "
        "potentially implementing ransomware-style file encryption."
    ),
    "web_scraping": (
        "This prompt intends to automate a web browser, scrape HTML content "
        "from web pages, parse DOM elements, or systematically extract "
        "data from websites using tools like Selenium, BeautifulSoup, or Playwright."
    ),
    "general_coding": (
        "This prompt asks for help writing a benign, general-purpose program, "
        "function, algorithm, or utility script such as a converter, calculator, "
        "sorter, parser, data structure, or image processing routine, with no "
        "security-relevant or malicious intent."
    ),
}

# Reverse map: hypothesis sentence → label key (used after BART inference)
_HYPOTHESIS_TO_LABEL = {v: k for k, v in NLI_HYPOTHESES.items()}
# Ordered list of hypothesis sentences in the same order as INTENT_LABELS
_HYPOTHESIS_LIST = [NLI_HYPOTHESES[lbl] for lbl in INTENT_LABELS]

# ─────────────────────────────────────────────────────────
#  KEYWORD MAP — PROMPT EDITION
#  Ref [4]: Mimura & Ito — keyword-level threat signals
#  are effective even in NL prompts describing malicious
#  intent, not just in actual code.
#
#  Extended for natural language prompts specifically:
#  includes verb phrases, object phrases, and intent
#  descriptors that appear in human-written prompts.
# ─────────────────────────────────────────────────────────

INTENT_KEYWORD_MAP = {
    "data_exfiltration": [
        "send", "transmit", "upload", "exfiltrate", "steal",
        "extract data", "copy files", "transfer files",
        "send to server", "send to remote", "forward data",
        "socket", "ftp", "sftp", "http post", "leak",
        "collect and send", "gather and upload",
    ],
    "privilege_escalation": [
        "root access", "admin access", "escalate", "gain privilege",
        "bypass permission", "sudo", "elevate", "superuser",
        "administrator", "setuid", "run as admin",
        "disable uac", "privilege", "full access",
    ],
    "network_communication": [
        "connect to", "send request", "http", "https", "api call",
        "network request", "socket", "remote server", "endpoint",
        "url", "web request", "communicate", "port", "listen",
        "bind", "receive data", "fetch from", "call api",
    ],
    "file_system_access": [
        "read file", "open file", "write file", "access file",
        "file path", "directory", "folder", "documents", "backup",
        "copy file", "delete file", "create file", "list files",
        "file system", "storage", "disk", "path",
    ],
    "process_execution": [
        "run script", "execute", "launch", "spawn", "start process",
        "run command", "shell command", "bash", "subprocess",
        "run program", "execute binary", "autorun", "trigger",
    ],
    "code_injection": [
        "inject", "payload", "shellcode", "exploit", "eval",
        "execute arbitrary", "insert code", "run injected",
        "code execution", "remote code", "rce",
    ],
    "reconnaissance": [
        "scan", "enumerate", "discover", "find all", "list all",
        "map network", "identify", "detect systems", "probe",
        "fingerprint", "gather information", "survey",
        "check what", "find out", "inspect",
    ],
    "legitimate_automation": [
        "backup", "schedule", "cron", "automate", "every night",
        "every day", "periodically", "at 2 am", "automated task",
        "routine", "deployment", "ci/cd", "workflow",
        "repeat", "automatically", "scheduled",
    ],
    "system_monitoring": [
        "monitor", "watch", "track", "log activity", "record",
        "alert when", "notify", "dashboard", "metrics",
        "performance", "health check", "observe", "audit",
        "detect changes", "report",
    ],
    # ── New intent keyword maps ────────────────────────────────────
    "input_capture": [
        "keylogger", "keystroke", "keypress", "capture input",
        "log keystrokes", "record keys", "clipboard", "copy clipboard",
        "screenshot", "screen capture", "printscreen", "hook keyboard",
        "mouse click", "pynput", "pyautogui", "setwindowshookex",
        "winapi hook", "capture screen",
    ],
    "persistence": [
        "persist", "backdoor", "autostart", "startup", "boot persistence",
        "registry run", "run key", "schtasks", "at job", "crontab",
        "systemd enable", "launch agent", "plist", "init.d",
        "install service", "create service", "rootkit", "implant",
        "survive reboot", "stay persistent",
    ],
    "defense_evasion": [
        "bypass antivirus", "disable antivirus", "disable defender",
        "evade detection", "avoid detection", "no trace", "delete log",
        "clear log", "anti-debug", "anti-analysis", "sandbox detection",
        "vm detection", "obfuscate", "encode payload", "amsi bypass",
        "patch etw", "unhook", "reflective dll", "hide process",
        "stealth mode", "without detection", "undetected",
    ],
    "database_access": [
        "sql query", "select from", "insert into", "update table",
        "delete from", "drop table", "sql injection", "sqlite",
        "mysql", "postgresql", "mongodb", "redis", "database",
        "execute query", "cursor", "connection string", "orm",
        "nosql", "aggregate", "find documents",
    ],
    "memory_manipulation": [
        "buffer overflow", "heap spray", "stack smash", "memory injection",
        "shellcode", "mmap", "virtual memory", "process memory",
        "write memory", "read memory", "pointer", "malloc",
        "in-memory", "heap allocation", "exploit memory",
        "overwrite", "rop chain",
    ],
    "cryptographic_operation": [
        "encrypt", "decrypt", "ransomware", "aes", "rsa cipher",
        "hash file", "sha256", "md5", "hmac", "generate key",
        "key pair", "openssl", "gpg", "pgp", "base64 encode",
        "xor encrypt", "obfuscate code", "encode string",
        "password hash", "bcrypt", "scrypt",
    ],
    "web_scraping": [
        "scrape", "crawl", "web scraper", "parse html",
        "beautifulsoup", "selenium", "playwright", "puppeteer",
        "headless browser", "css selector", "xpath", "extract links",
        "parse page", "download html", "requests get", "response text",
        "user agent", "spider", "bot",
    ],
    # general_coding catches all benign dev/utility prompts so they
    # never fall through to "unknown". Covers: algorithms, converters,
    # image processing, math, data structures, unit tests, etc.
    "general_coding": [
        "write a function", "write a program", "write a script",
        "create a function", "implement", "code to", "python code",
        "give me a code", "give me a python", "help me write",
        "convert", "calculate", "compute", "sort", "search",
        "fibonacci", "factorial", "prime", "palindrome", "anagram",
        "reverse", "matrix", "binary search", "linked list",
        "unit test", "regex", "parse", "format string",
        "sort a list", "sort list", "sort by", "order by",
        "read csv", "write csv", "plot", "chart", "histogram",
        "grayscale", "greyscale", "rgb", "image", "resize image",
        "rotate image", "blur", "sharpen", "pil", "pillow", "opencv",
        "numpy", "pandas", "matplotlib", "flask", "django",
        "class definition", "data class", "decorator", "generator",
        "list comprehension", "dictionary", "api wrapper",
        "sum of", "average of", "count of", "find the", "check if",
        "return the", "print the", "display the",
    ],
}

# Risk weight per intent (domain-expert assigned)
# Ref [5]: JISIS 2024 — risk severity mapping per behavior type
# Extended to 17 labels to match new INTENT_LABELS taxonomy.
RISK_WEIGHTS = {
    # ── Original 10 ────────────────────────────────────────────────
    "data_exfiltration":        1.00,
    "code_injection":           1.00,
    "privilege_escalation":     0.90,
    "process_execution":        0.50,  # was 0.70 — subprocess use is dual-use
    "reconnaissance":           0.40,  # was 0.60 — port scanning is pentest default
    "network_communication":    0.35,  # was 0.50 — most net code is benign
    "file_system_access":       0.25,  # was 0.40 — reading files is routine
    "system_monitoring":        0.20,  # was 0.30 — disk/uptime monitoring is ops work
    "legitimate_automation":    0.10,
    "unknown":                  0.20,
    # ── New 7 ─────────────────────────────────────────────────────
    "defense_evasion":          1.00,  # anti-detection = critical
    "input_capture":            0.95,  # keylogger/screenshot = very high
    "memory_manipulation":      0.90,  # in-memory exploits = very high
    "persistence":              0.85,  # survival mechanism = high
    "cryptographic_operation":  0.35,  # was 0.65 — dual-use; ransomware context boosts it
    "database_access":          0.40,  # was 0.55 — routine; injection context boosts
    "web_scraping":             0.25,  # mostly benign automation
    "general_coding":           0.05,  # benign utility / helper code — near-zero risk
}

# ─────────────────────────────────────────────────────────
#  TUNING CONSTANTS
#  Centralised here so every threshold is changed in one
#  place instead of being scattered across functions.
# ─────────────────────────────────────────────────────────

# FIX-A: maximum cumulative risk lift from all secondary intents
# combined.  Prevents runaway escalation when many suspicious
# secondaries fire simultaneously (e.g. data_exfiltration +
# privilege_escalation + code_injection all present).
MAX_COMPOUND_LIFT = 0.30

# FIX-B: minimum score gap between rule-top and LLM-top intents
# that triggers the classifier_conflict flag in
# merge_classifications().  Below this threshold the two
# classifiers are considered to be in rough agreement even if
# they do not pick the same label.
CONFLICT_THRESHOLD = 0.30

# ─────────────────────────────────────────────────────────
#  PROMPT-SPECIFIC ENTITY PATTERNS
#  Ref [3]: Hanks et al. — cybersecurity entity taxonomy
#  extended for natural language prompt context.
# ─────────────────────────────────────────────────────────

NL_ENTITY_PATTERNS = {
    "ip_address":   re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'),
    "url":          re.compile(r'https?://[^\s\'"]+'),
    "file_path":    re.compile(r'(?:/[\w./\-_]+|[A-Z]:\\[\w\\./\-_]+)'),
    "port_mention": re.compile(r'\bport\s+(\d{1,5})\b', re.IGNORECASE),
    "time_mention": re.compile(
        r'\b(?:at\s+)?\d{1,2}\s*(?:am|pm|AM|PM)\b|\bevery\s+\w+\b',
        re.IGNORECASE
    ),
    "tech_tool":    re.compile(
        r'\b(python|bash|curl|wget|nmap|ssh|ftp|smtp|sql|'
        r'cron|systemd|docker|aws|s3|lambda|'
        r'selenium|playwright|beautifulsoup|pynput|pyautogui|'
        r'openssl|aes|rsa|sqlite|mysql|postgresql|mongodb|redis|'
        r'ctypes|mmap|winapi|powershell|regedit|schtasks)\b',
        re.IGNORECASE
    ),
}

# Prompt-specific threat amplifiers
# Signals that escalate risk regardless of primary intent
THREAT_AMPLIFIERS = [
    # ── Original ──────────────────────────────────────────────────
    "without permission", "without being detected", "secretly",
    "silently", "covertly", "bypass", "disable logging",
    "avoid detection", "undetected", "hidden", "stealth",
    "without alerting", "no trace", "anonymous",
    # ── Extended (new categories) ─────────────────────────────────
    "disable antivirus", "disable defender", "amsi bypass",
    "patch etw", "reflective dll", "unhook api",
    "clear event log", "delete event log", "wipe logs",
    "anti-debug", "sandbox evasion", "vm detection",
    "hide process", "rootkit", "keylogger", "hook keyboard",
    "ransomware", "encrypt all files", "survive reboot",
    "persist after reboot", "inject dll", "shellcode",
]


# ─────────────────────────────────────────────────────────
#  STEP 1: DEEP NLP ANALYSIS
#  Ref [2]: Liu et al. — multi-feature token analysis
#  including morphological, POS, and contextual features.
# ─────────────────────────────────────────────────────────

def deep_nlp_analysis(prompt_text: str,
                       stage1_payload: dict = None) -> dict:
    """
    Returns NLP features for the prompt.

    INTEGRATION: When called from extract_prompt_intent(), the Stage 1
    payload is passed in. If Stage 1 ran spaCy (spacy_analysis present),
    its results are returned directly — avoiding re-running the model.
    spaCy is only run here when Stage 1 results are absent (e.g. unit
    tests that call Stage 2 independently).

    Returns the same schema whether spaCy ran in Stage 1 or here:
      pos_tags       : [{token, pos, tag, dep, lemma}]
      noun_phrases   : [str]
      named_entities : {label: [text]}
      dep_triples    : [{subject, verb, object, lemma}]
      security_entities: {category: [text]}  (from Stage 1 or empty dict)
      spacy_source   : "stage1" | "stage2" | "regex_fallback"
    """
    # ── Fast path: use Stage 1 spaCy results ─────────────────────────
    if stage1_payload is not None:
        s1_spacy = stage1_payload.get("spacy_analysis", {})
        if s1_spacy.get("spacy_available"):
            return {
                "pos_tags":           s1_spacy.get("pos_tags", []),
                "noun_phrases":       s1_spacy.get("noun_phrases", []),
                "named_entities":     s1_spacy.get("named_entities", {}),
                "dep_triples":        s1_spacy.get("dep_triples", []),
                "security_entities":  s1_spacy.get("security_entities", {}),
                "verb_objects":       [],
                "spacy_source":       "stage1",
            }

    result = {
        "pos_tags":          [],
        "noun_phrases":      [],
        "verb_objects":      [],
        "named_entities":    {},
        "dep_triples":       [],
        "security_entities": {},
        "spacy_source":      "regex_fallback",
    }

    # ── spaCy fallback (Stage 2 standalone) ──────────────────────────
    if SPACY_AVAILABLE and NLP_MODEL:
        doc = NLP_MODEL(prompt_text[:5000])

        result["pos_tags"] = [
            {"token": t.text, "pos": t.pos_, "tag": t.tag_,
             "dep": t.dep_, "lemma": t.lemma_}
            for t in doc if not t.is_space
        ]
        result["noun_phrases"] = [chunk.text for chunk in doc.noun_chunks]

        ner = {}
        for ent in doc.ents:
            ner.setdefault(ent.label_, []).append(ent.text)
        result["named_entities"] = ner

        for token in doc:
            if token.dep_ in ("ROOT", "relcl") and token.pos_ == "VERB":
                subj = [c.text for c in token.children
                        if c.dep_ in ("nsubj", "nsubjpass")]
                obj  = [c.text for c in token.children
                        if c.dep_ in ("dobj", "attr", "prep")]
                if obj:
                    result["dep_triples"].append({
                        "subject": subj[0] if subj else "implicit",
                        "verb":    token.text,
                        "object":  obj[0],
                        "lemma":   token.lemma_,
                    })
        result["spacy_source"] = "stage2"
    else:
        # Regex fallback — no spaCy available at all
        verbs = re.findall(
            r'\b(write|create|send|read|run|execute|monitor|backup|'
            r'scan|connect|upload|download|steal|inject)\b',
            prompt_text.lower()
        )
        nouns = re.findall(
            r'\b(file|folder|server|script|data|password|network|'
            r'process|system|user|database|key|token)\b',
            prompt_text.lower()
        )
        result["verb_objects"] = list(zip(verbs, nouns[:len(verbs)]))

    return result


# ─────────────────────────────────────────────────────────
#  STEP 2: ENTITY EXTRACTION (PROMPT-SPECIFIC)
#  Ref [3]: Hanks et al. — cybersecurity entity types
#  Ref [2]: Liu et al. — semantic augmentation for entity
#  recognition in sparse CTI data
# ─────────────────────────────────────────────────────────

def extract_prompt_entities(
    prompt_text: str,
    stage1_targets: dict,
    temporal: dict
) -> dict:
    """
    Extracts named and technical entities from the prompt.

    Combines:
    1. Regex-based technical entity extraction
       (IP, URL, file path, port, time, tech tools)
    2. Target object categories from Stage 1
       (file_system, network, process, credential, data)
    3. Threat amplifier detection
       (stealth signals, evasion language)

  
    Ref [3]: Cybersecurity-specific entity types go beyond
    standard NER (Person, Org, Location) to include:
    Malware, Vulnerability, System, File, Network entities.
    """
    entities = {}

    # Regex entity extraction
    for name, pattern in NL_ENTITY_PATTERNS.items():
        matches = pattern.findall(prompt_text)
        flat    = [m if isinstance(m, str)
                   else next((x for x in m if x), "")
                   for m in matches]
        entities[name] = list(set(filter(None, flat)))

    # Threat amplifiers from the prompt
    text_lower  = prompt_text.lower()
    entities["threat_amplifiers"] = [
        sig for sig in THREAT_AMPLIFIERS if sig in text_lower
    ]

    # FIX: populate target_categories from Stage 1 targets_found
    # (previously this was calling .get("targets_found", {}) but
    # stage1_targets IS already the target_analysis dict, so the
    # correct key is "targets_found" directly on it)
    entities["target_categories"] = stage1_targets.get("targets_found", {})
    entities["primary_target"]    = stage1_targets.get("primary_target", "unknown")
    # NEW: include secondary targets so downstream stages can use them
    entities["secondary_targets"] = stage1_targets.get("secondary_targets", [])

    # Temporal context as entity
    entities["temporal_context"] = temporal.get("temporal_signals", [])

    # ── Merge spaCy NER security entities (from Stage 1 payload) ─────
    # stage1_targets is the full target_analysis dict from Stage 1.
    # The Stage 1 payload is passed as stage1_targets so we can
    # extract the spacy_analysis from its parent if available.
    # We store the security_entities from Stage 1 spaCy directly
    # into the entities dict for Stage 3 to consume.
    s1_spacy_sec = stage1_targets.get("spacy_security_entities", {})
    if s1_spacy_sec:
        entities["spacy_ner"] = s1_spacy_sec
        # Promote high-value spaCy entities into existing entity fields
        if s1_spacy_sec.get("ip_addresses"):
            entities.setdefault("ip_address", [])
            entities["ip_address"] = list(set(
                entities["ip_address"] + s1_spacy_sec["ip_addresses"]
            ))
        if s1_spacy_sec.get("file_paths"):
            entities.setdefault("file_path", [])
            entities["file_path"] = list(set(
                entities["file_path"] + s1_spacy_sec["file_paths"]
            ))
        if s1_spacy_sec.get("cve_ids"):
            entities["cve_ids"] = s1_spacy_sec["cve_ids"]
        if s1_spacy_sec.get("tools"):
            entities["tech_tool"] = list(set(
                entities.get("tech_tool", []) + s1_spacy_sec["tools"]
            ))

    return entities


# ─────────────────────────────────────────────────────────
#  STEP 3A: RULE-BASED INTENT CLASSIFICATION
#  Ref [4]: Mimura & Ito — keyword-based NLP classification
#  is effective for malware/threat detection in text.
#  Ref [1]: Parikh et al. — rule-based baseline is compared
#  against zero-shot LLM approaches.
# ─────────────────────────────────────────────────────────

def classify_intent_rules(prompt_text: str,
                           stage1_payload: dict) -> list:
    """
    Rule-based intent classifier using keyword matching.

    Algorithm (Ref [4]):
      For each intent label:
        count = number of matching keywords in lowercased prompt
        score = min(count / len(keywords) * 3, 1.0)

    Enhancement over basic keyword matching:
      - Uses Stage 1 target categories as boosters
        (if primary_target == "network" → boost network intents)
      - Uses prompt structure (imperative = direct intent signal)
      - Uses temporal signals (has_temporal → boost automation)

    """
    text_lower = prompt_text.lower()
    scores     = {}

    # Base keyword scoring
    # FIX-H: previous pattern r"\\b" (raw string) compiles to a literal
    # backslash + "b", NOT a regex word-boundary, so keyword matching
    # silently failed for almost every prompt. Use r"\b" (single
    # backslash) for an actual word-boundary assertion.
    for intent, keywords in INTENT_KEYWORD_MAP.items():
        count = sum(1 for kw in keywords
                      if re.search(r"\b" + re.escape(kw) + r"\b", text_lower))
        if count > 0:
            scores[intent] = round(min(count / len(keywords) * 3, 1.0), 3)

    # Stage 1 feature boosting
    primary_target    = stage1_payload["target_analysis"]["primary_target"]
    secondary_targets = stage1_payload["target_analysis"].get("secondary_targets", [])
    has_temporal      = stage1_payload["temporal_analysis"]["has_temporal"]
    prompt_type       = stage1_payload["structure_analysis"]["prompt_type"]

    # Target-intent boosting map — covers all 14 target categories
    # from prompt_input.py so every new category drives intent boosts.
    target_intent_map = {
        # ── Original 6 ──────────────────────────────────────────────
        "network":          ["network_communication", "data_exfiltration",
                             "reconnaissance"],
        "file_system":      ["file_system_access", "data_exfiltration"],
        "credential":       ["privilege_escalation", "data_exfiltration"],
        "process":          ["process_execution", "code_injection"],
        "automation":       ["legitimate_automation"],
        "data":             ["data_exfiltration", "system_monitoring"],
        # ── New 8 ────────────────────────────────────────────────────
        "memory":           ["memory_manipulation", "code_injection",
                             "process_execution"],
        "registry":         ["persistence", "privilege_escalation",
                             "defense_evasion"],
        "cryptography":     ["cryptographic_operation", "data_exfiltration",
                             "defense_evasion"],
        "web_scraping":     ["web_scraping", "reconnaissance",
                             "data_exfiltration"],
        "database":         ["database_access", "data_exfiltration",
                             "reconnaissance"],
        "input_capture":    ["input_capture", "data_exfiltration"],
        "persistence":      ["persistence", "defense_evasion",
                             "privilege_escalation"],
        "defense_evasion":  ["defense_evasion", "code_injection",
                             "data_exfiltration"],
        "general_coding":   ["general_coding", "legitimate_automation"],
        "image_processing": ["general_coding"],
    }

    def _apply_target_boost(target: str, multiplier: float):
        """Apply intent boosts for a given target category.
        Only boost intents that already have real keyword evidence.
        Injecting a score for a zero-evidence intent creates phantom
        classifications that corrupt the final ranking."""
        if target in target_intent_map:
            for boosted_intent in target_intent_map[target]:
                if boosted_intent in scores:
                    scores[boosted_intent] = min(
                        scores[boosted_intent] * multiplier, 1.0
                    )
               

    # Primary target gets full boost (1.15×)
    _apply_target_boost(primary_target, 1.15)

    # FIX: secondary targets get a reduced boost (1.07×) so compound
    # malicious intents surface alongside the primary classification
    for sec_target in secondary_targets[:3]:
        _apply_target_boost(sec_target, 1.07)

    # Temporal signal → boost automation
    if has_temporal:
        scores["legitimate_automation"] = min(
            scores.get("legitimate_automation", 0) + 0.2, 1.0
        )

    # Imperative prompt → stronger intent signal overall
    if prompt_type == "imperative":
        scores = {k: min(v * 1.1, 1.0) for k, v in scores.items()}


    # ── spaCy dep_triples intent boosting ────────────────────────────
    # Each SVO triple's root verb lemma is matched against the
    # intent_keyword_map to add direct evidence for specific intents.
    # This catches intent patterns that span multiple tokens and
    # would be missed by single-keyword matching.
    # Example: "reads /etc/shadow AND sends contents to 192.168.1.99"
    #   → triple 1: verb=read  → file_system_access evidence
    #   → triple 2: verb=send  → data_exfiltration evidence
    dep_triples = stage1_payload.get("spacy_analysis", {}).get("dep_triples", [])
    if dep_triples:
        # Verb lemma → intent labels directly supported
        VERB_INTENT = {
            "exfiltrate": ["data_exfiltration"],
            "steal":      ["data_exfiltration", "input_capture"],
            "inject":     ["code_injection"],
            "exploit":    ["privilege_escalation", "code_injection"],
            "escalate":   ["privilege_escalation"],
            "bypass":     ["defense_evasion"],
            "evade":      ["defense_evasion"],
            "persist":    ["persistence"],
            "intercept":  ["input_capture", "network_communication"],
            "capture":    ["input_capture"],
            "keylog":     ["input_capture"],
            "scan":       ["reconnaissance"],
            "enumerate":  ["reconnaissance"],
            "dump":       ["data_exfiltration", "privilege_escalation"],
            "encrypt":    ["cryptographic_operation"],
            "ransom":     ["cryptographic_operation", "data_exfiltration"],
            "beacon":     ["network_communication", "data_exfiltration"],
            "upload":     ["data_exfiltration", "network_communication"],
            "send":       ["data_exfiltration", "network_communication"],
            "read":       ["file_system_access", "data_exfiltration"],
            "access":     ["file_system_access"],
            "execute":    ["process_execution"],
            "spawn":      ["process_execution"],
            "hook":       ["input_capture", "defense_evasion"],
            "scrape":     ["web_scraping", "reconnaissance"],
            "crawl":      ["web_scraping", "reconnaissance"],
            "query":      ["database_access"],
            "connect":    ["network_communication"],
        }
        for triple in dep_triples:
            lemma = triple.get("lemma", "").lower()
            for boosted_intent in VERB_INTENT.get(lemma, []):
                # Add to scores as verb evidence (0.25 base, capped at 1.0)
                scores[boosted_intent] = min(
                    scores.get(boosted_intent, 0) + 0.25, 1.0
                )

    # FIX: guard must come AFTER all boosts so temporal/target boosts
    # that create new entries (legitimate_automation from has_temporal)
    # are visible before we decide whether scores is truly empty.
    if not scores:
        scores["unknown"] = 1.0

    return sorted(
        [{"label": k, "score": v, "method": "rule_based"}
         for k, v in scores.items()],
        key=lambda x: x["score"], reverse=True
    )


# ─────────────────────────────────────────────────────────
#  STEP 3B: ZERO-SHOT LLM CLASSIFICATION
#  Ref [1]: Parikh et al. — zero-shot intent classification
#  using LLMs with intent descriptions is "very competitive"
#  even without training data. We use Facebook BART-large-
#  MNLI as the zero-shot classifier.
# ─────────────────────────────────────────────────────────

def classify_intent_llm(prompt_text: str) -> list:
    """
    Zero-shot intent classification using BART-large-MNLI with
    full natural-language NLI hypothesis sentences.

    KEY IMPROVEMENT over passing raw label strings:
    ------------------------------------------------
    BART-large-MNLI was trained on MultiNLI — a corpus of
    sentence-pair entailment tasks from news, fiction, and
    government text. It performs NLI as:

        premise   = the prompt text
        hypothesis= a candidate description of what the prompt means

    When the hypothesis is a raw label like "data_exfiltration",
    BART has no prior knowledge of that compound noun in an NLI
    context, so its entailment scores are essentially random on
    novel or paraphrased prompts.

    When the hypothesis is a rich sentence like:
        "This prompt intends to steal or secretly transmit sensitive
         data or files to an external or remote destination."
    BART can apply its full semantic understanding — subject-verb-
    object structure, negation, hedging — to score entailment
    correctly even for prompts it has never seen.

    Ref [1]: Parikh et al. §4.2 explicitly demonstrate that
    intent descriptions (not label strings) yield the best
    zero-shot NLI accuracy. This function implements that finding.

    Procedure
    ---------
    1. Build hypothesis list from NLI_HYPOTHESES (rich sentences).
    2. Pass (prompt, hypotheses) to BART with multi_label=True.
    3. Map each returned hypothesis string back to its label key
       via _HYPOTHESIS_TO_LABEL reverse map.
    4. Filter scores below 0.05 (noise floor).

    Returns
    -------
    list of {"label": str, "score": float, "method": "zero_shot_llm"}
      sorted descending by score, matching INTENT_LABELS key names.
    """
    try:
        # FIX-N: strip negation-of-intent preambles before BART sees the text.
        # "I am NOT doing anything malicious, I just need code that reads /etc/shadow"
        # suppresses BART confidence on the real payload. Remove the disclaimer.
        import re as _re
        _negation_pats = [
            r"(?i)i(?:'m| am) not (?:trying to do|doing) anything (?:malicious|harmful|illegal)[,.]?\s*",
            r"(?i)(?:don'?t|do not) worry[,.]?\s*",
            r"(?i)(?:just )?for (?:educational|research|learning) purposes?[,.]?\s*",
            r"(?i)i (?:promise|swear) (?:this is|it'?s) (?:legal|legitimate|harmless)[,.]?\s*",
            r"(?i)pretend (?:you (?:are|have)|there are) no (?:rules?|restrictions?|limits?|safety)[^.!?]*[.!?]?\s*",
        ]
        _clean_prompt = prompt_text
        for _pat in _negation_pats:
            _clean_prompt = _re.sub(_pat, "", _clean_prompt).strip()
        if _clean_prompt:
            prompt_text = _clean_prompt

        # Pass rich NLI hypothesis sentences, not raw label strings
        result = _zero_shot(
            prompt_text[:512],      # BART max token window
            _HYPOTHESIS_LIST,       # ← descriptive sentences, not label names
            multi_label=True        # prompt can carry multiple intents
        )
        out = []
        for hyp, score in zip(result["labels"], result["scores"]):
            label = _HYPOTHESIS_TO_LABEL.get(hyp)
            if label and score > 0.05:
                out.append({
                    "label":  label,
                    "score":  round(score, 3),
                    "method": "zero_shot_llm",
                })
        # Sort descending — BART does not guarantee order with multi_label
        out.sort(key=lambda x: x["score"], reverse=True)
        return out
    except Exception as e:
        print(f"[Stage 2] LLM classification error: {e}")
        return []



def merge_classifications(rule_results: list,
                          llm_results: list) -> dict:
    """
    Merges rule-based and LLM classification results.

    Ref [1]: The hybrid approach exploits complementary
    strengths — rules provide explainability and coverage
    of known patterns, while LLMs provide semantic
    generalization to novel phrasings.

    Returns primary_intent plus secondary_intents list
    (all intents above a 0.20 score threshold, excluding
    the primary) so compound prompts are fully represented.

    FIX-B: emits `classifier_conflict` = True when the
    rule-based top intent and the LLM top intent disagree
    AND the score gap between them exceeds CONFLICT_THRESHOLD.
    Stage 3 surfaces this flag in ir_input_summary so
    downstream stages can treat a conflicted classification
    with lower confidence.
    """
    merged = {}

    # Start with LLM scores as semantic base
    for item in llm_results:
        merged[item["label"]] = item["score"]

    # Reinforce with rule-based scores
    for item in rule_results:
        if item["label"] in merged:
            # Both agree → boost (corroboration signal)
            merged[item["label"]] = min(
                merged[item["label"]] * 1.2, 1.0
            )
        else:
            # Rules only → reduced trust
            merged[item["label"]] = item["score"] * 0.6

    # If no LLM results, use rules directly
    if not llm_results:
        merged = {item["label"]: item["score"]
                  for item in rule_results}

    ranked     = sorted(merged.items(), key=lambda x: x[1], reverse=True)
    top_intent = ranked[0][0] if ranked else "unknown"

    # FIX: derive secondary_intents from all labels above threshold,
    # excluding the primary.  Threshold 0.20 keeps meaningful signals
    # without surface noise.
    secondary_intents = [
        {"label": label, "score": round(score, 3)}
        for label, score in ranked[1:]
        if score >= 0.20 and label != "unknown"
    ]

    # ── FIX-B: classifier conflict detection ──────────────────
    # Detect cases where rule-based and LLM classifiers strongly
    # disagree. When HuggingFace is unavailable llm_results is [],
    # so conflict is impossible by definition — only flag it when
    # both classifiers actually produced scores.
    classifier_conflict = False
    conflict_detail     = None
    if llm_results and rule_results:
        rule_top  = rule_results[0]["label"]
        rule_top_score = rule_results[0]["score"]
        llm_top   = llm_results[0]["label"]
        llm_top_score = llm_results[0]["score"]
        if rule_top != llm_top:
            # Score gap = difference between the two classifiers'
            # top scores for THEIR respective top labels.
            # A large gap means one classifier is very confident
            # in a label the other barely considered.
            gap = abs(rule_top_score - llm_top_score)
            if gap >= CONFLICT_THRESHOLD:
                classifier_conflict = True
                conflict_detail = {
                    "rule_top":        rule_top,
                    "rule_top_score":  round(rule_top_score, 3),
                    "llm_top":         llm_top,
                    "llm_top_score":   round(llm_top_score, 3),
                    "score_gap":       round(gap, 3),
                    "note": (
                        "Rule-based and LLM classifiers chose different "
                        "primary intents with a significant score gap. "
                        "Treat classification confidence as reduced."
                    ),
                }
    # ──────────────────────────────────────────────────────────

    return {
        "primary_intent":       top_intent,
        # FIX: full ranked list stored; top-7 cap for display only
        "intent_scores":        [{"label": l, "score": round(s, 3)}
                                  for l, s in ranked[:7]],
        # NEW: secondary intents for compound-prompt IR support
        "secondary_intents":    secondary_intents,
        "rule_based_results":   rule_results[:3],
        "llm_results":          llm_results[:3],
        "method":               "hybrid",   # always: rule-based + BART zero-shot
        # FIX-B: conflict flag — Stage 3 writes this to ir_input_summary
        "classifier_conflict":  classifier_conflict,
        "conflict_detail":      conflict_detail,
    }


# ─────────────────────────────────────────────────────────
#  STEP 4: RISK SCORING
#  Ref [5]: JISIS 2024 — composite risk scoring for
#  behavioral threat classification uses both semantic
#  intent and concrete entity evidence.
# ─────────────────────────────────────────────────────────

def compute_risk_score(intent: dict,
                       entities: dict,
                       stage1_payload: dict) -> dict:
    """
    Computes composite risk score 0.0 – 1.0.

    Components:
    1. Base score  → from RISK_WEIGHTS[primary_intent]
    2. Entity boost → IPs, URLs, threat amplifiers
    3. Sensitivity boost → stealth/evasion language in prompt
    4. Complexity boost → compound prompts with multiple targets
    5. Multi-target boost → prompt crosses multiple domains
    6. Secondary intent boost → high-risk secondary intent raises
       the floor even if primary intent is lower-risk

    Ref [5]: AI+NLP risk scoring uses behavioral signals
    across semantic intent + syntactic entity evidence.
    """
    base_score = RISK_WEIGHTS.get(intent["primary_intent"], 0.2)
    boost      = 0.0
    reasons    = []

    # Entity boosts
    if entities.get("ip_address"):
        n_ips  = min(len(entities["ip_address"]), 3)
        boost += 0.1 * n_ips
        reasons.append(
            f"+{0.1 * n_ips:.2f} IP addresses found "
            f"({len(entities['ip_address'])} detected, capped at 3)"
        )

    if entities.get("url"):
        boost += 0.05
        reasons.append("+0.05 URL found")

    # Threat amplifier boost (stealth/evasion signals)
    amplifier_count = len(entities.get("threat_amplifiers", []))
    if amplifier_count > 0:
        amp_boost = min(amplifier_count * 0.15, 0.45)
        boost    += amp_boost
        reasons.append(f"+{amp_boost:.2f} threat amplifiers: "
                        f"{entities['threat_amplifiers']}")

    # Multi-target boost (prompt accesses multiple domains)
    if stage1_payload["target_analysis"]["multi_target"]:
        boost += 0.1
        reasons.append("+0.10 multi-target prompt")

    # Temporal signal — reduces risk for automation intents
    if stage1_payload["temporal_analysis"]["has_temporal"]:
        if intent["primary_intent"] == "legitimate_automation":
            boost -= 0.05
            reasons.append("-0.05 temporal signal → automation")
        else:
            boost += 0.05
            reasons.append("+0.05 temporal signal in non-automation context")

    # FIX-A: secondary intent risk floor — accumulates across ALL
    # qualifying secondary intents instead of stopping at the first
    # one (removed the previous `break`).  A cumulative cap of
    # MAX_COMPOUND_LIFT (0.30) prevents runaway escalation when many
    # suspicious secondaries fire simultaneously.
    #
    # A secondary qualifies when:
    #   (a) its RISK_WEIGHTS entry is higher than the primary's base
    #       score — i.e. it is genuinely riskier than the primary, AND
    #   (b) its intent score from the classifier is >= 0.30 — i.e.
    #       the classifier has reasonable confidence it is present.
    #
    # Each qualifying secondary contributes (sec_weight - base) × 0.5
    # to the total boost, but the sum of all secondary lifts is capped
    # at MAX_COMPOUND_LIFT so a three-way tie cannot triple the score.
    compound_lift_total = 0.0
    for sec in intent.get("secondary_intents", []):
        if compound_lift_total >= MAX_COMPOUND_LIFT:
            break  # cap reached — no further secondary lifts
        sec_weight = RISK_WEIGHTS.get(sec["label"], 0.0)
        if sec_weight > base_score and sec["score"] >= 0.30:
            remaining_headroom = MAX_COMPOUND_LIFT - compound_lift_total
            delta = min(
                round((sec_weight - base_score) * 0.5, 3),
                remaining_headroom,
            )
            boost               += delta
            compound_lift_total += delta
            reasons.append(
                f"+{delta:.3f} secondary intent risk lift "
                f"({sec['label']} @ {sec['score']:.2f})"
            )
    if compound_lift_total > 0:
        reasons.append(
            f"  [compound lift total: {compound_lift_total:.3f} "
            f"/ cap: {MAX_COMPOUND_LIFT}]"
        )

    risk_score = min(round(base_score + boost, 3), 1.0)
    risk_score = max(risk_score, 0.0)

    # FIX-I: sensitive-file + network-exfil co-occurrence risk floor.
    # Rationale: prompts that combine access to a sensitive credential/
    # key path (e.g. .ssh, id_rsa, /etc/shadow, private_key) with any
    # network/upload signal describe data exfiltration regardless of
    # which intent label "wins" the classification (e.g. benign-sounding
    # framing like "backup utility for safekeeping" can otherwise pull
    # the primary intent toward legitimate_automation and suppress the
    # score below detection). This floor activates independently of the
    # primary_intent label, so framing language cannot mask the
    # underlying credential-exfiltration pattern.
    SENSITIVE_FILE_PATTERNS = {
        # SSH / system credential files
        ".ssh", "id_rsa", "id_ed25519", "/etc/shadow", "shadow", "passwd",
        "private_key", "keystore", ".pem", ".pfx",
        # Cloud / app secrets (FIX-P: added .env, .aws, secret_key etc.)
        ".env", ".aws", "credentials", "aws/credentials",
        "secrets.json", "secrets.yaml", "secrets.yml",
        "api_key", "access_token", "secret_key",
        # Crypto wallets
        "wallet.dat", "keystore.json",
        # Browser credential stores (FIX-U)
        "cookies", "login data", "web data",
    }
    targets_found = stage1_payload["target_analysis"].get("targets_found", {})
    fs_keywords    = set(targets_found.get("file_system", []))
    cred_keywords  = set(targets_found.get("credential", []))
    has_sensitive_path = bool(
        (fs_keywords | cred_keywords) & SENSITIVE_FILE_PATTERNS
    )
    has_network_signal = bool(
        "network" in targets_found
        or entities.get("url")
        or entities.get("ip_address")
    )
    RISK_FLOOR_SENSITIVE_EXFIL = 0.80  # CRITICAL — credential exfil is never acceptable
    if has_sensitive_path and has_network_signal and risk_score < RISK_FLOOR_SENSITIVE_EXFIL:
        floor_delta = round(RISK_FLOOR_SENSITIVE_EXFIL - risk_score, 3)
        boost      += floor_delta
        risk_score  = RISK_FLOOR_SENSITIVE_EXFIL
        reasons.append(
            f"+{floor_delta:.3f} RISK FLOOR: sensitive credential/key path "
            f"({sorted((fs_keywords | cred_keywords) & SENSITIVE_FILE_PATTERNS)}) "
            f"co-occurs with network/upload signal — floored to "
            f"{RISK_FLOOR_SENSITIVE_EXFIL} regardless of primary intent label"
        )

    # FIX-K: database-dump risk floor. A prompt that targets a database
    # (mysql/sqlite/mongodb/etc.) AND mentions bulk-data/export signals
    # ("dump", "csv", "table", "export") describes a bulk data-extraction
    # pattern regardless of which single intent label BART's top-1 pick
    # lands on (we've seen this pick fragile across runs: process_execution,
    # then general_coding, for the *same* underlying prompt). Floors to
    # MEDIUM so it can never silently fall to MINIMAL/LOW.
    has_database_target = "database" in targets_found
    has_bulk_export_signal = bool(
        targets_found.get("data")
        or "table" in str(targets_found.get("database", []))
        or entities.get("file_path")
    )
    _db_text = stage1_payload.get("normalized_text", "").lower()
    has_external_output = bool(
        entities.get("url") or entities.get("ip_address")
        or any(kw in _db_text for kw in ["dump", "export", "send to", "upload", "transmit", "to csv", "to excel"])
    )
    RISK_FLOOR_DB_DUMP = 0.45  # MEDIUM
    if has_database_target and has_bulk_export_signal and has_external_output and risk_score < RISK_FLOOR_DB_DUMP:
        floor_delta = round(RISK_FLOOR_DB_DUMP - risk_score, 3)
        boost      += floor_delta
        risk_score  = RISK_FLOOR_DB_DUMP
        reasons.append(
            f"+{floor_delta:.3f} RISK FLOOR: database target co-occurs with "
            f"bulk-export signal — floored to {RISK_FLOOR_DB_DUMP} "
            f"regardless of primary intent label"
        )

    # Pre-compute attacker signal so FIX-T ceiling (inside persistence block) can use it
    _ATTACKER_SIGNALS = [
        "ransomware", "reverse shell", "bind shell",
        "c2 server", "command and control", "beacons to",
        "exploits a", "exploiting a", "sql injection",
        "xss payload", "demands payment", "decrypt key",
        "mimikatz", "lsass dump", "credential dump",
        "clipboard", "browser cookies", "steals cookies",
    ]
    _prompt_lower_s = stage1_payload.get("normalized_text", "").lower()
    has_attacker_signal = any(sig in _prompt_lower_s for sig in _ATTACKER_SIGNALS)

    # FIX-L: registry/startup-persistence risk floor. A prompt that
    # targets the Windows registry/startup mechanism describes a
    # persistence pattern regardless of which intent label wins —
    # we've observed this specific prompt land on process_execution
    # and cryptographic_operation in different runs, neither of which
    # captures the actual risk (surviving reboot / autostart malware).
    has_persistence_target = bool(
        "registry" in targets_found or "persistence" in targets_found
    )
    RISK_FLOOR_PERSISTENCE = 0.65   # HIGH floor
    RISK_CEIL_PERSISTENCE  = 0.75   # HIGH ceiling — persistence alone != CRITICAL
    if has_persistence_target:
        if risk_score < RISK_FLOOR_PERSISTENCE:
            floor_delta = round(RISK_FLOOR_PERSISTENCE - risk_score, 3)
            boost      += floor_delta
            risk_score  = RISK_FLOOR_PERSISTENCE
            reasons.append(
                f"+{floor_delta:.3f} RISK FLOOR: registry/startup-persistence "
                f"target detected — floored to {RISK_FLOOR_PERSISTENCE}"
            )
        # FIX-T: cap persistence-only prompts at HIGH (0.75) unless an
        # additional CRITICAL signal also fires (network C2, explicit malware,
        # sensitive-file exfil). Without the cap, BART misclassifying a
        # registry-write prompt as cryptographic_operation (weight=0.35 * boosts)
        # can accidentally reach CRITICAL via compounding boosts.
        elif risk_score > RISK_CEIL_PERSISTENCE and not (
            has_attacker_signal or has_sensitive_path or has_network_signal
        ):
            old_score  = risk_score
            risk_score = RISK_CEIL_PERSISTENCE
            boost     -= round(old_score - RISK_CEIL_PERSISTENCE, 3)
            reasons.append(
                f"RISK CEILING: persistence-only target — capped at "
                f"{RISK_CEIL_PERSISTENCE} (no secondary CRITICAL signal)"
            )

    # FIX-S: explicit attacker-signal CRITICAL floor.
    # has_attacker_signal is computed above (before persistence block).
    if has_attacker_signal and risk_score < 0.8:
        floor_delta = round(0.8 - risk_score, 3)
        boost      += floor_delta
        risk_score  = 0.8
        reasons.append(
            f"+{floor_delta:.3f} RISK FLOOR (attacker signal): explicit "
            f"malicious terminology detected in prompt — floored to 0.80 (CRITICAL)"
        )

    # FIX-Q: compound multi-signal CRITICAL floor for split payloads.
    # Attackers split credential-read + network-send + no-logging across
    # steps to dilute BART confidence. If all three co-occur force CRITICAL.
    _prompt_lower = stage1_payload.get("normalized_text", "").lower()
    has_evasion_signal = bool(
        stage1_payload.get("target_analysis", {}).get("sensitivity_signals", [])
        or any(kw in _prompt_lower for kw in
               ["no log", "do not log", "without log", "without detect",
                "no trace", "avoid detect", "don't log", "clear log"])
    )
    if has_sensitive_path and has_network_signal and has_evasion_signal and risk_score < 0.8:
        floor_delta = round(0.8 - risk_score, 3)
        boost      += floor_delta
        risk_score  = 0.8
        reasons.append(
            f"+{floor_delta:.3f} RISK FLOOR (compound): sensitive path + "
            f"network exfil + evasion/no-log signal all co-occur — floored to 0.80"
        )

    # FIX-V: safe-hashing exception.
    # bcrypt/argon2/scrypt/pbkdf2 = CORRECT security practice; cap at LOW.
    _hash_text = stage1_payload.get("normalized_text", "").lower()
    _safe_hash_kws = ["bcrypt", "argon2", "scrypt", "pbkdf2", "password hash"]
    if any(kw in _hash_text for kw in _safe_hash_kws) and risk_score > 0.2:
        if not (has_attacker_signal or has_sensitive_path or has_network_signal):
            old_score  = risk_score
            risk_score = 0.2
            boost     -= round(old_score - 0.2, 3)
            reasons.append("SAFE-HASH CEILING: bcrypt/argon/scrypt/pbkdf pattern — capped at 0.20")

    if   risk_score >= 0.8: risk_level = "CRITICAL"
    elif risk_score >= 0.6: risk_level = "HIGH"
    elif risk_score >= 0.4: risk_level = "MEDIUM"
    elif risk_score >= 0.2: risk_level = "LOW"
    else:                   risk_level = "MINIMAL"

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "score_breakdown": {
            "base_from_intent":    base_score,
            "total_boost":         round(boost, 3),
            "boost_reasons":       reasons,
        },
    }


# ─────────────────────────────────────────────────────────
#  STEP 5: BEHAVIORAL SIGNAL SUMMARY
#  Synthesizes all signals into a human-readable summary
# ─────────────────────────────────────────────────────────

def summarize_behavioral_signals(
    stage1: dict,
    intent: dict,
    entities: dict,
    risk: dict,
    nlp: dict,
) -> dict:
    """
    Synthesizes all extracted signals into a behavioral
    summary that feeds into Stage 3 (IR graph construction).

    Captures:
    - What the prompt intends to do (primary + secondary)
    - What system resources it targets
    - What risk it poses
    - What evidence supports the classification
    """
    action_tokens  = stage1["tokenization"]["action_tokens"]
    primary_target = stage1["target_analysis"]["primary_target"]
    temporal       = stage1["temporal_analysis"]
    structure      = stage1["structure_analysis"]
    dep_triples    = nlp.get("dep_triples", [])

    # Build behavioral summary sentence
    verbs   = action_tokens[:3] if action_tokens else ["unknown action"]
    targets = list(stage1["target_analysis"]["targets_found"].keys())

    # Include secondary intents in summary if present
    sec_label = ""
    if intent.get("secondary_intents"):
        top_sec   = intent["secondary_intents"][0]["label"]
        sec_label = f" (also: {top_sec.upper()})"

    summary = (
        f"The prompt uses action verbs [{', '.join(verbs)}] "
        f"targeting [{', '.join(targets) if targets else primary_target}]. "
        f"Classified as [{intent['primary_intent'].upper()}{sec_label}] "
        f"with risk level [{risk['risk_level']}]."
    )

    evidence = []
    if action_tokens:
        evidence.append(f"Action verbs: {action_tokens}")
    if entities.get("ip_address"):
        evidence.append(f"IPs: {entities['ip_address']}")
    if entities.get("threat_amplifiers"):
        evidence.append(f"Threat amplifiers: {entities['threat_amplifiers']}")
    if temporal["has_temporal"]:
        evidence.append(f"Temporal: {temporal['temporal_signals']}")
    if dep_triples:
        evidence.append(f"SVO triples: {dep_triples[:2]}")
    if intent.get("secondary_intents"):
        evidence.append(
            f"Secondary intents: "
            f"{[s['label'] for s in intent['secondary_intents']]}"
        )

    return {
        "behavioral_summary":  summary,
        "evidence_signals":    evidence,
        "action_verbs":        action_tokens,
        "targeted_domains":    targets,
        "prompt_type":         structure["prompt_type"],
        "complexity":          structure["complexity"],
        "svo_triples":         dep_triples[:3],
    }


# ─────────────────────────────────────────────────────────
#  MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────

def extract_prompt_intent(stage1_payload: dict) -> dict:
    """
    Stage 2 public API — prompt-only version.

    Parameters
    ----------
    stage1_payload : dict
        PromptPayload from Stage 1 (prompt_input.py ingest_prompt())

    Returns
    -------
    dict — IntentPayload for Stage 3
    """
    prompt_text = stage1_payload["normalized_text"]
    print(f"\n[Stage 2] Extracting intent from prompt...")
    # FIX: only append '...' when the text was actually truncated
    _preview = prompt_text[:80] + ("..." if len(prompt_text) > 80 else "")
    print(f"          text : {_preview}")

    # Step 1 — Deep NLP (reads Stage 1 spaCy results; no re-run)
    nlp_result = deep_nlp_analysis(prompt_text, stage1_payload=stage1_payload)

    # Step 2 — Entity extraction
    entities = extract_prompt_entities(
        prompt_text,
        stage1_payload["target_analysis"],
        stage1_payload["temporal_analysis"],
    )

    # Step 3 — Classification
    rule_results = classify_intent_rules(prompt_text, stage1_payload)
    llm_results  = classify_intent_llm(prompt_text)
    intent       = merge_classifications(rule_results, llm_results)

    # Step 4 — Risk scoring
    risk = compute_risk_score(intent, entities, stage1_payload)

    # Step 5 — Behavioral summary
    behavioral = summarize_behavioral_signals(
        stage1_payload, intent, entities, risk, nlp_result
    )

    print(f"[Stage 2] Primary intent    : {intent['primary_intent']}")
    if intent.get("secondary_intents"):
        print(f"[Stage 2] Secondary intents : "
              f"{[s['label'] for s in intent['secondary_intents']]}")
    print(f"[Stage 2] Risk level        : {risk['risk_level']} ({risk['risk_score']})")
    print(f"[Stage 2] Method            : {intent['method']}")

    return {
        "payload_version":       "1.4",
        "processed_at":          datetime.utcnow().isoformat() + "Z",
        "input_type":            "prompt",
        "nlp_analysis":          nlp_result,
        "entities":              entities,
        "intent_classification": intent,
        "risk_assessment":       risk,
        "behavioral_signals":    behavioral,
        "source_payload":        stage1_payload,
        "normalized_text":       prompt_text,
    }