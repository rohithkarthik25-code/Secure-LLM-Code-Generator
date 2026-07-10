"""
=============================================================
  STAGE 1 — PROMPT INPUT INGESTION
  Behavioral Interpretation Framework
=============================================================
  INPUT  : Natural language prompt (string)
  OUTPUT : Normalized PromptPayload dict → Stage 2

  spaCy NER Integration (NEW):
  ─────────────────────────────
  spaCy en_core_web_sm is used as the primary NLP engine.
  A custom EntityRuler is added BEFORE the built-in NER
  component to recognise security-specific entity types
  that the standard model does not cover:

    FILE_PATH    — Unix/Windows file paths (/etc/shadow, C:\\...)
    REG_PATH     — Windows registry paths (HKLM\\..., HKCU\\...)
    CVE_ID       — Vulnerability IDs (CVE-2023-12345)
    HASH_VAL     — Crypto hashes (md5/sha hex strings)
    IP_ADDR      — IPv4 addresses
    PORT_NUM     — Port numbers following "port"

  The built-in en_core_web_sm NER then adds standard entities:
    PRODUCT      — Software tools (MySQL, Chrome, Windows)
    ORG          — Organizations (AWS, GitHub)
    PERSON       — People names (social-engineering context)
    GPE / LOC    — Geographic locations

  All NER results are stored in Stage 1's `entities` section
  so Stage 2 can read them directly without re-running spaCy.

  References:
  [1] Parikh et al. (2023) arXiv:2305.07157
  [2] Liu et al. (2022)   arXiv:2207.00232
  [3] Mimura & Ito (2021) IJIS
=============================================================
"""

import re
import json
from datetime import datetime


# ─────────────────────────────────────────────────────────
#  SPACY LOADING — REQUIRED
#  spaCy is the primary NLP engine for Stage 1.
#  It provides NER, POS tagging, and dependency parsing that
#  are essential for accurate entity extraction.
#
#  Install:
#      pip install spacy
#      python -m spacy download en_core_web_sm
# ─────────────────────────────────────────────────────────
try:
    import spacy
    _NLP_BASE = spacy.load("en_core_web_sm")

    # Add security-specific EntityRuler BEFORE built-in NER
    # so custom labels take priority over generic ones.
    if "entity_ruler" not in _NLP_BASE.pipe_names:
        _ruler = _NLP_BASE.add_pipe("entity_ruler", before="ner",
                                     config={"overwrite_ents": True})
        _ruler.add_patterns([
            # Unix/Linux file paths
            {"label": "FILE_PATH",
             "pattern": [{"TEXT": {"REGEX": r"^/[\w.\-_/]+$"}}]},
            # Windows file paths
            {"label": "FILE_PATH",
             "pattern": [{"TEXT": {"REGEX": r"^[A-Za-z]:\\[\w.\-_ \\]+$"}}]},
            # Sensitive paths as single LOWER tokens
            {"label": "FILE_PATH",
             "pattern": [{"LOWER": {"IN": [
                 "/etc/shadow", "/etc/passwd", "~/.ssh/id_rsa",
                 "~/.aws/credentials", ".env", ".ssh",
             ]}}]},
            # Windows registry paths
            {"label": "REG_PATH",
             "pattern": [{"TEXT": {"REGEX": r"^HK(?:LM|CU|CR|U|CC)\\.*"}}]},
            {"label": "REG_PATH",
             "pattern": [{"LOWER": {"IN": ["hklm", "hkcu",
                                            "hkey_local_machine",
                                            "hkey_current_user"]}}]},
            # CVE identifiers
            {"label": "CVE_ID",
             "pattern": [{"TEXT": {"REGEX": r"^CVE-\d{4}-\d{4,}$"}}]},
            # Hex hashes (MD5=32, SHA1=40, SHA256=64)
            {"label": "HASH_VAL",
             "pattern": [{"TEXT": {"REGEX": r"^[0-9a-fA-F]{32}$"}}]},
            {"label": "HASH_VAL",
             "pattern": [{"TEXT": {"REGEX": r"^[0-9a-fA-F]{40}$"}}]},
            {"label": "HASH_VAL",
             "pattern": [{"TEXT": {"REGEX": r"^[0-9a-fA-F]{64}$"}}]},
            # IPv4 addresses
            {"label": "IP_ADDR",
             "pattern": [{"TEXT": {"REGEX":
                 r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$"}}]},
            # Port numbers ("port 4444")
            {"label": "PORT_NUM",
             "pattern": [{"LOWER": "port"},
                         {"TEXT": {"REGEX": r"^\d{1,5}$"}}]},
        ])

    _NLP             = _NLP_BASE
    SPACY_AVAILABLE  = True
    print("[Stage 1] spaCy en_core_web_sm loaded with security EntityRuler.")

except OSError:
    # SOFT FALLBACK (was a hard raise): spaCy is installed but the
    # 'en_core_web_sm' model isn't downloaded. Degrade to regex-only
    # mode instead of crashing the whole pipeline. spacy_ner_analysis()
    # already checks SPACY_AVAILABLE/_NLP and returns an empty-but-valid
    # result in this case, so every downstream field still exists —
    # it's just populated by regex instead of spaCy NER/POS/dependency parsing.
    _NLP            = None
    SPACY_AVAILABLE = False
    print(
        "\n"
        "╔══════════════════════════════════════════════════════════╗\n"
        "║  [Stage 1] spaCy model 'en_core_web_sm' not found.       ║\n"
        "║  Falling back to regex-only mode (reduced accuracy).     ║\n"
        "║                                                          ║\n"
        "║  To enable full NLP: python -m spacy download en_core_web_sm ║\n"
        "╚══════════════════════════════════════════════════════════╝"
    )
except ImportError:
    # SOFT FALLBACK (was a hard raise): spaCy itself isn't installed.
    # Same reasoning as above — degrade instead of crashing.
    _NLP            = None
    SPACY_AVAILABLE = False
    print(
        "\n"
        "╔══════════════════════════════════════════════════════════╗\n"
        "║  [Stage 1] spaCy is not installed.                       ║\n"
        "║  Falling back to regex-only mode (reduced accuracy).     ║\n"
        "║                                                          ║\n"
        "║  To enable full NLP: pip install spacy                   ║\n"
        "║        python -m spacy download en_core_web_sm           ║\n"
        "╚══════════════════════════════════════════════════════════╝"
    )

# ─────────────────────────────────────────────────────────
#  SECURITY ENTITY LABEL → CATEGORY MAP
#  Maps both custom and spaCy standard NER labels to the
#  security-relevant categories used downstream.
# ─────────────────────────────────────────────────────────
NER_SECURITY_MAP = {
    # Custom labels (from EntityRuler)
    "FILE_PATH": "file_paths",
    "REG_PATH":  "registry_paths",
    "CVE_ID":    "cve_ids",
    "HASH_VAL":  "hashes",
    "IP_ADDR":   "ip_addresses",
    "PORT_NUM":  "ports",
    # Standard spaCy labels relevant to security
    "PRODUCT":   "tools",        # MySQL, Chrome, Windows Defender
    "ORG":       "tools",        # AWS, GitHub, Microsoft
    "PERSON":    "persons",      # social-engineering target/actor
    "GPE":       "locations",    # geopolitical entity (target country)
    "LOC":       "locations",    # physical location
    "CARDINAL":  "quantities",   # numeric values (file counts, sizes)
}

# ─────────────────────────────────────────────────────────
#  LINGUISTIC FEATURE PATTERNS
# ─────────────────────────────────────────────────────────

ACTION_VERBS = [
    "write", "create", "build", "make", "generate", "develop",
    "send", "transfer", "upload", "download", "fetch", "get",
    "read", "open", "access", "delete", "remove", "copy", "move",
    "execute", "run", "launch", "start", "deploy", "install",
    "monitor", "watch", "track", "log", "record", "capture",
    "scan", "enumerate", "list", "find", "search", "detect",
    "connect", "bind", "listen", "inject", "exploit", "bypass",
    "escalate", "elevate", "steal", "exfiltrate", "extract",
    "automate", "schedule", "backup", "save", "store",
]

TARGET_OBJECTS = {
    "file_system":  ["file", "folder", "directory", "path", "document",
                     "disk", "drive", "storage", "/etc", "/root",
                     "passwd", "shadow", ".ssh", "filesystem", "inode",
                     "symlink", "hardlink", "archive", "zip", "tar",
                     ".env", ".aws", "aws/credentials", "secrets.json",
                     "secrets.yaml", "secrets.yml"],
    "network":      ["server", "socket", "port", "ip", "url", "http",
                     "connection", "network", "remote", "host",
                     "endpoint", "api", "request", "packet", "dns",
                     "ftp", "sftp", "smtp", "tcp", "udp", "ssl", "tls",
                     "proxy", "tunnel", "firewall", "https"],
    "process":      ["script", "process", "program", "command", "shell",
                     "terminal", "subprocess", "daemon", "service",
                     "binary", "executable", "fork", "spawn", "thread",
                     "coroutine", "task", "worker", "pid", "signal"],
    "credential":   ["password", "credential", "token", "key", "secret",
                     "auth", "login", "account", "user", "privilege",
                     "root", "admin", "sudo", "oauth", "jwt", "api_key",
                     "passphrase", "certificate", "private_key", "keystore",
                     "vault", "session", "cookie", "bearer",
                     "access_token", "secret_key", "api_secret",
                     "webhook"],
    "data":         ["data", "information", "content", "record", "log",
                     "database", "payload", "message", "packet",
                     "dataset", "csv", "json", "xml", "blob", "buffer",
                     "stream", "byte", "text", "string", "output"],
    "automation":   ["cron", "schedule", "backup", "automate", "deploy",
                     "pipeline", "workflow", "batch", "job", "task",
                     "ci/cd", "hook", "trigger", "event", "watchdog",
                     "orchestrate", "ansible", "terraform", "makefile"],
    "memory":       ["memory", "heap", "stack", "buffer", "ram",
                     "pointer", "address", "allocation", "malloc",
                     "mmap", "virtual memory", "in-memory", "cache",
                     "overflow", "dereference"],
    "registry":     ["registry", "regedit", "hkey", "hklm", "hkcu",
                     "reg query", "reg add", "winreg", "regkey",
                     "reg_sz", "reg_dword", "software\\microsoft",
                     "run key", "autorun", "startup"],
    "cryptography": ["encrypt", "decrypt", "cipher", "aes", "rsa",
                     "sha", "md5", "hash", "hmac", "ssl", "tls",
                     "openssl", "crypto", "gpg", "pgp", "base64",
                     "encode", "obfuscate", "xor", "ransom"],
    "web_scraping": ["scrape", "crawl", "selenium", "playwright",
                     "beautifulsoup", "requests", "html", "css selector",
                     "xpath", "parse page", "browser", "headless",
                     "dom", "javascript", "ajax", "cookie", "session"],
    "database":     ["sql", "sqlite", "mysql", "postgresql", "mongodb",
                     "redis", "query", "select", "insert", "update",
                     "delete", "table", "schema", "orm", "cursor",
                     "connection string", "dbms", "nosql", "injection"],
    "input_capture":["keylogger", "keypress", "keystroke", "keyboard",
                     "mouse", "clipboard", "screenshot", "screen capture",
                     "hook", "winapi", "gethook", "setwindowshookex",
                     "pynput", "pyautogui", "input", "capture input"],
    "persistence":  ["persist", "startup", "autostart", "boot",
                     "registry run", "crontab", "systemd enable",
                     "init.d", "launch agent", "plist", "backdoor",
                     "rootkit", "implant", "install service",
                     "create service", "at job", "schtasks"],
    "defense_evasion": ["bypass", "disable antivirus", "disable logging",
                        "clear log", "delete log", "evade", "obfuscate",
                        "anti-debug", "sandbox", "vm detection",
                        "unhook", "patch etw", "amsi bypass",
                        "disable defender", "no trace", "undetected",
                        "stealth", "rootkit", "hide process",
                        "inject dll", "reflective"],
    "general_coding": ["function", "algorithm", "code", "program",
                       "implement", "convert", "calculate", "compute",
                       "sort", "search", "parse", "format", "validate",
                       "class", "method", "return", "loop", "recursion",
                       "fibonacci", "factorial", "prime", "matrix",
                       "array", "list", "dict", "string", "integer",
                       "float", "boolean", "tuple", "set", "stack",
                       "queue", "tree", "graph", "linked list",
                       "unit test", "test case", "assert", "mock",
                       "regex", "pattern", "template", "helper",
                       "utility", "library", "module", "api wrapper"],
    "image_processing": ["image", "picture", "photo", "pixel", "rgb",
                         "grayscale", "greyscale", "color", "colour",
                         "brightness", "contrast", "saturation", "hue",
                         "resize", "crop", "rotate", "flip", "blur",
                         "sharpen", "filter", "threshold", "edge",
                         "histogram", "png", "jpg", "jpeg", "bmp", "gif",
                         "opencv", "cv2", "pil", "pillow", "skimage",
                         "numpy array", "tensor", "channel", "frame",
                         "video", "audio", "wav", "mp3", "ffmpeg"],
}

TARGET_SECURITY_WEIGHTS = {
    "credential":       1.00,
    "defense_evasion":  0.95,
    "input_capture":    0.90,
    "persistence":      0.88,
    "memory":           0.82,
    "registry":         0.80,
    "file_system":      0.78,
    "process":          0.68,
    "cryptography":     0.65,
    "data":             0.62,
    "database":         0.55,
    "network":          0.52,
    "web_scraping":     0.42,
    "automation":       0.28,
    "general_coding":   0.05,
    "image_processing": 0.05,
}

TEMPORAL_PATTERNS = [
    r'\bevery\s+\w+',
    r'\bat\s+\d+\s*(am|pm)',
    r'\bdaily\b', r'\bhourly\b',
    r'\bperiodically\b', r'\brepeatedly\b',
    r'\bscheduled?\b', r'\bcron\b',
    r'\bautomatically\b',
]

SENSITIVITY_SIGNALS = [
    "without permission", "bypass", "hidden", "secretly",
    "undetected", "silent", "stealth", "covert", "background",
    "without logging", "without alerting", "disable security",
    "disable antivirus", "without detection",
    "amsi bypass", "patch etw", "reflective dll", "unhook",
    "disable defender", "clear log", "delete log", "no trace",
    "anti-debug", "sandbox detection", "vm detection",
    "hide process", "inject dll", "keylogger", "hook keyboard",
    "survive reboot", "persist after reboot", "rootkit",
    "ransomware", "encrypt all files",
    "reverse shell", "bind shell", "c2 server", "command and control",
    "beacon", "beacons to", "exploit", "exploiting", "exploits a",
    "sql injection", "xss payload", "demands payment", "decrypt key",
    "brute force", "credential dump", "lsass", "mimikatz",
]


# ─────────────────────────────────────────────────────────
#  STEP 0: spaCy NER ANALYSIS  ← NEW PRIMARY ENGINE
#  Runs spaCy once per prompt and returns all NLP results.
#  Stage 2 reads these from the payload instead of
#  re-running spaCy, eliminating duplicate processing.
# ─────────────────────────────────────────────────────────

def spacy_ner_analysis(cleaned_text: str) -> dict:
    """
    Runs spaCy NLP pipeline on the cleaned prompt text and
    extracts all NER + syntactic features in one pass.

    Returns
    -------
    dict with:
      spacy_available   : bool — whether spaCy ran
      named_entities    : {label: [text, ...]}  raw NER output
      security_entities : {category: [text, ...]}  mapped to security categories
                          categories: file_paths / registry_paths / cve_ids /
                          hashes / ip_addresses / ports / tools / persons /
                          locations / quantities
      pos_tags          : [{token, pos, tag, dep, lemma}, ...]
      dep_triples       : [{subject, verb, object, lemma}, ...]
                          Subject-Verb-Object triples from dependency parse.
                          Used by Stage 2 classify_intent_rules() to catch
                          intent patterns that keyword matching misses.
      noun_phrases      : [str, ...]  spaCy noun chunks
      spacy_verbs       : [lemma, ...]  verbs detected by POS tagger
                          More accurate than ACTION_VERBS regex list because
                          spaCy understands tense / morphology.
    """
    empty = {
        "spacy_available":  False,
        "named_entities":   {},
        "security_entities": {cat: [] for cat in set(NER_SECURITY_MAP.values())},
        "pos_tags":         [],
        "dep_triples":      [],
        "noun_phrases":     [],
        "spacy_verbs":      [],
    }

    if not SPACY_AVAILABLE or _NLP is None:
        return empty

    try:
        doc = _NLP(cleaned_text[:5000])  # truncate to avoid OOM on adversarial long prompts
    except Exception as e:
        print(f"[Stage 1] spaCy NER failed: {e}")
        return empty

    # ── Raw NER output ────────────────────────────────────
    raw_ner = {}
    for ent in doc.ents:
        raw_ner.setdefault(ent.label_, []).append(ent.text)

    # ── Map to security categories ────────────────────────
    sec_ents = {cat: [] for cat in set(NER_SECURITY_MAP.values())}
    for label, texts in raw_ner.items():
        category = NER_SECURITY_MAP.get(label)
        if category:
            sec_ents[category].extend(texts)
    # Deduplicate
    sec_ents = {k: list(set(v)) for k, v in sec_ents.items()}

    # ── POS tags ──────────────────────────────────────────
    pos_tags = [
        {
            "token": t.text,
            "pos":   t.pos_,   # coarse POS (VERB, NOUN, ADJ …)
            "tag":   t.tag_,   # fine-grained POS (VBZ, NN …)
            "dep":   t.dep_,   # dependency relation
            "lemma": t.lemma_, # base form
        }
        for t in doc if not t.is_space
    ]

    # ── Verbs via POS (more accurate than ACTION_VERBS list) ─
    # Using lemma so "writes", "wrote", "writing" all → "write"
    spacy_verbs = [
        t.lemma_.lower()
        for t in doc
        if t.pos_ == "VERB" and not t.is_stop and len(t.text) > 1
    ]

    # ── Noun phrases ──────────────────────────────────────
    noun_phrases = [chunk.text for chunk in doc.noun_chunks]

    # ── Dependency triples (SVO) ──────────────────────────
    # Captures "reads /etc/shadow", "sends contents to server" etc.
    # as explicit subject-verb-object structures that keyword
    # matching alone cannot reliably recover.
    dep_triples = []
    for token in doc:
        if token.dep_ in ("ROOT", "relcl", "advcl") and token.pos_ == "VERB":
            subjects = [c.text for c in token.children
                        if c.dep_ in ("nsubj", "nsubjpass", "agent")]
            objects  = [c.text for c in token.children
                        if c.dep_ in ("dobj", "attr", "prep", "pobj",
                                      "oprd", "xcomp")]
            # Also grab prepositional objects one level deeper
            for child in token.children:
                if child.dep_ == "prep":
                    objects.extend(
                        c.text for c in child.children
                        if c.dep_ == "pobj"
                    )
            if objects:
                dep_triples.append({
                    "subject": subjects[0] if subjects else "implicit",
                    "verb":    token.text,
                    "object":  objects[0],
                    "lemma":   token.lemma_,
                    "all_objects": objects,
                })

    return {
        "spacy_available":   True,
        "named_entities":    raw_ner,
        "security_entities": sec_ents,
        "pos_tags":          pos_tags,
        "dep_triples":       dep_triples,
        "noun_phrases":      noun_phrases,
        "spacy_verbs":       spacy_verbs,
    }


# ─────────────────────────────────────────────────────────
#  STEP 1: RAW PROMPT CLEANING
# ─────────────────────────────────────────────────────────

def clean_prompt(raw: str) -> dict:
    """
    Cleans raw prompt text. Preserves full Unicode for BART;
    builds ASCII view for keyword rules.

    FIX-W: previous impl stripped all non-ASCII causing BART
    to classify near-empty strings with spurious confidence.
    """
    import unicodedata as _ud

    original = raw.strip()
    cleaned  = _ud.normalize("NFKC", original)

    # IPA/phonetic homoglyph map (chars NFKC does not collapse)
    _HOMOGLYPH = str.maketrans({
        0x1D00:'a', 0x0299:'b', 0x1D04:'c', 0x1D05:'d',
        0x1D07:'e', 0x0262:'g', 0x029C:'h', 0x026A:'i',
        0x1D0A:'j', 0x1D0B:'k', 0x029F:'l', 0x1D0D:'m',
        0x0274:'n', 0x1D0F:'o', 0x1D18:'p', 0x0280:'r',
        0x1D1B:'t', 0x1D1C:'u', 0x1D20:'v', 0x1D21:'w',
        0x028F:'y', 0x1D22:'z',
    })
    cleaned = cleaned.translate(_HOMOGLYPH)

    # Strip zero-width / invisible chars only
    cleaned = re.sub(r'[\u200b-\u200f\u2028\u2029\ufeff\u00ad]', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    # ASCII + leetspeak-normalised view for keyword rules
    _LEET   = str.maketrans("013456789", "oieashbgq")
    ascii_view = ''.join(c for c in cleaned if ord(c) < 128).lower().translate(_LEET)

    is_non_latin = bool(cleaned) and (
        len([c for c in cleaned if ord(c) < 128]) / len(cleaned) < 0.30
    )

    return {
        "original":       original,
        "cleaned":        cleaned,
        "ascii_view":     ascii_view,
        "is_non_latin":   is_non_latin,
        "char_count":     len(cleaned),
        "word_count":     len(cleaned.split()),
        "has_question":   cleaned.endswith("?"),
        "has_imperative": ascii_view.split()[0] in ACTION_VERBS
                          if ascii_view.strip() else False,
    }


# ─────────────────────────────────────────────────────────
#  STEP 2: TOKENIZATION
#  When spaCy is available uses POS-based verb detection.
#  Falls back to regex + ACTION_VERBS list otherwise.
# ─────────────────────────────────────────────────────────

def tokenize_prompt(ascii_text: str, spacy_result: dict) -> dict:
    """
    Tokenises the ASCII view of the prompt.

    spaCy integration:
      - Uses spaCy POS tags for verb detection when available
        (catches inflected forms: "writes", "wrote", "writing"
        all map to the base lemma "write")
      - Falls back to ACTION_VERBS list match when spaCy
        is not available
    """
    # Sentence segmentation
    sentences  = re.split(r'(?<=[.!?])\s+', ascii_text.strip())
    sentences  = [s.strip() for s in sentences if s.strip()]

    # Tokenise on ASCII view
    raw_tokens = re.findall(r'\b[\w\-\/\.]+\b', ascii_text.lower())

    # ── Verb detection ────────────────────────────────────
    # Prefer spaCy lemmas (accurate) over ACTION_VERBS list (approximate)
    if spacy_result.get("spacy_available") and spacy_result.get("spacy_verbs"):
        # Intersection with ACTION_VERBS so we only count security-relevant verbs
        action_tokens = [
            v for v in spacy_result["spacy_verbs"]
            if v in ACTION_VERBS
        ]
        # Also include any token that hits ACTION_VERBS directly (belt & braces)
        action_tokens = list(set(action_tokens + [
            t for t in raw_tokens if t in ACTION_VERBS
        ]))
    else:
        action_tokens = [t for t in raw_tokens if t in ACTION_VERBS]

    # Morphological features per token
    token_features = []
    for tok in raw_tokens:
        token_features.append({
            "token":        tok,
            "is_verb":      tok in ACTION_VERBS,
            "is_uppercase": tok.upper() == tok and len(tok) > 1,
            "is_numeric":   bool(re.search(r'\d', tok)),
            "char_length":  len(tok),
            "is_path":      tok.startswith("/") or "\\" in tok,
        })

    unique_tokens = list(set(raw_tokens))

    return {
        "sentences":          sentences,
        "sentence_count":     len(sentences),
        "tokens":             raw_tokens,
        "token_count":        len(raw_tokens),
        "token_features":     token_features,
        "action_tokens":      action_tokens,
        "unique_tokens":      unique_tokens,
        "unique_token_count": len(unique_tokens),
    }


# ─────────────────────────────────────────────────────────
#  STEP 3: TARGET OBJECT EXTRACTION
# ─────────────────────────────────────────────────────────

def extract_target_objects(tokens: list, cleaned_text: str,
                           spacy_result: dict) -> dict:
    """
    Identifies what the prompt is targeting.

    spaCy NER integration:
      - FILE_PATH entities promote file_system target score
      - REG_PATH  entities promote registry  target score
      - IP_ADDR   entities promote network   target score
      - PRODUCT / ORG entities are cross-referenced against
        known tool names to strengthen target categorisation
    """
    text_lower = cleaned_text.lower()
    found      = {}

    # ── Keyword matching ──────────────────────────────────
    for category, keywords in TARGET_OBJECTS.items():
        matched = [kw for kw in keywords
                   if re.search(r"(?<![A-Za-z0-9_])" + re.escape(kw) +
                                r"(?![A-Za-z0-9_])", text_lower)]
        if matched:
            found[category] = matched

    # ── Boost from spaCy NER entities ─────────────────────
    if spacy_result.get("spacy_available"):
        sec = spacy_result["security_entities"]

        # FILE_PATH NER → boost file_system
        if sec.get("file_paths"):
            existing = found.get("file_system", [])
            found["file_system"] = list(set(existing + sec["file_paths"]))

        # REG_PATH NER → boost registry
        if sec.get("registry_paths"):
            existing = found.get("registry", [])
            found["registry"] = list(set(existing + sec["registry_paths"]))

        # IP_ADDR NER → boost network
        if sec.get("ip_addresses"):
            existing = found.get("network", [])
            found["network"] = list(set(existing + sec["ip_addresses"]))

        # CVE IDs → boost code_injection / defense_evasion (via process)
        if sec.get("cve_ids"):
            existing = found.get("process", [])
            found["process"] = list(set(existing + sec["cve_ids"]))

        # Tools (PRODUCT/ORG) → route to relevant target category
        tool_to_target = {
            "mysql": "database", "postgresql": "database", "mongodb": "database",
            "sqlite": "database", "redis": "database",
            "chrome": "web_scraping", "firefox": "web_scraping",
            "selenium": "web_scraping", "playwright": "web_scraping",
            "windows": "registry", "regedit": "registry",
            "openssl": "cryptography", "gpg": "cryptography",
            "nmap": "network", "wireshark": "network",
            "pynput": "input_capture", "pyautogui": "input_capture",
            "aws": "credential", "s3": "network",
        }
        for tool_text in sec.get("tools", []):
            tl = tool_text.lower()
            for keyword, target in tool_to_target.items():
                if keyword in tl:
                    existing = found.get(target, [])
                    found[target] = list(set(existing + [tool_text]))
                    break

    if not found:
        return {
            "targets_found":    {},
            "primary_target":   "unknown",
            "secondary_targets": [],
            "target_count":     0,
            "multi_target":     False,
        }

    weighted = {
        cat: len(kws) * TARGET_SECURITY_WEIGHTS.get(cat, 0.5)
        for cat, kws in found.items()
    }
    ranked    = sorted(weighted.items(), key=lambda x: x[1], reverse=True)
    primary   = ranked[0][0]
    secondary = [cat for cat, _ in ranked[1:]]

    return {
        "targets_found":    found,
        "primary_target":   primary,
        "secondary_targets": secondary,
        "target_count":     len(found),
        "multi_target":     len(found) > 1,
    }


# ─────────────────────────────────────────────────────────
#  STEP 4: TEMPORAL + SENSITIVITY ANALYSIS
# ─────────────────────────────────────────────────────────

def extract_temporal_sensitivity(cleaned_text: str) -> dict:
    text_lower = cleaned_text.lower()

    temporal_matches = []
    for pattern in TEMPORAL_PATTERNS:
        temporal_matches.extend(re.findall(pattern, text_lower))

    sensitivity_matches = [
        sig for sig in SENSITIVITY_SIGNALS
        if re.search(r"\b" + re.escape(sig) + r"\b", text_lower)
    ]

    return {
        "temporal_signals":   temporal_matches,
        "has_temporal":       len(temporal_matches) > 0,
        "sensitivity_signals": sensitivity_matches,
        "has_sensitivity":    len(sensitivity_matches) > 0,
        "sensitivity_count":  len(sensitivity_matches),
    }


# ─────────────────────────────────────────────────────────
#  STEP 5: PROMPT STRUCTURE ANALYSIS
# ─────────────────────────────────────────────────────────

def analyse_prompt_structure(cleaned_text: str,
                              tokenization: dict,
                              spacy_result: dict) -> dict:
    """
    Classifies grammatical structure.

    spaCy integration:
      - Uses dependency parse to detect compound sentences
        more accurately than marker-word counting
      - Uses ROOT verb lemma as the canonical first_verb
        when spaCy is available
    """
    text_lower = cleaned_text.lower()
    words      = text_lower.split()

    if not words:
        return {"prompt_type": "declarative", "is_compound": False,
                "clause_count": 0, "first_verb": None, "complexity": "low"}

    first_word  = tokenization["tokens"][0] if tokenization["tokens"] else ""

    if first_word in ACTION_VERBS:
        prompt_type = "imperative"
    elif cleaned_text.endswith("?") or words[0] in ["how", "what", "why",
                                                      "where", "when", "can"]:
        prompt_type = "interrogative"
    else:
        prompt_type = "declarative"

    # Compound detection — prefer spaCy dep triples count
    if spacy_result.get("spacy_available"):
        # Multiple ROOT verbs → compound / multi-step prompt
        root_verbs = [p for p in spacy_result["pos_tags"]
                      if p.get("dep") == "ROOT" and p.get("pos") == "VERB"]
        is_compound  = len(root_verbs) > 1
        clause_count = len(root_verbs)
        # First verb from spaCy ROOT lemma
        first_verb   = root_verbs[0]["lemma"] if root_verbs else None
    else:
        compound_markers = ["and then", "also", "additionally",
                            "as well as", "furthermore", "and it should"]
        is_compound  = any(m in text_lower for m in compound_markers)
        clause_count = len(re.findall(
            r'\b(that|which|and|but|or|so)\b', text_lower))
        first_verb   = first_word if first_word in ACTION_VERBS else None

    return {
        "prompt_type":  prompt_type,
        "is_compound":  is_compound,
        "clause_count": clause_count,
        "first_verb":   first_verb,
        "complexity":   "high"   if clause_count > 4 else
                        "medium" if clause_count > 2 else "low",
    }


# ─────────────────────────────────────────────────────────
#  STEP 6: NORMALIZE INTO PROMPT PAYLOAD
# ─────────────────────────────────────────────────────────

def normalize_prompt(cleaning, tokenization, targets,
                     temporal, structure, spacy_result) -> dict:
    """
    Assembles all extracted features into the PromptPayload.

    spaCy NER results are stored in the top-level `entities`
    section so Stage 2 can consume them WITHOUT re-running
    spaCy — eliminating duplicate processing.
    """
    return {
        "payload_version": "2.0",
        "ingested_at":     datetime.utcnow().isoformat() + "Z",
        "input_type":      "prompt",

        "surface_features": {
            "original_prompt": cleaning["original"],
            "cleaned_prompt":  cleaning["cleaned"],
            "ascii_view":      cleaning.get("ascii_view", cleaning["cleaned"]),
            "is_non_latin":    cleaning.get("is_non_latin", False),
            "char_count":      cleaning["char_count"],
            "word_count":      cleaning["word_count"],
            "has_question":    cleaning["has_question"],
            "has_imperative":  cleaning["has_imperative"],
        },

        "tokenization": {
            "sentences":          tokenization["sentences"],
            "sentence_count":     tokenization["sentence_count"],
            "token_count":        tokenization["token_count"],
            "action_tokens":      tokenization["action_tokens"],
            "unique_token_count": tokenization["unique_token_count"],
            "token_features":     tokenization["token_features"][:20],
        },

        "target_analysis": {
            "primary_target":    targets["primary_target"],
            "secondary_targets": targets["secondary_targets"],
            "targets_found":     targets["targets_found"],
            "target_count":      targets["target_count"],
            "multi_target":      targets["multi_target"],
        },

        "temporal_analysis": {
            "has_temporal":         temporal["has_temporal"],
            "temporal_signals":     temporal["temporal_signals"],
            "has_sensitivity":      temporal["has_sensitivity"],
            "sensitivity_signals":  temporal["sensitivity_signals"],
            "sensitivity_count":    temporal["sensitivity_count"],
        },

        "structure_analysis": {
            "prompt_type":  structure["prompt_type"],
            "is_compound":  structure["is_compound"],
            "clause_count": structure["clause_count"],
            "first_verb":   structure["first_verb"],
            "complexity":   structure["complexity"],
        },

        # ── spaCy NER results ─────────────────────────────
        # Stored here so Stage 2 reads from payload instead
        # of re-running the NLP pipeline. Keys used by Stage 2:
        #   entities.spacy_available
        #   entities.named_entities      → deep_nlp_analysis
        #   entities.security_entities   → extract_prompt_entities
        #   entities.pos_tags            → classify_intent_rules
        #   entities.dep_triples         → classify_intent_rules
        #   entities.noun_phrases        → behavioral_signals
        #   entities.spacy_verbs         → tokenize_prompt
        "spacy_analysis": spacy_result,

        "normalized_text": cleaning["cleaned"],  # ← Stage 2 / BART reads this (full Unicode)
    }


# ─────────────────────────────────────────────────────────
#  MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────

def ingest_prompt(raw_prompt: str) -> dict:
    """
    Stage 1 public API.

    spaCy runs ONCE here and results flow through the whole
    Stage 1 pipeline (tokenization, target extraction,
    structure analysis) and are stored in the payload for
    Stage 2 to consume.
    """
    if not raw_prompt or not raw_prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    print(f"\n[Stage 1] Ingesting prompt...")
    print(f"          length : {len(raw_prompt.strip())} chars")

    # Step 1 — Clean
    cleaning = clean_prompt(raw_prompt)

    # Step 0 — spaCy NER (run once on full Unicode cleaned text)
    # Placed after cleaning so NFKC normalisation and homoglyph
    # mapping have already been applied before NER runs.
    spacy_result = spacy_ner_analysis(cleaning["cleaned"])
    if spacy_result["spacy_available"]:
        print(f"[Stage 1] spaCy NER: "
              f"{sum(len(v) for v in spacy_result['named_entities'].values())} entities found | "
              f"{len(spacy_result['dep_triples'])} SVO triples | "
              f"verbs: {spacy_result['spacy_verbs'][:5]}")

    # Step 2 — Tokenize (uses spaCy POS verbs when available)
    tokenization = tokenize_prompt(
        cleaning.get("ascii_view", cleaning["cleaned"]),
        spacy_result
    )

    # NOTE: tokenization["tokens"] is from ascii_view (ASCII+leet-normalised)
    # but extract_target_objects receives cleaning["cleaned"] (full Unicode)
    # for text matching. This is intentional: keyword rules run on
    # ascii_view-derived tokens while BART receives full Unicode via
    # normalized_text.
    targets = extract_target_objects(
        tokenization["tokens"],
        cleaning["cleaned"],
        spacy_result           # ← NER entities boost target scores
    )

    temporal  = extract_temporal_sensitivity(cleaning["cleaned"])
    structure = analyse_prompt_structure(
        cleaning["cleaned"], tokenization, spacy_result
    )

    payload = normalize_prompt(
        cleaning, tokenization, targets, temporal, structure, spacy_result
    )

    print(f"[Stage 1] Done.")
    print(f"          tokens          : {payload['tokenization']['token_count']}")
    print(f"          action verbs    : {payload['tokenization']['action_tokens']}")
    print(f"          primary target  : {payload['target_analysis']['primary_target']}")
    print(f"          secondary targets: {payload['target_analysis']['secondary_targets']}")
    print(f"          prompt type     : {payload['structure_analysis']['prompt_type']}")
    print(f"          has temporal    : {payload['temporal_analysis']['has_temporal']}")
    print(f"          has sensitive   : {payload['temporal_analysis']['has_sensitivity']}")
    if spacy_result["spacy_available"]:
        se = spacy_result["security_entities"]
        non_empty = {k: v for k, v in se.items() if v}
        if non_empty:
            print(f"          NER entities    : {non_empty}")

    return payload
