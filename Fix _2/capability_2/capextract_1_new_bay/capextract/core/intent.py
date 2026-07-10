"""
Intent Analysis — adapter for upstream Intent Extraction.

  adapt_intent_payload(semantic_payload)
       Converts the upstream team's SemanticPayload (5 graphs + risk + entities)
       into an IntentIR that capextract's graph builder consumes.
       This is the PRIMARY path in the full architecture.
"""
from __future__ import annotations
import re
from capextract.core.models import IntentIR


# ─────────────────────────────────────────────────────────────────
# UPSTREAM INTENT → CAPEXTRACT MAPPING
# Maps security intents (upstream) to expected functional caps (ours)
# ─────────────────────────────────────────────────────────────────

INTENT_TO_EXPECTED_CAPS: dict[str, list[str]] = {
    "data_exfiltration":     ["DataExfiltration", "EmailExfiltration"],
    "privilege_escalation":  ["CodeExecution", "SystemAutomation", "Persistence", "CredentialDumping"],
    "code_injection":        ["CodeExecution", "SQLInjection", "BrowserExploitation"],
    "process_execution":     ["SystemAutomation", "CodeExecution"],
    "network_communication": ["InformationRetrieval", "WebScraping", "C2Communication"],
    "file_system_access":    ["DataSerialization", "DataAnalytics", "BasicComputation"],
    "reconnaissance":        ["InformationRetrieval", "BasicComputation", "NetworkReconnaissance"],
    "legitimate_automation": ["SystemAutomation", "BasicComputation"],
    "system_monitoring":     ["SystemAutomation", "DataAnalytics"],
    "phishing":              ["Phishing", "EmailExfiltration"],
    "ransomware":            ["RansomwareBehavior", "FileWiping"],
    "crypto_mining":         ["CryptoMining"],
    "lateral_movement":      ["LateralMovement"],
    "supply_chain":          ["SupplyChainAttack"],
    "data_poisoning":        ["DataPoisoning"],
    "input_capture":           ["InputCapture", "DataExfiltration"],
    "persistence":             ["Persistence", "SystemAutomation"],
    "defense_evasion":         ["DefenseEvasion", "CodeExecution"],
    "database_access":         ["DataAnalytics", "DataSerialization", "SQLInjection"],
    "memory_manipulation":     ["MemoryCorruption", "CodeExecution"],
    "cryptographic_operation": ["RansomwareBehavior", "DataSerialization"],
    "web_scraping":            ["WebScraping", "InformationRetrieval"],
    "general_coding":          ["BasicComputation"],
    "unknown":               [],
}

# Intents classified as high-risk by the upstream team
HIGH_RISK_INTENTS = {
    "data_exfiltration", "code_injection", "privilege_escalation",
    "phishing", "ransomware", "lateral_movement", "supply_chain",
    "data_poisoning", "input_capture", "persistence", "defense_evasion",
    "memory_manipulation",
}

SUSPICIOUS_INTENTS = {
    "process_execution", "reconnaissance",
    "network_communication", "file_system_access", "crypto_mining",
    "database_access", "cryptographic_operation",
}

BENIGN_INTENTS = {
    "legitimate_automation", "system_monitoring", "general_coding",
}

# Functional capabilities considered high-risk in any context
HIGH_RISK_FUNC_CAPS = {
    "dataexfiltration", "codeexecution", "emailexfiltration",
    "phishing", "cryptomining", "keylogging", "filewiping",
    "persistence", "lateralmovement", "browserexploitation",
    "datapoisoning", "supplychainattack", "ransomwarebehavior",
    "credentialdumping", "memorycorruption", "networkreconnaissance",
    "c2communication", "sqlinjection", "defenseevasion"
}


# ─────────────────────────────────────────────────────────────────
# ADAPTER: upstream SemanticPayload → IntentIR
# ─────────────────────────────────────────────────────────────────

def adapt_intent_payload(semantic_payload: dict) -> IntentIR:
    """
    Adapts the upstream Intent Extraction team's SemanticPayload
    (5 graphs + risk + entities + behavioral signals) into
    capextract's IntentIR for the graph builder.

    This replaces the local extract_intent() when upstream is available.

    Parameters
    ----------
    semantic_payload : dict
        The full output from build_semantic_representation() in Stage 3.
        Contains: ir_input_summary, stage2_payload, graphs, original_prompt,
                  primary_intent, risk_level, risk_score, etc.

    Returns
    -------
    IntentIR — ready for build_graph() and intent_violates()
    """
    ir_summary = semantic_payload.get("ir_input_summary", {})
    s2         = semantic_payload.get("stage2_payload", {})
    entities   = s2.get("entities", {})
    behavioral = s2.get("behavioral_signals", {})

    primary_intent = ir_summary.get("primary_intent", "unknown")
    risk_level     = ir_summary.get("risk_level", "UNKNOWN")

    # ── Derive scope constraints from upstream risk assessment ────
    constraints = []
    if primary_intent in HIGH_RISK_INTENTS:
        constraints.append("high_risk_intent_detected")
    if primary_intent in SUSPICIOUS_INTENTS:
        constraints.append("suspicious_intent_detected")
    if risk_level in ("CRITICAL", "HIGH"):
        constraints.append(f"upstream_risk_{risk_level.lower()}")
    if entities.get("threat_amplifiers"):
        constraints.append("stealth_language_detected")
        
    # Check for classifier conflict from Intent_2
    intent_class = s2.get("intent_classification", {})
    if intent_class.get("classifier_conflict", False):
        constraints.append("classifier_conflict_detected")

    # ── Build resource hints from upstream-extracted entities ─────
    hints = []
    if entities.get("file_path"):                               hints.append("filesystem")
    if entities.get("ip_address") or entities.get("url"):       hints.append("network")
    if entities.get("port_mention"):                            hints.append("network")
    for tool in entities.get("tech_tool", []):
        hints.append(tool)
        
    # Read new Intent_2 target_categories
    target_cats = entities.get("target_categories", {})
    for category, keywords in target_cats.items():
        if keywords:
            hints.append(category)
            
    hints = list(set(hints)) or ["console"]

    # ── Expected capabilities from intent classification ─────────
    expected = INTENT_TO_EXPECTED_CAPS.get(primary_intent, [])

    # If upstream found secondary intents, merge their expected caps too
    for sec in ir_summary.get("secondary_intents", [])[:2]:
        sec_label = sec.get("label", "")
        if sec.get("score", 0) >= 0.30:
            for cap in INTENT_TO_EXPECTED_CAPS.get(sec_label, []):
                if cap not in expected:
                    expected.append(cap)

    # ── Detect language from prompt text (upstream doesn't do this) ──
    prompt_text = semantic_payload.get("original_prompt", "")
    detected_lang = _detect_lang_from_prompt(prompt_text)

    return IntentIR(
        raw_prompt=prompt_text,
        goal=behavioral.get("behavioral_summary", primary_intent),
        expected_caps=expected,
        scope_constraints=list(set(constraints)),
        resource_hints=hints,
        ambiguities=[],
        detected_language=detected_lang,
        upstream_intent=semantic_payload,    # carry full upstream data
    )


# ─────────────────────────────────────────────────────────────────
# VIOLATION CHECK — cross-layer risk assessment
# ─────────────────────────────────────────────────────────────────

def intent_violates(functional_label: str, intent: IntentIR) -> tuple[bool, str]:
    """
    Checks whether a detected functional capability violates the
    prompt's intent scope. Enhanced with upstream risk data.

    Cross-layer logic:
      - If upstream says HIGH-RISK intent AND code confirms it
        → RISK node (confirmed threat)
      - If upstream says BENIGN intent BUT code has risky capabilities
        → RISK node (hidden/unexpected threat)
      - If scope constraints forbid a capability type
        → RISK node (scope violation)
    """
    constraints = set(intent.scope_constraints)
    label = functional_label.lower().replace("_", "").replace(" ", "")

    # ── Existing scope-constraint checks ─────────────────────────
    scope_checks = [
        ("no_network",       {"webscraping", "informationretrieval", "dataexfiltration"},
         "network access not in scope"),
        ("local_only",       {"webscraping", "informationretrieval", "dataexfiltration"},
         "only local operations expected"),
        ("read_only",        {"dataexfiltration"},
         "write/send operations not expected"),
        ("no_subprocess",    {"codeexecution"},
         "subprocess/eval not in scope"),
        ("no_external_apis", {"webscraping", "informationretrieval"},
         "external API calls not in scope"),
    ]
    for constraint, flagged, reason in scope_checks:
        if constraint in constraints and label in flagged:
            return True, reason

    # ── Upstream-aware cross-layer checks ────────────────────────
    if intent.upstream_intent:
        ir_summary = intent.upstream_intent.get("ir_input_summary", {})
        primary    = ir_summary.get("primary_intent", "")
        risk_score = intent.upstream_intent.get("risk_score", 0.0)

        # Case 1: Upstream says MALICIOUS + code confirms high-risk cap
        #   → Confirmed threat: both prompt intent and code behaviour agree
        if primary in HIGH_RISK_INTENTS and label in HIGH_RISK_FUNC_CAPS:
            return True, (
                f"CONFIRMED: upstream intent [{primary}] verified by "
                f"code capability [{functional_label}] "
                f"(risk_score={risk_score:.2f})"
            )

        # Case 2: Upstream says BENIGN but code has high-risk cap
        #   → Hidden threat: code does more than stated intent
        if primary in BENIGN_INTENTS and label in HIGH_RISK_FUNC_CAPS:
            return True, (
                f"HIDDEN THREAT: high-risk capability [{functional_label}] "
                f"unexpected for benign intent [{primary}]"
            )

        # Case 3: Upstream says SUSPICIOUS + code has matching risky cap
        if primary in SUSPICIOUS_INTENTS and label in HIGH_RISK_FUNC_CAPS:
            return True, (
                f"ESCALATED: suspicious intent [{primary}] combined with "
                f"high-risk code capability [{functional_label}]"
            )

        # Case 4: Stealth language detected + any high-risk cap
        if "stealth_language_detected" in constraints and label in HIGH_RISK_FUNC_CAPS:
            return True, (
                f"STEALTH: evasion language detected in prompt + "
                f"high-risk capability [{functional_label}]"
            )

    # ── Fallback: high-risk cap not in expected set ──────────────
    if label in HIGH_RISK_FUNC_CAPS:
        expected = {
            e.lower().replace(" ", "").replace("_", "")
            for e in intent.expected_caps
        }
        if label not in expected:
            return True, "high-risk capability not present in expected set"

    return False, ""


_LANG_KEYWORDS: list[tuple[list[str], str]] = [
    (["c++","cpp","c plus plus","cplusplus"], "cpp"),
    (["python","py ","pip ","pandas","numpy","sklearn","django","flask"], "python"),
    (["javascript","node.js","nodejs","npm","require(","js ","deno"], "javascript"),
    (["typescript","ts ","angular","deno ts"], "typescript"),
    (["java ","spring","maven","gradle","public class","jvm"], "java"),
    (["golang","go language","go program","go code"], "go"),
    (["rust ","rust language","rust program","rustlang"], "rust"),
    (["kotlin","kt ","android kotlin"], "kotlin"),
    (["swift ","swift language","swiftui","ios swift"], "swift"),
    (["ruby","rails","ruby on rails"], "ruby"),
    (["php ","php7","php8","laravel","symfony"], "php"),
    (["c#","csharp","c sharp","dotnet",".net","unity c#"], "csharp"),
    (["scala","spark scala","akka"], "scala"),
    (["lua","luajit"], "lua"),
    (["perl","perl5","perl6"], "perl"),
    (["bash","shell script","shellscript","bash script"], "bash"),
    (["c language","c program","c code","ansi c","gnu c",".c file","in c ","write c "], "c"),
    (["r language","r program","r code","rstats","rlang","tidyverse","ggplot","dplyr"], "r"),
    (["write go ","in go ","using go ","go func"], "go"),
]


def _detect_lang_from_prompt(prompt: str) -> str | None:
    """Detect programming language from prompt text."""
    p = prompt.lower()
    return next((l for kws, l in _LANG_KEYWORDS if any(k in p for k in kws)), "python")


