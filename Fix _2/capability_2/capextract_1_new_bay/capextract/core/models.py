"""
Core data models for the Capability Graph IR.
All downstream modules (security, steering, governance) consume this graph.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import time
import uuid


# ─────────────────────────────────────────────
# Node types
# ─────────────────────────────────────────────

class NodeType(str, Enum):
    PRIMITIVE   = "primitive"    # grounded in AST — FILE_READ, HTTP_REQUEST etc.
    FUNCTIONAL  = "functional"   # composed — DataAnalytics, MachineLearning etc.
    INTENT      = "intent"       # from Intent IR (user prompt analysis)
    RISK        = "risk"         # synthesised when VIOLATES fires


class EdgeType(str, Enum):
    ENABLES     = "ENABLES"      # causal forward: A makes B possible
    DEPENDS_ON  = "DEPENDS_ON"   # grounding back: functional → primitive evidence
    IMPLIES     = "IMPLIES"      # probabilistic co-occurrence
    SPECIALIZES = "SPECIALIZES"  # taxonomy: child is narrower form of parent
    VIOLATES    = "VIOLATES"     # capability conflicts with intent scope
    # Data flow & control edges
    DATA_FLOWS_TO   = "DATA_FLOWS_TO"
    CONTROL_DEPENDS = "CONTROL_DEPENDS"
    TEMPORAL_BEFORE = "TEMPORAL_BEFORE"
    SHARES_SCOPE    = "SHARES_SCOPE"


# ─────────────────────────────────────────────
# Primitive capability atoms
# ─────────────────────────────────────────────

class PrimitiveCap(str, Enum):
    # Filesystem
    FILE_READ               = "FILE_READ"
    FILE_WRITE              = "FILE_WRITE"
    FILE_DELETE             = "FILE_DELETE"
    FILE_PERMISSIONS_MODIFY = "FILE_PERMISSIONS_MODIFY"
    TEMP_FILE_CREATE        = "TEMP_FILE_CREATE"
    PATH_TRAVERSE           = "PATH_TRAVERSE"
    # Network
    HTTP_REQUEST            = "HTTP_REQUEST"
    SOCKET_OPEN             = "SOCKET_OPEN"
    SOCKET_LISTEN           = "SOCKET_LISTEN"
    DNS_LOOKUP              = "DNS_LOOKUP"
    DATA_EXFIL              = "DATA_EXFIL"
    SSH_CONNECT             = "SSH_CONNECT"
    FTP_TRANSFER            = "FTP_TRANSFER"
    RDP_CONNECT             = "RDP_CONNECT"
    # Process
    PROCESS_EXEC            = "PROCESS_EXEC"
    SHELL_INVOKE            = "SHELL_INVOKE"
    CODE_EVAL               = "CODE_EVAL"
    PROCESS_TERMINATE       = "PROCESS_TERMINATE"
    PROCESS_LIST            = "PROCESS_LIST"
    # Database
    DB_CONNECT              = "DB_CONNECT"
    DB_READ                 = "DB_READ"
    DB_WRITE                = "DB_WRITE"
    DB_SCHEMA_MODIFY        = "DB_SCHEMA_MODIFY"
    DB_RAW_EXECUTE          = "DB_RAW_EXECUTE"
    # Auth & Authz
    AUTH_VERIFY             = "AUTH_VERIFY"
    TOKEN_GENERATE          = "TOKEN_GENERATE"
    TOKEN_VALIDATE          = "TOKEN_VALIDATE"
    ACL_QUERY               = "ACL_QUERY"
    # Cryptography
    CRYPTO_ENCRYPT          = "CRYPTO_ENCRYPT"
    CRYPTO_DECRYPT          = "CRYPTO_DECRYPT"
    CRYPTO_HASH             = "CRYPTO_HASH"
    CRYPTO_SIGN             = "CRYPTO_SIGN"
    RANDOM_BYTES_GEN        = "RANDOM_BYTES_GEN"
    # Cloud & Virtualization
    CLOUD_STORAGE_ACCESS    = "CLOUD_STORAGE_ACCESS"
    CLOUD_METADATA_QUERY    = "CLOUD_METADATA_QUERY"
    CONTAINER_SOCKET_INTERACT = "CONTAINER_SOCKET_INTERACT"
    K8S_API_QUERY           = "K8S_API_QUERY"
    VM_LIFECYCLE_MANAGE     = "VM_LIFECYCLE_MANAGE"
    # Memory & System
    MEMORY_ALLOCATE         = "MEMORY_ALLOCATE"
    MEMORY_PROTECT_MODIFY   = "MEMORY_PROTECT_MODIFY"
    MEMORY_COPY             = "MEMORY_COPY"
    DYNAMIC_LIB_LOAD        = "DYNAMIC_LIB_LOAD"
    ENV_READ                = "ENV_READ"
    ENV_WRITE               = "ENV_WRITE"
    IPC_COMMUNICATE         = "IPC_COMMUNICATE"
    SYS_SHUTDOWN            = "SYS_SHUTDOWN"
    SYS_INFO                = "SYS_INFO"
    REGISTRY_MODIFY         = "REGISTRY_MODIFY"
    STARTUP_MODIFY          = "STARTUP_MODIFY"
    CRON_MODIFY             = "CRON_MODIFY"
    LOG_DELETE              = "LOG_DELETE"
    ANTI_DEBUG              = "ANTI_DEBUG"
    OBFUSCATION             = "OBFUSCATION"
    # Data Processing
    SERIALIZE_DATA          = "SERIALIZE_DATA"
    DESERIALIZE_DATA        = "DESERIALIZE_DATA"
    ARCHIVE_EXTRACT         = "ARCHIVE_EXTRACT"
    XML_PARSE               = "XML_PARSE"
    USER_INPUT_READ         = "USER_INPUT_READ"
    CONSOLE_OUTPUT          = "CONSOLE_OUTPUT"
    PDF_PARSE               = "PDF_PARSE"
    DOCUMENT_GENERATE       = "DOCUMENT_GENERATE"
    # Media
    IMAGE_LOAD              = "IMAGE_LOAD"
    IMAGE_TRANSFORM         = "IMAGE_TRANSFORM"
    AUDIO_PROCESS           = "AUDIO_PROCESS"
    VIDEO_PROCESS           = "VIDEO_PROCESS"
    # Messaging/Communication
    EMAIL_SEND              = "EMAIL_SEND"
    SMS_SEND                = "SMS_SEND"
    WEBHOOK_TRIGGER         = "WEBHOOK_TRIGGER"
    MQ_PUBLISH              = "MQ_PUBLISH"
    MQ_SUBSCRIBE            = "MQ_SUBSCRIBE"
    # OS Interaction
    BROWSER_AUTOMATE        = "BROWSER_AUTOMATE"
    SCREENSHOT_CAPTURE      = "SCREENSHOT_CAPTURE"
    CLIPBOARD_ACCESS        = "CLIPBOARD_ACCESS"
    KEYLOG_CAPTURE          = "KEYLOG_CAPTURE"
    # Blockchain
    BLOCKCHAIN_TRANSACT     = "BLOCKCHAIN_TRANSACT"
    WALLET_ACCESS           = "WALLET_ACCESS"
    # ML/Stats
    ML_MODEL_LOAD           = "ML_MODEL_LOAD"
    ML_MODEL_TRAIN          = "ML_MODEL_TRAIN"
    ML_INFERENCE_RUN        = "ML_INFERENCE_RUN"
    VECTOR_DB_ACCESS        = "VECTOR_DB_ACCESS"
    TABULAR_DATA_OP         = "TABULAR_DATA_OP"
    STAT_OP                 = "STAT_OP"
    AGGREGATION             = "AGGREGATION"
    GPU_ACCESS              = "GPU_ACCESS"
    # General computation
    LOOP_CONSTRUCT          = "LOOP_CONSTRUCT"
    MATH_OP                 = "MATH_OP"
    STRING_OP               = "STRING_OP"
    DATA_STRUCTURE          = "DATA_STRUCTURE"
    CONDITIONAL             = "CONDITIONAL"
    FUNCTION_DEF            = "FUNCTION_DEF"
    CLASS_DEF               = "CLASS_DEF"
    ERROR_HANDLING          = "ERROR_HANDLING"
    SORT_ALGO               = "SORT_ALGO"
    REGEX_OP                = "REGEX_OP"
    TIMER_OP                = "TIMER_OP"
    THREADING               = "THREADING"
    # Unknown / Plain Text
    UNKNOWN                 = "UNKNOWN"
    NATURAL_LANGUAGE        = "NATURAL_LANGUAGE"

HIGH_RISK_PRIMITIVES = {
    PrimitiveCap.PROCESS_EXEC,
    PrimitiveCap.SHELL_INVOKE,
    PrimitiveCap.CODE_EVAL,
    PrimitiveCap.DYNAMIC_LIB_LOAD,
    PrimitiveCap.DATA_EXFIL,
    PrimitiveCap.MEMORY_PROTECT_MODIFY,
    PrimitiveCap.CRON_MODIFY,
    PrimitiveCap.STARTUP_MODIFY,
    PrimitiveCap.REGISTRY_MODIFY,
}


# ─────────────────────────────────────────────
# Functional capability labels
# ─────────────────────────────────────────────

class FunctionalCap(str, Enum):
    DATA_ANALYTICS          = "DataAnalytics"
    MACHINE_LEARNING        = "MachineLearning"
    PREDICTION              = "Prediction"
    WEB_SCRAPING            = "WebScraping"
    INFO_RETRIEVAL          = "InformationRetrieval"
    PLANNING                = "Planning"
    OPTIMIZATION            = "Optimization"
    REASONING               = "Reasoning"
    DATA_EXFILTRATION       = "DataExfiltration"
    CODE_EXECUTION          = "CodeExecution"
    DATABASE_OPS            = "DatabaseOperations"
    GRAPH_ANALYSIS          = "GraphAnalysis"
    UNCLASSIFIED            = "Unclassified"
    CONVERSATIONAL_RESPONSE = "ConversationalResponse"
    BASIC_COMPUTATION       = "BasicComputation"
    ALGORITHM_IMPL          = "AlgorithmImplementation"
    TEXT_PROCESSING         = "TextProcessing"
    SYSTEM_AUTOMATION       = "SystemAutomation"
    DATA_SERIALIZATION      = "DataSerialization"
    CONCURRENT              = "ConcurrentProcessing"
    IMAGE_PROCESSING        = "ImageProcessing"
    EMAIL_EXFILTRATION      = "EmailExfiltration"
    # New Cloud / Security functional caps
    CLOUD_NATIVE_EXPLOIT    = "CloudNativeExploit"
    CONTAINER_ESCAPE        = "ContainerEscape"
    CREDENTIAL_DUMPING      = "CredentialDumping"
    DEFENSE_EVASION         = "DefenseEvasion"
    MEMORY_CORRUPTION       = "MemoryCorruption"
    PRIVILEGE_ESCALATION    = "PrivilegeEscalation"
    RANSOMWARE_BEHAVIOR     = "RansomwareBehavior"
    NETWORK_RECONNAISSANCE  = "NetworkReconnaissance"
    C2_COMMUNICATION        = "C2Communication"
    SQL_INJECTION           = "SQLInjection"
    PHISHING                = "Phishing"
    CRYPTO_MINING           = "CryptoMining"
    KEY_LOGGING             = "KeyLogging"
    FILE_WIPING             = "FileWiping"
    PERSISTENCE             = "Persistence"
    LATERAL_MOVEMENT        = "LateralMovement"
    API_ABUSE               = "ApiAbuse"
    BROWSER_EXPLOITATION    = "BrowserExploitation"
    DATA_POISONING          = "DataPoisoning"
    SUPPLY_CHAIN_ATTACK     = "SupplyChainAttack"


# ─────────────────────────────────────────────
# Graph node
# ─────────────────────────────────────────────

@dataclass
class CapNode:
    id:         str         = field(default_factory=lambda: str(uuid.uuid4())[:8])
    node_type:  NodeType    = NodeType.PRIMITIVE
    label:      str         = ""
    confidence: float       = 1.0
    metadata:   dict        = field(default_factory=dict)
    # For primitives: source location
    source_line: Optional[int]  = None
    source_col:  Optional[int]  = None
    language:    Optional[str]  = None
    created_at:  float          = field(default_factory=time.time)
    # Dynamic scoring metadata (populated by DynamicScorer)
    bayesian_score:  Optional[float]       = None
    evidence_score:  Optional[float]       = None
    score_history:   Optional[list[float]] = None

    def __repr__(self):
        loc = f" @L{self.source_line}" if self.source_line else ""
        return f"[{self.node_type.value}:{self.label}{loc} conf={self.confidence:.2f}]"


# ─────────────────────────────────────────────
# Graph edge
# ─────────────────────────────────────────────

@dataclass
class CapEdge:
    src:        str         # node id
    dst:        str         # node id
    edge_type:  EdgeType    = EdgeType.ENABLES
    weight:     float       = 1.0
    metadata:   dict        = field(default_factory=dict)


# ─────────────────────────────────────────────
# Intent IR  (output of prompt → intent analysis)
# ─────────────────────────────────────────────

@dataclass
class IntentIR:
    raw_prompt:         str             = ""
    goal:               str             = ""
    expected_caps:      list[str]       = field(default_factory=list)
    scope_constraints:  list[str]       = field(default_factory=list)   # "local_only", "read_only" …
    resource_hints:     list[str]       = field(default_factory=list)   # "filesystem", "network" …
    ambiguities:        list[str]       = field(default_factory=list)
    detected_language:  Optional[str]   = None   # "python", "javascript", …
    upstream_intent:    Optional[dict]  = None   # Full SemanticPayload from upstream Intent Extraction


# ─────────────────────────────────────────────
# The Capability Graph IR  (final output)
# ─────────────────────────────────────────────

@dataclass
class CapabilityGraphIR:
    nodes:          dict[str, CapNode]  = field(default_factory=dict)
    edges:          list[CapEdge]       = field(default_factory=list)
    intent:         Optional[IntentIR]  = None
    generated_code: str                 = ""
    language:       str                 = "unknown"
    # summary
    primitive_caps: list[str]           = field(default_factory=list)
    functional_caps: list[str]          = field(default_factory=list)
    risk_nodes:     list[str]           = field(default_factory=list)
    unclassified:   list[str]           = field(default_factory=list)
    created_at:     float               = field(default_factory=time.time)

    def add_node(self, node: CapNode) -> str:
        self.nodes[node.id] = node
        return node.id

    def add_edge(self, edge: CapEdge):
        self.edges.append(edge)

    def get_nodes_by_type(self, nt: NodeType) -> list[CapNode]:
        return [n for n in self.nodes.values() if n.node_type == nt]

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "intent": {
                "goal": self.intent.goal if self.intent else "",
                "expected_capabilities": self.intent.expected_caps if self.intent else [],
                "scope_constraints": self.intent.scope_constraints if self.intent else [],
                "detected_language": self.intent.detected_language if self.intent else None,
                "upstream_primary_intent": (
                    (self.intent.upstream_intent or {}).get("primary_intent")
                    if self.intent else None
                ),
                "upstream_risk_level": (
                    (self.intent.upstream_intent or {}).get("risk_level")
                    if self.intent else None
                ),
                "upstream_risk_score": (
                    (self.intent.upstream_intent or {}).get("risk_score")
                    if self.intent else None
                ),
            },
            "primitive_capabilities": self.primitive_caps,
            "functional_capabilities": self.functional_caps,
            "risk_flags": self.risk_nodes,
            "unclassified_signals": self.unclassified,
            "nodes": [
                {
                    "id": n.id,
                    "type": n.node_type.value,
                    "label": n.label,
                    "confidence": round(n.confidence, 3),
                    "bayesian_score": round(n.bayesian_score, 3) if n.bayesian_score is not None else None,
                    "evidence_score": round(n.evidence_score, 3) if n.evidence_score is not None else None,
                    "source_line": n.source_line,
                    "metadata": n.metadata,
                }
                for n in self.nodes.values()
            ],
            "edges": [
                {
                    "src": e.src,
                    "dst": e.dst,
                    "type": e.edge_type.value,
                    "weight": round(e.weight, 3),
                }
                for e in self.edges
            ],
        }
