"""
AST/CPG parser — pure Tree-sitter parsing.

Supports Python, JavaScript, TypeScript, Java, Go, C++, Rust, C#, Ruby, PHP via Tree-sitter.

Extracts primitive capability signals from:
  - import statements
  - function calls
  - attribute access chains

Designed for incremental use: feed(code) can be called repeatedly
as new tokens arrive; only dirty subtrees are re-parsed.
"""

from __future__ import annotations
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from tree_sitter import Language, Parser, Node

from capextract.core.models import PrimitiveCap, CapNode, NodeType, HIGH_RISK_PRIMITIVES
from capextract.core.vector_mapper import VectorMapper


# ─────────────────────────────────────────────
# Build language objects lazily
# ─────────────────────────────────────────────

def _make_parser(lang_obj, name: str) -> Parser:
    if hasattr(lang_obj, 'language'):
        lang = Language(lang_obj.language())
    elif hasattr(lang_obj, f'language_{name}'):
        lang_func = getattr(lang_obj, f'language_{name}')
        lang = Language(lang_func())
    else:
        raise AttributeError(f"Could not find language function in {lang_obj}")
    p = Parser(lang)
    return p


_PARSERS: dict[str, Parser] = {}
_TS_SUPPORTED = {
    "python", "javascript", "typescript", "java", "go",
    "cpp", "rust", "c_sharp", "ruby", "php"
}

def get_parser(language: str) -> Optional[Parser]:
    lang_key = language.lower()
    if lang_key not in _TS_SUPPORTED:
        return None
    if lang_key in _PARSERS:
        return _PARSERS[lang_key]

    try:
        if lang_key == "python":
            import tree_sitter_python as ts_python
            _PARSERS["python"] = _make_parser(ts_python, "python")
        elif lang_key in ("javascript", "typescript"):
            import tree_sitter_javascript as ts_javascript
            p = _make_parser(ts_javascript, "javascript")
            _PARSERS["javascript"] = p
            _PARSERS["typescript"] = p
        elif lang_key == "java":
            import tree_sitter_java as ts_java
            _PARSERS["java"] = _make_parser(ts_java, "java")
        elif lang_key == "go":
            import tree_sitter_go as ts_go
            _PARSERS["go"] = _make_parser(ts_go, "go")
        elif lang_key == "cpp":
            import tree_sitter_cpp as ts_cpp
            _PARSERS["cpp"] = _make_parser(ts_cpp, "cpp")
        elif lang_key == "rust":
            import tree_sitter_rust as ts_rust
            _PARSERS["rust"] = _make_parser(ts_rust, "rust")
        elif lang_key == "c_sharp":
            import tree_sitter_c_sharp as ts_csharp
            _PARSERS["c_sharp"] = _make_parser(ts_csharp, "c_sharp")
        elif lang_key == "ruby":
            import tree_sitter_ruby as ts_ruby
            _PARSERS["ruby"] = _make_parser(ts_ruby, "ruby")
        elif lang_key == "php":
            import tree_sitter_php as ts_php
            _PARSERS["php"] = _make_parser(ts_php, "php")
    except ImportError as e:
        import sys
        print(f"Warning: Could not import tree-sitter language package for '{lang_key}': {e}", file=sys.stderr)
        return None

    return _PARSERS.get(lang_key)


# Language detection heuristics (order: more specific first)
_LANG_HINTS: list[tuple[str, str]] = [
    ("php",        r'<\?php|\$\w+\s*=|function\s+\w+\s*\(|echo\s|->(\w+)|::(\w+)'),
    ("c_sharp",    r'using\s+System|namespace\s+\w+|Console\.\w+|class\s+\w+\s*:\s*\w+'),
    ("rust",       r'\bfn\s+\w+|let\s+mut\s|impl\s+\w+|pub\s+fn|use\s+std::|println!\s*\(|vec!\[|match\s+\w+\s*\{'),
    ("go",         r'package\s+\w+|import\s*\(|func\s+\w+\s*\(|:=|fmt\.|go\s+func'),
    ("cpp",        r'\b#include\s*<|using\s+namespace\s+std|cout\s*<<|cin\s*>>|std::|nullptr|template\s*<|::std'),
    ("java",       r'\bpublic\s+class\b|\bimport\s+java\.|\@Override|System\.out\.print'),
    ("ruby",       r'\bdef\s+\w+|^\s*end\b|puts\s|require\s+["\']|class\s+\w+\s*<|attr_accessor'),
    ("typescript", r':\s*(string|number|boolean|void)\b|interface\s+\w+\s*\{|import\s+\{'),
    ("javascript", r'\brequire\s*\(|\bconst\b|\blet\b|\bfetch\s*\(|=>|module\.exports'),
    ("python",     r'\bdef\b|\bimport\b|\bprint\s*\(|\bpandas\b|\bnumpy\b|from\s+\w+\s+import'),
]


@dataclass
class ParsedSignal:
    """A single primitive capability signal extracted from an AST node."""
    capability: PrimitiveCap
    source_line: int
    source_col:  int
    trigger:     str          # what pattern fired
    confidence:  float = 1.0
    metadata:    dict  = field(default_factory=dict)


class IncrementalParser:
    """
    Wraps Tree-sitter for incremental parsing across multiple languages.
    Call feed(code_so_far) as the LLM streams tokens.
    Returns new ParsedSignals since the last call.
    """

    def __init__(self, language: str = "python"):
        self.language = language.lower()
        self._parser = get_parser(self.language)
        self._has_treesitter = self._parser is not None
        self._tree    = None
        self._last_code = b""
        self._seen_signals: set[str] = set()   # dedup key -> already emitted
        
        # Load Vector Mapper (replacing tier 1 rules)
        self.vector_mapper = VectorMapper.get_instance()

    def feed(self, code: str) -> list[ParsedSignal]:
        """
        Parse code_so_far incrementally.
        Returns only NEW signals not seen in previous calls.
        """
        all_signals: list[ParsedSignal] = []

        if self._has_treesitter and self._parser is not None:
            code_bytes = code.encode("utf-8")

            if self._tree is None:
                self._tree = self._parser.parse(code_bytes)
            else:
                old_len = len(self._last_code)
                new_len = len(code_bytes)
                self._tree.edit(
                    start_byte=old_len,
                    old_end_byte=old_len,
                    new_end_byte=new_len,
                    start_point=(code.count('\n', 0, old_len), 0),
                    old_end_point=(code.count('\n', 0, old_len), 0),
                    new_end_point=(code.count('\n'), 0),
                )
                self._tree = self._parser.parse(code_bytes, self._tree)

            self._last_code = code_bytes
            all_signals.extend(self._extract(self._tree.root_node, code))

        # Dedup: only return new signals
        new_signals = []
        for s in all_signals:
            key = f"{s.capability}:{s.source_line}:{s.trigger}"
            if key not in self._seen_signals:
                self._seen_signals.add(key)
                new_signals.append(s)
        return new_signals

    # ── Tree-sitter extraction helpers ──────────────────────────

    def _extract(self, root: Node, code: str) -> list[ParsedSignal]:
        signals: list[ParsedSignal] = []
        self._walk(root, code, signals)
        
        # If no programmatic capabilities were found, but text exists, it's likely conversational/natural language.
        if len(signals) == 0 and len(code.strip()) > 5:
            signals.append(ParsedSignal(PrimitiveCap.NATURAL_LANGUAGE, 1, 1, "natural_language_text", 1.0))
            
        return signals

    def _walk(self, node: Node, code: str, out: list[ParsedSignal]):
        ntype = node.type.lower() if callable(getattr(node, 'type', None)) else node.type.lower()
        text = self._node_text(node, code)
        if len(text) > 0 and len(text) < 150:
            if "import" in ntype or "require" in ntype or "use" in ntype:
                matches = self.vector_mapper.match_primitive(text)
                for cap, score in matches:
                    out.append(ParsedSignal(cap, node.start_point[0] + 1, node.start_point[1], f"import:{text[:30]}", score))
            elif "call" in ntype or "invocation" in ntype:
                matches = self.vector_mapper.match_primitive(text)
                for cap, score in matches:
                    metadata = {}
                    if cap == PrimitiveCap.SHELL_INVOKE and "shell=True" in text:
                        metadata["high_risk"] = True
                        score = 1.0
                    out.append(ParsedSignal(cap, node.start_point[0] + 1, node.start_point[1], f"call:{text[:30]}", score, metadata))
        elif "assignment" in ntype:
            left = node.child_by_field_name("left")
            right = node.child_by_field_name("right")
            if left and right and left.type == "identifier" and right.type == "identifier":
                t_var = self._node_text(left, code)
                s_var = self._node_text(right, code)
                out.append(ParsedSignal(PrimitiveCap.MEMORY_COPY, node.start_point[0] + 1, node.start_point[1], "assignment", 0.5, {"target": t_var, "source": s_var}))

        for child in node.children:
            self._walk(child, code, out)

    @staticmethod
    def _ntype(node) -> str:
        return node.type() if callable(getattr(node, "type", None)) else node.type

    @staticmethod
    def _nrow(node) -> int:
        sp = node.start_point() if callable(getattr(node, "start_point", None)) else node.start_point
        return sp.row if hasattr(sp, "row") else sp[0]

    @staticmethod
    def _ncol(node) -> int:
        sp = node.start_point() if callable(getattr(node, "start_point", None)) else node.start_point
        return sp.column if hasattr(sp, "column") else sp[1]

    @staticmethod
    def _node_text(node, code: str) -> str:
        if node is None:
            return ""
        try:
            start = node.start_byte() if callable(getattr(node, "start_byte", None)) else node.start_byte
            end = node.end_byte() if callable(getattr(node, "end_byte", None)) else node.end_byte
            return code[start:end]
        except Exception:
            return ""


# ─────────────────────────────────────────────
# Language detection
# ─────────────────────────────────────────────

def detect_language(code: str) -> str:
    """Heuristically detect the programming language of a code snippet."""
    for lang, pattern in _LANG_HINTS:
        if re.search(pattern, code):
            return lang
    return "python"  # safe default


def signals_to_cap_nodes(signals: list[ParsedSignal], language: str) -> list[CapNode]:
    """Convert raw ParsedSignals to CapNodes for the graph."""
    nodes = []
    for s in signals:
        node = CapNode(
            node_type=NodeType.PRIMITIVE,
            label=s.capability.value,
            confidence=s.confidence,
            source_line=s.source_line,
            source_col=s.source_col,
            language=language,
            metadata={
                "trigger": s.trigger,
                "high_risk": s.capability in HIGH_RISK_PRIMITIVES,
                **s.metadata,
            },
        )
        nodes.append(node)
    return nodes
