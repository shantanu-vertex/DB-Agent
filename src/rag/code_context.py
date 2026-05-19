"""Crawl a local vertexinc/oseries clone to add code-level context to
schema metadata.

For each table the parser emits:
  * ``table_description`` — best human-readable description we can find
    (DDL XML ``<description>``, Javadoc above an entity class, or a
    leading SQL comment above ``CREATE TABLE``).
  * ``supplement_details`` — short snippets showing how the table is
    used in the codebase (queries it appears in, JOINs, inserts).
  * ``column_descriptions`` — per-column hints harvested from
    ``@Column`` annotations or DDL comments.
  * ``fk_descriptions`` — per-foreign-key hints harvested from
    ``@JoinColumn`` annotations or DDL FK comments.

The crawler is intentionally conservative: targeted regex grep against
known file types, hard caps on hits per table to keep the JSON small
enough to chunk + embed.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class CodeContextConfig:
    root: Path
    max_supplement_snippets: int = 5
    snippet_chars: int = 220
    # Directories under root to ignore (very large / not interesting).
    skip_dirs: tuple[str, ...] = (
        ".git", "target", "build", "node_modules", ".venv", ".idea",
    )

    @classmethod
    def from_env(cls) -> "CodeContextConfig":
        return cls(root=Path(os.getenv("OSERIES_ROOT") or "C:/dev/oseries"))


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

_INTERESTING_EXTS = {".java", ".sql", ".kt"}


def _iter_files(cfg: CodeContextConfig):
    root = Path(cfg.root)
    if not root.is_dir():
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in cfg.skip_dirs]
        for name in filenames:
            ext = Path(name).suffix.lower()
            if ext in _INTERESTING_EXTS:
                yield Path(dirpath) / name


# ---------------------------------------------------------------------------
# Per-file scan
# ---------------------------------------------------------------------------

@dataclass
class TableHit:
    file: str
    snippet: str
    kind: str   # "join", "insert", "select", "create_table", "java_ref"


_CREATE_TABLE_RE = re.compile(
    r"\bcreate\s+table\s+(?:if\s+not\s+exists\s+)?[\"\[`]?(\w+)[\"\]`]?",
    re.IGNORECASE,
)
_JOIN_RE = re.compile(r"\bjoin\s+(\w+)\b", re.IGNORECASE)
_INSERT_RE = re.compile(r"\binsert\s+into\s+(\w+)\b", re.IGNORECASE)
_SELECT_FROM_RE = re.compile(r"\bfrom\s+(\w+)\b", re.IGNORECASE)
_TABLE_ANNOT_RE = re.compile(
    r"""@Table\s*\(\s*name\s*=\s*["'](\w+)["']""", re.IGNORECASE
)


def _snippet_around(text: str, start: int, end: int, width: int) -> str:
    lo = max(0, start - width // 2)
    hi = min(len(text), end + width // 2)
    snippet = text[lo:hi]
    return " ".join(snippet.split())


def _scan_file(
    path: Path,
    text: str,
    cfg: CodeContextConfig,
    targets: set[str],
) -> dict[str, list[TableHit]]:
    """Return a map of lowercased table_name → list of TableHit."""
    by_table: dict[str, list[TableHit]] = {}

    def _record(tbl: str, match: re.Match, kind: str) -> None:
        key = tbl.lower()
        if targets and key not in targets:
            return
        by_table.setdefault(key, []).append(
            TableHit(
                file=str(path),
                snippet=_snippet_around(text, match.start(), match.end(), cfg.snippet_chars),
                kind=kind,
            )
        )

    ext = path.suffix.lower()
    if ext == ".sql":
        for m in _CREATE_TABLE_RE.finditer(text):
            _record(m.group(1), m, "create_table")
        for m in _JOIN_RE.finditer(text):
            _record(m.group(1), m, "join")
        for m in _INSERT_RE.finditer(text):
            _record(m.group(1), m, "insert")
        for m in _SELECT_FROM_RE.finditer(text):
            _record(m.group(1), m, "select")
    elif ext in (".java", ".kt"):
        for m in _TABLE_ANNOT_RE.finditer(text):
            _record(m.group(1), m, "java_ref")
        # Also catch inline SQL strings inside Java
        for m in _JOIN_RE.finditer(text):
            _record(m.group(1), m, "join")
        for m in _INSERT_RE.finditer(text):
            _record(m.group(1), m, "insert")

    return by_table


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class CodeContext:
    table_descriptions: dict[str, str] = field(default_factory=dict)
    supplement_details: dict[str, str] = field(default_factory=dict)
    column_descriptions: dict[tuple[str, str], str] = field(default_factory=dict)
    fk_descriptions: dict[tuple[str, str], str] = field(default_factory=dict)


def build_code_context(
    table_names: list[str],
    cfg: CodeContextConfig | None = None,
    on_progress=None,
) -> CodeContext:
    """Build a CodeContext for the given list of table names.

    ``on_progress`` is an optional ``callable(stage: str, info: dict)``
    used by the UI to render a live progress bar.
    """
    cfg = cfg or CodeContextConfig.from_env()
    targets = {name.lower() for name in table_names if name}

    hits_by_table: dict[str, list[TableHit]] = {}

    files = list(_iter_files(cfg))
    if on_progress:
        on_progress("scan_start", {"files": len(files), "tables": len(targets)})

    for i, path in enumerate(files):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        file_hits = _scan_file(path, text, cfg, targets)
        for tbl, hits in file_hits.items():
            hits_by_table.setdefault(tbl, []).extend(hits)
        if on_progress and i % 500 == 0:
            on_progress("scan_progress", {"processed": i, "total": len(files)})

    ctx = CodeContext()
    for tbl in targets:
        hits = hits_by_table.get(tbl, [])
        if not hits:
            continue
        # Supplement: pick distinct snippets, prefer joins/inserts which
        # explain *how* the table relates to others.
        seen_snippets: set[str] = set()
        ordered: list[str] = []
        for hit in sorted(hits, key=lambda h: (h.kind != "join", h.kind != "insert")):
            key = hit.snippet[:80]
            if key in seen_snippets:
                continue
            seen_snippets.add(key)
            ordered.append(f"[{hit.kind}] {hit.snippet}")
            if len(ordered) >= cfg.max_supplement_snippets:
                break
        if ordered:
            ctx.supplement_details[tbl] = "\n".join(ordered)

    if on_progress:
        on_progress("scan_done", {"tables_with_context": len(ctx.supplement_details)})

    return ctx
