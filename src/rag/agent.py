"""DB Agent orchestration.

Implements the workflow shown on the design whiteboard:

  User question
      │
      ▼
  Management Portal  ──source select──►  ┌──────────────────────┐
                                         │  1. DB Extractor     │  (MCP/Postgres or oseries XML)
                                         │  2. JSON generator   │
                                         │  3. Code-context     │  (crawl C:/dev/oseries)
                                         │  4. Rich JSON        │
                                         │  5. RAG + Vector DB  │  (TF-IDF retrieval)
                                         │  6. LLM answer       │
                                         └──────────────────────┘

The point of this module is to give the UI a single ``Agent`` entry
point that hides every step, so users only have to:
  * pick a source (``"mcp"`` or ``"json"``)
  * ask a question
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .chunker import Chunk, chunk_text
from .code_context import CodeContext, CodeContextConfig, build_code_context
from .llm import generate_answer
from .oseries_loader import OSeriesConfig
from .oseries_loader import ensure_schema_json as ensure_oseries_schema_json
from .pg_loader import PgConfig
from .pg_loader import ensure_schema_json as ensure_pg_schema_json
from .retriever import RetrievalResult, TfidfRetriever
from .rich_schema import (
    build_rich_schema,
    default_rich_cache_path,
    rich_schema_hash,
    write_rich_schema,
)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

Source = str  # "mcp" | "json"
StepCallback = Callable[[str, str, dict[str, Any]], None]


@dataclass
class AgentConfig:
    oseries_root: Path = Path("C:/dev/oseries")
    oseries_modules: list[str] | None = None  # respects OSERIES_MODULES if None
    pg: PgConfig | None = None  # only used when source="mcp"
    chunk_size: int = 900
    overlap: int = 150
    enable_code_context: bool = True

    def __post_init__(self) -> None:
        if self.oseries_modules is None:
            raw = os.getenv("OSERIES_MODULES", "").strip()
            self.oseries_modules = [m.strip() for m in raw.split(",") if m.strip()] or None


@dataclass
class AgentState:
    source: Source | None = None
    rich_path: Path | None = None
    rich_hash: str | None = None
    chunks_indexed: int = 0
    retriever: TfidfRetriever | None = None
    last_trace: list[dict[str, Any]] = field(default_factory=list)
    # In-memory cache of the parsed rich JSON so get_rich_for_table()
    # doesn't re-read the ~1 MB file on every lookup.
    rich_cache: list[dict[str, Any]] | None = None


class Agent:
    """End-to-end orchestrator. One instance per Streamlit session."""

    def __init__(self, cfg: AgentConfig | None = None) -> None:
        self.cfg = cfg or AgentConfig()
        self.state = AgentState()

    # ------------------------------------------------------------------
    # Tracing
    # ------------------------------------------------------------------
    def _emit(self, on_step: StepCallback | None, stage: str, status: str, info: dict[str, Any] | None = None) -> None:
        entry = {
            "stage": stage,
            "status": status,
            "timestamp": time.time(),
            "info": info or {},
        }
        self.state.last_trace.append(entry)
        if on_step is not None:
            try:
                on_step(stage, status, entry["info"])
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Step 1 — DB Extractor + JSON generator
    # ------------------------------------------------------------------
    def _extract_schema(self, source: Source, force: bool, on_step: StepCallback | None) -> tuple[Path, dict[str, Any]]:
        self._emit(on_step, "extract", "start", {"source": source})
        t0 = time.perf_counter()
        if source == "mcp":
            path, payload, _ = ensure_pg_schema_json(cfg=self.cfg.pg, force=force)
        elif source == "json":
            path, payload, _ = ensure_oseries_schema_json(
                cfg=OSeriesConfig(
                    root=self.cfg.oseries_root,
                    modules=self.cfg.oseries_modules,
                ),
                force=force,
            )
        else:
            raise ValueError(f"Unknown source: {source!r} (expected 'mcp' or 'json')")

        self._emit(on_step, "extract", "done", {
            "path": str(path),
            "tables": len(payload.get("tables", [])),
            "elapsed_ms": (time.perf_counter() - t0) * 1000,
        })
        return path, payload

    # ------------------------------------------------------------------
    # Step 2 — Code context (oseries crawl) with on-disk cache
    # ------------------------------------------------------------------
    def _code_ctx_cache_path(self, source: Source, table_names: list[str]) -> Path:
        # Key on the sorted table set so a schema-set change forces a fresh crawl.
        key = hashlib.sha256(
            "\n".join(sorted(table_names)).encode("utf-8")
        ).hexdigest()[:12]
        base = os.getenv("RICH_CACHE_DIR", "cache")
        return Path(base) / f"code_context_{source}_{key}.json"

    @staticmethod
    def _ctx_from_dict(d: dict[str, Any]) -> CodeContext:
        def _decode(map_in: dict[str, str]) -> dict[tuple[str, str], str]:
            out: dict[tuple[str, str], str] = {}
            for k, v in map_in.items():
                parts = k.split("|", 1)
                if len(parts) == 2:  # skip malformed keys defensively
                    out[(parts[0], parts[1])] = v
            return out

        return CodeContext(
            table_descriptions=d.get("table_descriptions", {}),
            supplement_details=d.get("supplement_details", {}),
            column_descriptions=_decode(d.get("column_descriptions", {})),
            fk_descriptions=_decode(d.get("fk_descriptions", {})),
        )

    @staticmethod
    def _ctx_to_dict(ctx: CodeContext) -> dict[str, Any]:
        return {
            "table_descriptions": ctx.table_descriptions,
            "supplement_details": ctx.supplement_details,
            "column_descriptions": {f"{t}|{c}": v for (t, c), v in ctx.column_descriptions.items()},
            "fk_descriptions": {f"{t}|{c}": v for (t, c), v in ctx.fk_descriptions.items()},
        }

    def _build_code_context(
        self,
        source: Source,
        payload: dict[str, Any],
        on_step: StepCallback | None,
        force: bool = False,
    ) -> CodeContext:
        if not self.cfg.enable_code_context:
            self._emit(on_step, "code_context", "skipped", {"reason": "disabled in config"})
            return CodeContext()

        table_names = [t["name"] for t in payload.get("tables", []) if t.get("name")]
        cache_path = self._code_ctx_cache_path(source, table_names)

        if not force and cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                ctx = self._ctx_from_dict(cached)
                self._emit(on_step, "code_context", "done", {
                    "tables_enriched": len(ctx.supplement_details),
                    "from_cache": True,
                    "cache_path": str(cache_path),
                    "elapsed_ms": 0.0,
                })
                return ctx
            except (OSError, json.JSONDecodeError):
                pass  # fall through to fresh crawl

        self._emit(on_step, "code_context", "start", {"tables": len(table_names)})
        t0 = time.perf_counter()
        ctx = build_code_context(
            table_names=table_names,
            cfg=CodeContextConfig(root=self.cfg.oseries_root),
        )
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(self._ctx_to_dict(ctx), ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            pass
        self._emit(on_step, "code_context", "done", {
            "tables_enriched": len(ctx.supplement_details),
            "from_cache": False,
            "elapsed_ms": (time.perf_counter() - t0) * 1000,
        })
        return ctx

    # ------------------------------------------------------------------
    # Step 3 — Rich JSON + index
    # ------------------------------------------------------------------
    def _build_index(
        self,
        source: Source,
        payload: dict[str, Any],
        code_ctx: CodeContext,
        on_step: StepCallback | None,
    ) -> tuple[Path, int]:
        self._emit(on_step, "rich_json", "start", {})
        t0 = time.perf_counter()
        rich = build_rich_schema(payload, code_ctx)
        rich_path = default_rich_cache_path(source)
        write_rich_schema(rich, rich_path)
        self.state.rich_hash = rich_schema_hash(rich)
        self.state.rich_path = rich_path
        self.state.rich_cache = rich  # avoid re-reading the file in get_rich_for_table
        self._emit(on_step, "rich_json", "done", {
            "path": str(rich_path),
            "tables": len(rich),
            "hash": self.state.rich_hash[:12],
            "elapsed_ms": (time.perf_counter() - t0) * 1000,
        })

        self._emit(on_step, "chunk", "start", {"chunk_size": self.cfg.chunk_size, "overlap": self.cfg.overlap})
        t0 = time.perf_counter()
        # Per-table chunking: each table becomes one or more chunks, each
        # prefixed with `TABLE {name}` so retrieval can anchor on the
        # table name. Avoids the bug where chunk_text() flattens whitespace
        # and glues adjacent tables together.
        chunks: list[Chunk] = []
        for entry in rich:
            tbl = entry["table_name"]
            cols_desc = "; ".join(
                f"{c['column_name']} ({c.get('data_type','')})"
                + (f" → {c['foreign_table']}.{c['foreign_column']}" if c.get("foreign_table") else "")
                for c in entry.get("columns", [])
            )
            body = (
                f"TABLE {tbl}\n"
                f"description: {entry.get('table_description','') or '—'}\n"
                f"supplement: {entry.get('supplement_details','') or '—'}\n"
                f"columns: {cols_desc or '—'}"
            )
            if len(body) <= self.cfg.chunk_size:
                chunks.append(Chunk(id=f"{tbl}#0", text=body))
            else:
                # Split big tables but keep the TABLE header on every chunk
                # so retrieval still matches the table name.
                sub_chunks = chunk_text(
                    body,
                    chunk_size=self.cfg.chunk_size,
                    overlap=self.cfg.overlap,
                )
                for i, sc in enumerate(sub_chunks):
                    text = sc.text if sc.text.startswith(f"TABLE {tbl}") else f"TABLE {tbl} :: {sc.text}"
                    chunks.append(Chunk(id=f"{tbl}#{i}", text=text))
        self._emit(on_step, "chunk", "done", {
            "chunks": len(chunks),
            "elapsed_ms": (time.perf_counter() - t0) * 1000,
        })

        self._emit(on_step, "index", "start", {})
        t0 = time.perf_counter()
        retriever = TfidfRetriever(chunks) if chunks else None
        self.state.retriever = retriever
        self.state.chunks_indexed = len(chunks)
        self.state.source = source
        vocab = len(retriever.vectorizer.vocabulary_) if retriever else 0
        self._emit(on_step, "index", "done", {
            "vocab": vocab,
            "elapsed_ms": (time.perf_counter() - t0) * 1000,
        })
        return rich_path, len(chunks)

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------
    def try_load_from_cache(self, source: Source) -> bool:
        """Fast path: rebuild the in-memory retriever from a previously
        written rich JSON cache, without re-extracting or re-crawling.

        Returns ``True`` if the cache existed and was loaded successfully.
        """
        rich_path = default_rich_cache_path(source)
        if not rich_path.exists():
            return False
        try:
            rich = json.loads(rich_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if not isinstance(rich, list) or not rich:
            return False

        # Re-chunk + re-index from the cached rich JSON (same as _build_index
        # but without re-extracting from source or re-crawling code).
        chunks: list[Chunk] = []
        for entry in rich:
            tbl = entry.get("table_name", "")
            if not tbl:
                continue
            cols_desc = "; ".join(
                f"{c['column_name']} ({c.get('data_type','')})"
                + (f" → {c['foreign_table']}.{c['foreign_column']}" if c.get("foreign_table") else "")
                for c in entry.get("columns", [])
            )
            body = (
                f"TABLE {tbl}\n"
                f"description: {entry.get('table_description','') or '—'}\n"
                f"supplement: {entry.get('supplement_details','') or '—'}\n"
                f"columns: {cols_desc or '—'}"
            )
            if len(body) <= self.cfg.chunk_size:
                chunks.append(Chunk(id=f"{tbl}#0", text=body))
            else:
                sub_chunks = chunk_text(
                    body,
                    chunk_size=self.cfg.chunk_size,
                    overlap=self.cfg.overlap,
                )
                for i, sc in enumerate(sub_chunks):
                    text = sc.text if sc.text.startswith(f"TABLE {tbl}") else f"TABLE {tbl} :: {sc.text}"
                    chunks.append(Chunk(id=f"{tbl}#{i}", text=text))

        if not chunks:
            return False
        self.state.retriever = TfidfRetriever(chunks)
        self.state.chunks_indexed = len(chunks)
        self.state.source = source
        self.state.rich_path = rich_path
        self.state.rich_hash = rich_schema_hash(rich)
        self.state.rich_cache = rich
        return True

    def prepare(
        self,
        source: Source,
        force: bool = False,
        on_step: StepCallback | None = None,
    ) -> dict[str, Any]:
        """Run the full extract → enrich → index pipeline."""
        self.state.last_trace = []
        _, payload = self._extract_schema(source, force=force, on_step=on_step)
        code_ctx = self._build_code_context(source, payload, on_step=on_step, force=force)
        rich_path, n_chunks = self._build_index(source, payload, code_ctx, on_step=on_step)
        return {
            "source": source,
            "rich_path": str(rich_path),
            "chunks": n_chunks,
            "tables": len(payload.get("tables", [])),
            "hash": self.state.rich_hash,
        }

    def ask(
        self,
        question: str,
        top_k: int = 5,
        on_step: StepCallback | None = None,
    ) -> tuple[str, list[RetrievalResult]]:
        if self.state.retriever is None:
            raise RuntimeError(
                "Agent is not prepared. Call agent.prepare(source='mcp' or 'json') first."
            )

        self._emit(on_step, "retrieve", "start", {"top_k": top_k})
        t0 = time.perf_counter()
        results = self.state.retriever.search(question, top_k=top_k)
        self._emit(on_step, "retrieve", "done", {
            "hits": len(results),
            "top_score": results[0].score if results else 0.0,
            "elapsed_ms": (time.perf_counter() - t0) * 1000,
        })

        self._emit(on_step, "generate", "start", {"contexts": len(results)})
        t0 = time.perf_counter()
        answer = generate_answer(question, results)
        self._emit(on_step, "generate", "done", {
            "answer_chars": len(answer),
            "elapsed_ms": (time.perf_counter() - t0) * 1000,
        })
        return answer, results

    def get_rich_for_table(self, table_name: str) -> dict[str, Any] | None:
        """Return the full rich-JSON entry for ``table_name`` (case-insensitive)."""
        rich = self.state.rich_cache
        if rich is None:
            if not self.state.rich_path or not Path(self.state.rich_path).exists():
                return None
            try:
                rich = json.loads(Path(self.state.rich_path).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None
            self.state.rich_cache = rich
        target = table_name.strip().lower()
        for entry in rich:
            if entry.get("table_name", "").lower() == target:
                return entry
        return None
