import time
from pathlib import Path
from typing import Any, Callable

from .chunker import Chunk, chunk_text
from .json_loader import load_json_text
from .llm import generate_answer
from .oseries_loader import (
    OSeriesConfig,
    ensure_schema_json as ensure_oseries_schema_json,
    schema_hash as oseries_schema_hash,
)
from .pg_loader import PgConfig, ensure_schema_json, schema_hash
from .retriever import RetrievalResult, TfidfRetriever


StepCallback = Callable[[str, str, dict[str, Any]], None]


class RAGPipeline:
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self.retriever: TfidfRetriever | None = None
        self.source: str | None = None  # "json" | "postgres"
        self.schema_path: Path | None = None
        self.schema_hash: str | None = None
        self.last_trace: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Tracing helpers
    # ------------------------------------------------------------------
    def _emit(
        self,
        on_step: StepCallback | None,
        stage: str,
        status: str,
        info: dict[str, Any] | None = None,
    ) -> None:
        entry = {
            "stage": stage,
            "status": status,
            "timestamp": time.time(),
            "info": info or {},
        }
        self.last_trace.append(entry)
        if on_step is not None:
            try:
                on_step(stage, status, entry["info"])
            except Exception:
                pass

    # ------------------------------------------------------------------
    # JSON file source
    # ------------------------------------------------------------------
    def index_json(
        self,
        json_path: str | Path,
        chunk_size: int = 900,
        overlap: int = 150,
        on_step: StepCallback | None = None,
    ) -> int:
        self.last_trace = []
        self._emit(on_step, "load_json", "start", {"path": str(json_path)})
        t0 = time.perf_counter()
        text = load_json_text(json_path)
        self._emit(
            on_step,
            "load_json",
            "done",
            {"chars": len(text), "elapsed_ms": (time.perf_counter() - t0) * 1000},
        )

        self._emit(on_step, "chunk", "start", {"chunk_size": chunk_size, "overlap": overlap})
        t0 = time.perf_counter()
        self.chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        self._emit(
            on_step,
            "chunk",
            "done",
            {"chunks": len(self.chunks), "elapsed_ms": (time.perf_counter() - t0) * 1000},
        )

        self._emit(on_step, "index", "start", {})
        t0 = time.perf_counter()
        self.retriever = TfidfRetriever(self.chunks) if self.chunks else None
        vocab_size = (
            len(self.retriever.vectorizer.vocabulary_) if self.retriever else 0
        )
        self._emit(
            on_step,
            "index",
            "done",
            {"vocab": vocab_size, "elapsed_ms": (time.perf_counter() - t0) * 1000},
        )

        self.source = "json"
        self.schema_path = Path(json_path)
        self.schema_hash = None
        return len(self.chunks)

    # ------------------------------------------------------------------
    # Postgres source - hash-based auto-detect
    # ------------------------------------------------------------------
    def index_postgres(
        self,
        cache_path: str | Path | None = None,
        cfg: PgConfig | None = None,
        chunk_size: int = 900,
        overlap: int = 150,
        force: bool = False,
        on_step: StepCallback | None = None,
    ) -> tuple[int, bool]:
        """Ensure a JSON snapshot of the live Postgres schema exists, then
        index it. Returns ``(chunk_count, changed)``.
        """
        self.last_trace = []
        self._emit(on_step, "introspect", "start", {"force": force})
        t0 = time.perf_counter()
        path, payload, changed = ensure_schema_json(cache_path=cache_path, cfg=cfg, force=force)
        self._emit(
            on_step,
            "introspect",
            "done",
            {
                "path": str(path),
                "changed": changed,
                "tables": len(payload.get("tables", [])),
                "views": len(payload.get("views", [])),
                "functions": len(payload.get("functions", [])),
                "triggers": len(payload.get("triggers", [])),
                "elapsed_ms": (time.perf_counter() - t0) * 1000,
            },
        )

        if changed or self.retriever is None or self.source != "postgres":
            self._emit(on_step, "flatten", "start", {})
            t0 = time.perf_counter()
            text = load_json_text(path)
            self._emit(
                on_step,
                "flatten",
                "done",
                {"chars": len(text), "elapsed_ms": (time.perf_counter() - t0) * 1000},
            )

            self._emit(on_step, "chunk", "start", {"chunk_size": chunk_size, "overlap": overlap})
            t0 = time.perf_counter()
            self.chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
            self._emit(
                on_step,
                "chunk",
                "done",
                {"chunks": len(self.chunks), "elapsed_ms": (time.perf_counter() - t0) * 1000},
            )

            self._emit(on_step, "index", "start", {})
            t0 = time.perf_counter()
            self.retriever = TfidfRetriever(self.chunks) if self.chunks else None
            vocab_size = (
                len(self.retriever.vectorizer.vocabulary_) if self.retriever else 0
            )
            self._emit(
                on_step,
                "index",
                "done",
                {"vocab": vocab_size, "elapsed_ms": (time.perf_counter() - t0) * 1000},
            )

            self.source = "postgres"
            self.schema_path = path
            self.schema_hash = schema_hash(payload)
        else:
            self._emit(
                on_step,
                "reuse_cache",
                "done",
                {"chunks": len(self.chunks), "schema_hash": self.schema_hash},
            )

        return len(self.chunks), changed

    # ------------------------------------------------------------------
    # OSeries XML source - parse the oseries-database Maven module
    # ------------------------------------------------------------------
    def index_oseries(
        self,
        cache_path: str | Path | None = None,
        cfg: OSeriesConfig | None = None,
        chunk_size: int = 900,
        overlap: int = 150,
        force: bool = False,
        on_step: StepCallback | None = None,
    ) -> tuple[int, bool]:
        """Build the schema JSON from the local clone of vertexinc/oseries
        (its ``oseries-database`` DDL XML files) and index it.

        Returns ``(chunk_count, changed)``.
        """
        self.last_trace = []
        self._emit(on_step, "parse_xml", "start", {"force": force})
        t0 = time.perf_counter()
        path, payload, changed = ensure_oseries_schema_json(
            cache_path=cache_path, cfg=cfg, force=force
        )
        self._emit(
            on_step,
            "parse_xml",
            "done",
            {
                "path": str(path),
                "changed": changed,
                "modules": len(payload.get("sources", [])),
                "tables": len(payload.get("tables", [])),
                "elapsed_ms": (time.perf_counter() - t0) * 1000,
            },
        )

        if changed or self.retriever is None or self.source != "oseries":
            self._emit(on_step, "flatten", "start", {})
            t0 = time.perf_counter()
            text = load_json_text(path)
            self._emit(
                on_step,
                "flatten",
                "done",
                {"chars": len(text), "elapsed_ms": (time.perf_counter() - t0) * 1000},
            )

            self._emit(on_step, "chunk", "start", {"chunk_size": chunk_size, "overlap": overlap})
            t0 = time.perf_counter()
            self.chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
            self._emit(
                on_step,
                "chunk",
                "done",
                {"chunks": len(self.chunks), "elapsed_ms": (time.perf_counter() - t0) * 1000},
            )

            self._emit(on_step, "index", "start", {})
            t0 = time.perf_counter()
            self.retriever = TfidfRetriever(self.chunks) if self.chunks else None
            vocab_size = (
                len(self.retriever.vectorizer.vocabulary_) if self.retriever else 0
            )
            self._emit(
                on_step,
                "index",
                "done",
                {"vocab": vocab_size, "elapsed_ms": (time.perf_counter() - t0) * 1000},
            )

            self.source = "oseries"
            self.schema_path = path
            self.schema_hash = oseries_schema_hash(payload)
        else:
            self._emit(
                on_step,
                "reuse_cache",
                "done",
                {"chunks": len(self.chunks), "schema_hash": self.schema_hash},
            )

        return len(self.chunks), changed

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def ask(
        self,
        question: str,
        top_k: int = 5,
        on_step: StepCallback | None = None,
    ) -> tuple[str, list[RetrievalResult]]:
        if not self.retriever:
            raise RuntimeError(
                "No index loaded. Upload a JSON file or refresh from Postgres first."
            )

        self.last_trace = []
        self._emit(on_step, "tokenize", "start", {"query": question})
        t0 = time.perf_counter()
        query_vec = self.retriever.vectorizer.transform([question])
        nz = int(query_vec.nnz)
        self._emit(
            on_step,
            "tokenize",
            "done",
            {"non_zero_terms": nz, "elapsed_ms": (time.perf_counter() - t0) * 1000},
        )

        self._emit(on_step, "retrieve", "start", {"top_k": top_k})
        t0 = time.perf_counter()
        results = self.retriever.search(question, top_k=top_k)
        self._emit(
            on_step,
            "retrieve",
            "done",
            {
                "hits": len(results),
                "top_score": results[0].score if results else 0.0,
                "elapsed_ms": (time.perf_counter() - t0) * 1000,
            },
        )

        self._emit(on_step, "generate", "start", {"contexts": len(results)})
        t0 = time.perf_counter()
        answer = generate_answer(question, results)
        self._emit(
            on_step,
            "generate",
            "done",
            {"answer_chars": len(answer), "elapsed_ms": (time.perf_counter() - t0) * 1000},
        )
        return answer, results

    # ------------------------------------------------------------------
    # Inspection helpers (used by UI)
    # ------------------------------------------------------------------
    def top_terms_for_query(self, question: str, limit: int = 10) -> list[tuple[str, float]]:
        """Return the highest-weight TF-IDF terms in the user's query."""
        if not self.retriever:
            return []
        vec = self.retriever.vectorizer.transform([question])
        if vec.nnz == 0:
            return []
        feature_names = self.retriever.vectorizer.get_feature_names_out()
        coo = vec.tocoo()
        pairs = sorted(
            zip(coo.col.tolist(), coo.data.tolist()),
            key=lambda pair: pair[1],
            reverse=True,
        )[:limit]
        return [(feature_names[int(idx)], float(weight)) for idx, weight in pairs]

    def all_scores_for_query(self, question: str) -> list[float]:
        """Return the per-chunk relevance scores for a query (for charts)."""
        if not self.retriever:
            return []
        vec = self.retriever.vectorizer.transform([question])
        scores = (self.retriever.matrix @ vec.T).toarray().ravel()
        return scores.tolist()
