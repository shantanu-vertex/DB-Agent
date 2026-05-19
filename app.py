"""Interactive UI for the DB Agent.

Exposes the inner workings of the RAG pipeline so users can see:
  * How the live schema is introspected
  * How JSON is flattened and chunked
  * The TF-IDF vocabulary and index
  * The exact terms a question matched
  * Per-chunk scores and rankings
  * A step-by-step trace of every pipeline stage with timings
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.rag import RAGPipeline
from src.rag.agent import Agent, AgentConfig
from src.rag.oseries_loader import OSeriesConfig
from src.rag.pg_loader import PgConfig

load_dotenv()

# ---------------------------------------------------------------------------
# Page config + global CSS
# ---------------------------------------------------------------------------
st.set_page_config(page_title="DB Agent · Interactive", page_icon="🧠", layout="wide")

st.markdown(
    """
    <style>
      .stage-card {
        padding: 0.7rem 0.9rem;
        border-radius: 8px;
        border: 1px solid rgba(125,125,125,0.25);
        margin-bottom: 0.35rem;
        background: rgba(125,125,125,0.05);
      }
      .stage-pill {
        display: inline-block;
        padding: 1px 8px;
        border-radius: 10px;
        font-size: 0.75rem;
        margin-left: 6px;
      }
      .pill-done   { background:#1f6f3a; color:#fff; }
      .pill-start  { background:#a17400; color:#fff; }
      .pill-error  { background:#a02020; color:#fff; }
      .metric-row  { font-size:0.85rem; opacity:0.85; margin-top:4px; }
      .small-mono  { font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
                     font-size: 0.78rem; opacity: 0.85; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "pipeline" not in st.session_state:
    st.session_state.pipeline = RAGPipeline()
if "indexed" not in st.session_state:
    st.session_state.indexed = False
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0
if "source_label" not in st.session_state:
    st.session_state.source_label = ""
if "index_trace" not in st.session_state:
    st.session_state.index_trace = []
if "query_trace" not in st.session_state:
    st.session_state.query_trace = []
if "last_question" not in st.session_state:
    st.session_state.last_question = ""
if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""
if "last_contexts" not in st.session_state:
    st.session_state.last_contexts = []
if "last_all_scores" not in st.session_state:
    st.session_state.last_all_scores = []
if "last_query_terms" not in st.session_state:
    st.session_state.last_query_terms = []

# Portal-mode state (simplified UI in front of the developer workbench).
if "agent" not in st.session_state:
    st.session_state.agent = Agent(
        AgentConfig(oseries_root=Path(os.getenv("OSERIES_ROOT", "C:/dev/oseries")))
    )
    # Try to fast-load a previously built rich JSON so the user can ask
    # immediately without clicking Prepare first.
    if st.session_state.agent.try_load_from_cache("json"):
        st.session_state["_portal_auto_loaded"] = "json"
if "portal_prepared" not in st.session_state:
    # If we auto-loaded above, treat the portal as prepared.
    st.session_state.portal_prepared = st.session_state.agent.state.retriever is not None
if "portal_source" not in st.session_state:
    st.session_state.portal_source = st.session_state.agent.state.source
if "portal_answer" not in st.session_state:
    st.session_state.portal_answer = ""
if "portal_contexts" not in st.session_state:
    st.session_state.portal_contexts = []

pipeline: RAGPipeline = st.session_state.pipeline
agent: Agent = st.session_state.agent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
STAGE_LABELS = {
    "introspect": "Introspect Postgres",
    "parse_xml": "Parse oseries-database XML",
    "reuse_cache": "Reused cached schema",
    "flatten": "Flatten JSON → text",
    "load_json": "Load JSON",
    "chunk": "Chunk text",
    "index": "Build TF-IDF index",
    "tokenize": "Vectorize query",
    "retrieve": "Retrieve top-k chunks",
    "generate": "Generate answer",
}


def _render_trace(trace: list[dict[str, Any]]) -> None:
    if not trace:
        st.caption("No steps recorded yet.")
        return
    for entry in trace:
        stage = entry["stage"]
        status = entry["status"]
        info = entry["info"]
        pill_class = {
            "done": "pill-done",
            "start": "pill-start",
            "error": "pill-error",
        }.get(status, "pill-start")
        label = STAGE_LABELS.get(stage, stage)
        elapsed = info.get("elapsed_ms")
        elapsed_str = f" · {elapsed:.1f} ms" if elapsed is not None else ""
        metrics = " · ".join(
            f"{k}={v}"
            for k, v in info.items()
            if k != "elapsed_ms" and not isinstance(v, (list, dict))
        )
        st.markdown(
            f"""<div class="stage-card">
              <b>{label}</b>
              <span class="stage-pill {pill_class}">{status}{elapsed_str}</span>
              <div class="metric-row small-mono">{metrics}</div>
            </div>""",
            unsafe_allow_html=True,
        )


def _read_schema_payload() -> dict[str, Any] | None:
    path = pipeline.schema_path
    if path and Path(path).exists():
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _make_step_recorder(container, key: str):
    """Return an on_step callback that incrementally renders to a Streamlit container."""
    placeholders: dict[str, Any] = {}

    def callback(stage: str, status: str, info: dict[str, Any]) -> None:
        entry = {"stage": stage, "status": status, "info": info, "timestamp": time.time()}
        st.session_state[key].append(entry)
        ph = placeholders.get(stage)
        if ph is None:
            ph = container.empty()
            placeholders[stage] = ph
        label = STAGE_LABELS.get(stage, stage)
        elapsed = info.get("elapsed_ms")
        elapsed_str = f" · {elapsed:.1f} ms" if elapsed is not None else ""
        pill_class = "pill-done" if status == "done" else "pill-start"
        metrics = " · ".join(
            f"{k}={v}"
            for k, v in info.items()
            if k != "elapsed_ms" and not isinstance(v, (list, dict))
        )
        ph.markdown(
            f"""<div class="stage-card">
              <b>{label}</b>
              <span class="stage-pill {pill_class}">{status}{elapsed_str}</span>
              <div class="metric-row small-mono">{metrics}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    return callback


# ---------------------------------------------------------------------------
# Sidebar: view selector first so the header/settings below can gate on it
# ---------------------------------------------------------------------------
view = st.sidebar.radio(
    "View",
    ["🎯 DB Agent (portal)", "🔧 Developer workbench"],
    help=(
        "Portal: ask questions about a DB table and get answers — "
        "the agent handles everything behind the scenes. "
        "Workbench: inspect every pipeline stage (chunks, vocab, "
        "scores, trace) for debugging."
    ),
)
st.sidebar.divider()

_is_portal = view == "🎯 DB Agent (portal)"

# ---------------------------------------------------------------------------
# Workbench-only header + status strip (skip on portal — it has its own)
# ---------------------------------------------------------------------------
if not _is_portal:
    st.title("🧠 DB Agent — Interactive Workbench")
    st.caption(
        "Watch the agent introspect, chunk, index, retrieve and answer — every stage exposed."
    )

    status_cols = st.columns(4)
    status_cols[0].metric("Status", "Indexed" if st.session_state.indexed else "Idle")
    status_cols[1].metric("Chunks", st.session_state.chunk_count)
    status_cols[2].metric(
        "Source",
        pipeline.source.upper() if pipeline.source else "—",
    )
    status_cols[3].metric(
        "Schema hash",
        (pipeline.schema_hash or "—")[:10] + ("…" if pipeline.schema_hash else ""),
    )

# ---------------------------------------------------------------------------
# Sidebar: settings (workbench tuning) + how it works
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    # The "Data source" radio is workbench-only — the portal has its own.
    if not _is_portal:
        source = st.radio(
            "Data source",
            ["OSeries XML", "Postgres (live)", "JSON file"],
            help=(
                "OSeries mode parses the local vertexinc/oseries DDL XML — no DB needed. "
                "Postgres mode introspects a live DB. JSON mode uploads a pre-built schema."
            ),
        )
    else:
        source = None  # not used in portal mode
    chunk_size = st.number_input("Chunk size", min_value=200, max_value=3000, value=900, step=100)
    overlap = st.number_input("Overlap", min_value=0, max_value=1000, value=150, step=50)
    top_k = st.slider("Top-k retrieval", min_value=1, max_value=15, value=5)

    st.divider()
    with st.expander("ℹ️ How the agent works", expanded=False):
        st.markdown(
            """
            **Pipeline stages**
            1. **Introspect / Parse / Load** — pull schema from Postgres,
               parse the oseries-database DDL XML, or read uploaded JSON.
            2. **Flatten** — convert nested JSON to `path: value` lines.
            3. **Chunk** — slide a window of `chunk_size` chars with `overlap`.
            4. **Index** — fit a TF-IDF vectorizer over chunks.
            5. **Tokenize** — convert your question to a sparse vector.
            6. **Retrieve** — cosine-sim against all chunks, pick top-k.
            7. **Generate** — OpenAI grounds an answer in those chunks
               (or a deterministic fallback if no API key).
            """
        )

# ---------------------------------------------------------------------------
# PORTAL VIEW (default — simplified UI)
# ---------------------------------------------------------------------------
if view == "🎯 DB Agent (portal)":
    st.title("🎯 DB Agent")
    st.caption(
        "Ask a question about a database table. The agent picks the source, "
        "crawls the oseries code for context, indexes it, and answers — "
        "all in the background."
    )

    # Source picker (in an expander so it doesn't dominate the page)
    with st.expander("⚙️ Schema source & rebuild", expanded=not st.session_state.portal_prepared):
        portal_src_label = st.radio(
            "Where should the agent get the schema from?",
            ["JSON (oseries XML — no DB needed)", "MCP (live Postgres introspection)"],
            horizontal=True,
            help=(
                "JSON parses the local vertexinc/oseries DDL XML — no DB connection. "
                "MCP introspects a running Postgres database."
            ),
        )
        portal_source = "json" if portal_src_label.startswith("JSON") else "mcp"

        if (
            st.session_state.portal_prepared
            and st.session_state.portal_source
            and st.session_state.portal_source != portal_source
        ):
            st.info(
                f"Source switched to **{portal_source.upper()}** but the indexed "
                f"data is still **{st.session_state.portal_source.upper()}**. "
                "Click **Rebuild index** to switch."
            )

        cprep_a, cprep_b = st.columns(2)
        prepare_clicked = cprep_a.button(
            "🔄 Rebuild index",
            use_container_width=True,
            help="Re-extract schema and re-index. Pick this after switching source or to refresh.",
        )
        rebuild_clicked = cprep_b.button(
            "💥 Force full rebuild",
            use_container_width=True,
            help="Ignore every cache (schema + code-context) and rebuild from scratch.",
        )

        if st.session_state.portal_prepared:
            st.caption(
                f"Currently indexed: **{st.session_state.portal_source.upper()}** · "
                f"{agent.state.chunks_indexed} chunks"
                + (f" · hash `{agent.state.rich_hash[:10]}`" if agent.state.rich_hash else "")
            )

    def _run_prepare(source: str, force: bool) -> bool:
        """Run the full prepare pipeline, render a live status panel.
        Returns True on success."""
        if int(overlap) >= int(chunk_size):
            st.error(
                f"Sidebar setting invalid: **Overlap ({overlap}) must be less than "
                f"chunk size ({chunk_size})**."
            )
            return False
        cfg_pg = None
        if source == "mcp":
            cfg_pg = PgConfig(
                host=os.getenv("PG_HOST", "localhost"),
                port=int(os.getenv("PG_PORT", "5432")),
                dbname=os.getenv("PG_DB", "postgres"),
                user=os.getenv("PG_USER", "postgres"),
                password=os.getenv("PG_PASSWORD", ""),
                schema=os.getenv("PG_SCHEMA", "public"),
            )
            if not cfg_pg.password:
                st.warning("PG_PASSWORD is empty — MCP mode will likely fail to connect.")
        agent.cfg.pg = cfg_pg
        agent.cfg.chunk_size = int(chunk_size)
        agent.cfg.overlap = int(overlap)

        with st.status("Preparing the agent…", expanded=True) as status:
            def _cb(stage: str, st_status: str, info: dict[str, Any]) -> None:
                label = {
                    "extract": "📥 Extract schema",
                    "code_context": "📚 Crawl oseries code",
                    "rich_json": "📝 Build rich JSON",
                    "chunk": "✂️ Chunk text",
                    "index": "🧮 Build TF-IDF index",
                }.get(stage, stage)
                elapsed = info.get("elapsed_ms")
                tail = f" ({elapsed:.0f} ms)" if elapsed is not None else ""
                metrics = " · ".join(
                    f"{k}={v}"
                    for k, v in info.items()
                    if k != "elapsed_ms" and not isinstance(v, (list, dict))
                )
                if st_status == "done":
                    status.write(f"✅ {label}{tail} — {metrics}")
                elif st_status == "skipped":
                    status.write(f"⏭️ {label} (skipped) — {metrics}")
                else:
                    status.write(f"⏳ {label}…")

            try:
                result = agent.prepare(source=source, force=force, on_step=_cb)
                st.session_state.portal_prepared = True
                st.session_state.portal_source = source
                st.session_state.portal_answer = ""
                st.session_state.portal_contexts = []
                status.update(label="Agent ready", state="complete", expanded=False)
                st.success(
                    f"Indexed {result['tables']} tables → {result['chunks']} chunks "
                    f"(source: {source.upper()})"
                )
                return True
            except Exception as exc:
                st.session_state.portal_prepared = False
                status.update(label="Failed", state="error")
                st.error(f"Agent failed: {exc}")
                return False

    if prepare_clicked or rebuild_clicked:
        _run_prepare(portal_source, force=rebuild_clicked)

    # -----------------------------------------------------------------
    # The primary thing: ask a question and get an answer
    # -----------------------------------------------------------------
    st.markdown("### 💬 Ask the agent")
    question = st.text_input(
        "Your question",
        placeholder="e.g. What columns does JurTypeSetMember have and which are foreign keys?",
        label_visibility="collapsed",
        key="portal_question",
    )
    ask_clicked = st.button("Ask", type="primary", use_container_width=True)

    if ask_clicked:
        if not question.strip():
            st.warning("Type a question above first.")
        else:
            # Auto-prepare if the agent isn't ready yet (first-time use,
            # no cache existed at startup).
            if not st.session_state.portal_prepared:
                st.info("Index not built yet — preparing now (first time only)…")
                if not _run_prepare(portal_source, force=False):
                    st.stop()
            with st.spinner("Thinking…"):
                try:
                    ans, ctxs = agent.ask(question, top_k=int(top_k))
                    st.session_state.portal_answer = ans
                    st.session_state.portal_contexts = [
                        {"chunk_id": c.chunk_id, "score": c.score, "text": c.text}
                        for c in ctxs
                    ]
                except RuntimeError as exc:
                    st.session_state.portal_prepared = False
                    st.error(f"{exc}\n\nClick **🔄 Rebuild index** above and try again.")
                except Exception as exc:
                    st.error(f"Query failed: {exc}")

    if st.session_state.portal_answer:
        st.markdown("#### Answer")
        st.markdown(st.session_state.portal_answer)

        with st.expander(f"Evidence — {len(st.session_state.portal_contexts)} retrieved chunks"):
            for ctx in st.session_state.portal_contexts:
                st.markdown(f"**{ctx['chunk_id']}** · score `{ctx['score']:.3f}`")
                st.code(ctx["text"][:600], language="text")

    # -----------------------------------------------------------------
    # Table lookup (secondary feature)
    # -----------------------------------------------------------------
    st.divider()
    st.markdown("### 📖 Inspect a single table")
    st.caption(
        "Get the enriched JSON (description, supplement details from the "
        "oseries codebase, columns with foreign-key links) for one table."
    )
    col_lookup, col_btn = st.columns([3, 1])
    lookup_name = col_lookup.text_input(
        "Table name",
        placeholder="e.g. JurTypeSetMember",
        label_visibility="collapsed",
        key="portal_table_lookup",
    )
    if col_btn.button("Show", use_container_width=True):
        if not lookup_name.strip():
            st.warning("Enter a table name first.")
        elif not st.session_state.portal_prepared:
            st.warning("Build the index first (click **Ask** or open the **Schema source** panel).")
        else:
            entry = agent.get_rich_for_table(lookup_name.strip())
            if entry:
                st.json(entry)
            else:
                st.warning(f"No table named `{lookup_name}` found in the rich JSON.")

    # In portal mode, do not render the workbench below.
    st.stop()


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_index, tab_schema, tab_chunks, tab_ask, tab_trace = st.tabs(
    [
        "1️⃣ Index",
        "2️⃣ Schema explorer",
        "3️⃣ Chunks & vocab",
        "4️⃣ Ask",
        "5️⃣ Trace",
    ]
)

# ---------------------------------------------------------------------------
# TAB 1 — Index (load + watch the pipeline run)
# ---------------------------------------------------------------------------
with tab_index:
    st.subheader("Build the index")

    if source == "OSeries XML":
        default_root = os.getenv("OSERIES_ROOT", "C:/dev/oseries")
        os_root = st.text_input(
            "oseries repo root",
            value=default_root,
            help="Path to your local clone of vertexinc/oseries (or the oseries-database directory).",
        )
        mods_default = os.getenv("OSERIES_MODULES", "")
        mods_input = st.text_input(
            "Modules (comma-separated, blank = all)",
            value=mods_default,
            help="e.g. tps,util,rte — short names matching oseries-database-<name> directories.",
        )

        os_cache_path = Path(
            os.getenv("OSERIES_CACHE_PATH", "cache/oseries_schema.json")
        )
        if os_cache_path.exists():
            st.info(f"Schema cache: `{os_cache_path}`")
        else:
            st.warning(
                f"No schema JSON at `{os_cache_path}`. Click **Parse oseries-database** to build it."
            )

        col_a, col_b = st.columns(2)
        parse_clicked = col_a.button("Parse oseries-database", type="primary")
        force_os_clicked = col_b.button(
            "Force re-parse", help="Ignore the cache and re-read all XML files."
        )

        if parse_clicked or force_os_clicked:
            if overlap >= chunk_size:
                st.warning("Overlap must be smaller than chunk size.")
            else:
                modules = [m.strip() for m in mods_input.split(",") if m.strip()] or None
                cfg = OSeriesConfig(root=Path(os_root), modules=modules)
                st.markdown("##### Live pipeline trace")
                live = st.container()
                st.session_state.index_trace = []
                recorder = _make_step_recorder(live, "index_trace")
                try:
                    count, changed = pipeline.index_oseries(
                        cache_path=os_cache_path,
                        cfg=cfg,
                        chunk_size=int(chunk_size),
                        overlap=int(overlap),
                        force=force_os_clicked,
                        on_step=recorder,
                    )
                    st.session_state.indexed = count > 0
                    st.session_state.chunk_count = count
                    st.session_state.source_label = (
                        f"OSeries XML ({os_root}) → {os_cache_path}"
                    )
                    if changed:
                        st.success(f"Schema JSON rebuilt from XML. Indexed {count} chunks.")
                    else:
                        st.success(f"XML unchanged — reused cache. Indexed {count} chunks.")
                except Exception as exc:
                    st.session_state.indexed = False
                    st.error(f"Failed to parse oseries-database: {exc}")

    elif source == "Postgres (live)":
        with st.expander(
            "Connection settings",
            expanded=not Path(os.getenv("SCHEMA_CACHE_PATH", "cache/schema.json")).exists(),
        ):
            c1, c2 = st.columns([3, 1])
            pg_host = c1.text_input("Host", value=os.getenv("PG_HOST", "localhost"))
            pg_port = c2.text_input("Port", value=os.getenv("PG_PORT", "5432"))
            pg_db = st.text_input("Database", value=os.getenv("PG_DB", "postgres"))
            c3, c4 = st.columns(2)
            pg_user = c3.text_input("User", value=os.getenv("PG_USER", "postgres"))
            pg_pass = c4.text_input("Password", value=os.getenv("PG_PASSWORD", ""), type="password")
            pg_schema = st.text_input("Schema", value=os.getenv("PG_SCHEMA", "public"))

            if st.button("Test connection"):
                try:
                    import psycopg

                    with psycopg.connect(
                        host=pg_host,
                        port=int(pg_port),
                        dbname=pg_db,
                        user=pg_user,
                        password=pg_pass,
                        connect_timeout=5,
                    ) as conn:
                        with conn.cursor() as cur:
                            cur.execute("SELECT version();")
                            ver = cur.fetchone()[0]
                    st.success(f"Connected. {ver}")
                except Exception as exc:
                    st.error(f"Connection failed: {exc}")

        cache_path = Path(os.getenv("SCHEMA_CACHE_PATH", "cache/schema.json"))
        if cache_path.exists():
            st.info(f"Schema cache: `{cache_path}`")
        else:
            st.warning(
                f"No schema JSON at `{cache_path}`. Fill in connection settings, then click "
                "**Refresh from Postgres**."
            )

        col_a, col_b = st.columns(2)
        refresh_clicked = col_a.button("Refresh from Postgres", type="primary")
        force_clicked = col_b.button("Force rebuild", help="Ignore the cache and re-introspect.")

        if refresh_clicked or force_clicked:
            if overlap >= chunk_size:
                st.warning("Overlap must be smaller than chunk size.")
            else:
                cfg = PgConfig(
                    host=pg_host,
                    port=int(pg_port),
                    dbname=pg_db,
                    user=pg_user,
                    password=pg_pass,
                    schema=pg_schema,
                )
                st.markdown("##### Live pipeline trace")
                live = st.container()
                st.session_state.index_trace = []
                recorder = _make_step_recorder(live, "index_trace")
                try:
                    count, changed = pipeline.index_postgres(
                        cache_path=cache_path,
                        cfg=cfg,
                        chunk_size=int(chunk_size),
                        overlap=int(overlap),
                        force=force_clicked,
                        on_step=recorder,
                    )
                    st.session_state.indexed = count > 0
                    st.session_state.chunk_count = count
                    st.session_state.source_label = (
                        f"Postgres ({pg_db}.{pg_schema}) → {cache_path}"
                    )
                    if changed:
                        st.success(f"Schema JSON regenerated. Indexed {count} chunks.")
                    else:
                        st.success(f"Schema unchanged — reused cache. Indexed {count} chunks.")
                except Exception as exc:
                    st.session_state.indexed = False
                    st.error(f"Failed to refresh from Postgres: {exc}")

    else:  # JSON mode
        with st.form("upload_form"):
            uploaded = st.file_uploader("Upload a JSON file", type=["json"])
            index_clicked = st.form_submit_button("Index JSON")

        if index_clicked:
            if uploaded is None:
                st.warning("Upload a JSON file first.")
            elif overlap >= chunk_size:
                st.warning("Overlap must be smaller than chunk size.")
            else:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
                    tmp.write(uploaded.getvalue())
                    tmp_path = Path(tmp.name)
                st.markdown("##### Live pipeline trace")
                live = st.container()
                st.session_state.index_trace = []
                recorder = _make_step_recorder(live, "index_trace")
                try:
                    count = pipeline.index_json(
                        tmp_path,
                        int(chunk_size),
                        int(overlap),
                        on_step=recorder,
                    )
                    st.session_state.indexed = True
                    st.session_state.chunk_count = count
                    st.session_state.source_label = f"JSON → {uploaded.name}"
                    st.success(f"Indexed {count} chunks.")
                except Exception as exc:
                    st.session_state.indexed = False
                    st.error(f"Failed to index JSON: {exc}")
                finally:
                    tmp_path.unlink(missing_ok=True)

    if st.session_state.indexed:
        st.divider()
        st.caption(f"Indexed: **{st.session_state.chunk_count} chunks** · {st.session_state.source_label}")

# ---------------------------------------------------------------------------
# TAB 2 — Schema explorer (only meaningful for Postgres / structured JSON)
# ---------------------------------------------------------------------------
with tab_schema:
    st.subheader("Schema explorer")
    payload = _read_schema_payload()
    if not payload:
        st.info("Index a Postgres source first to view tables, columns and relationships.")
    else:
        tables = payload.get("tables", [])
        views = payload.get("views", [])
        functions = payload.get("functions", [])
        triggers = payload.get("triggers", [])

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Tables", len(tables))
        m2.metric("Views", len(views))
        m3.metric("Functions", len(functions))
        m4.metric("Triggers", len(triggers))

        if tables:
            table_names = [t["name"] for t in tables]
            picked = st.selectbox("Inspect table", table_names)
            chosen = next(t for t in tables if t["name"] == picked)

            cc1, cc2 = st.columns([3, 2])
            with cc1:
                st.markdown(f"#### `{chosen['schema']}.{chosen['name']}`")
                if chosen.get("comment"):
                    st.caption(chosen["comment"])
                if chosen.get("columns"):
                    df = pd.DataFrame(chosen["columns"])
                    st.dataframe(df, use_container_width=True, hide_index=True)

            with cc2:
                st.markdown("**Primary key**")
                pk = chosen.get("primary_key") or []
                st.write(", ".join(pk) if pk else "—")

                st.markdown("**Foreign keys**")
                fks = chosen.get("foreign_keys") or []
                if fks:
                    st.dataframe(pd.DataFrame(fks), use_container_width=True, hide_index=True)
                else:
                    st.caption("None")

                st.markdown("**Indexes**")
                ixs = chosen.get("indexes") or []
                if ixs:
                    st.dataframe(pd.DataFrame(ixs), use_container_width=True, hide_index=True)
                else:
                    st.caption("None")

        # Relationship overview
        with st.expander("All foreign-key relationships"):
            edges = []
            for t in tables:
                for fk in t.get("foreign_keys") or []:
                    edges.append({
                        "from": f"{t['schema']}.{t['name']}.{fk['column']}",
                        "to": fk["references"],
                        "constraint": fk["constraint"],
                    })
            if edges:
                st.dataframe(pd.DataFrame(edges), use_container_width=True, hide_index=True)
            else:
                st.caption("No foreign keys detected.")

        with st.expander("Views"):
            if views:
                for v in views:
                    st.markdown(f"**{v['name']}**")
                    st.code(v["definition"], language="sql")
            else:
                st.caption("No views.")

        with st.expander("Functions"):
            if functions:
                for f in functions:
                    st.markdown(f"**{f['name']}** · `{f.get('language', '')}`")
                    st.code(f["definition"], language="sql")
            else:
                st.caption("No functions.")

# ---------------------------------------------------------------------------
# TAB 3 — Chunks & vocab
# ---------------------------------------------------------------------------
with tab_chunks:
    st.subheader("Chunk inspector")
    if not pipeline.chunks:
        st.info("Index a source first to see the produced chunks.")
    else:
        chunk_lengths = [len(c.text) for c in pipeline.chunks]
        c1, c2, c3 = st.columns(3)
        c1.metric("Total chunks", len(pipeline.chunks))
        c2.metric("Avg chunk length", f"{sum(chunk_lengths) // len(chunk_lengths)} chars")
        vocab = pipeline.retriever.vectorizer.vocabulary_ if pipeline.retriever else {}
        c3.metric("Vocabulary size", len(vocab))

        st.markdown("**Chunk length distribution**")
        st.bar_chart(pd.DataFrame({"length": chunk_lengths}))

        st.divider()
        st.markdown("**Browse chunks**")
        search = st.text_input("Filter chunks by substring", "")
        rows = []
        for c in pipeline.chunks:
            if not search or search.lower() in c.text.lower():
                rows.append({"id": c.id, "length": len(c.text), "preview": c.text[:200]})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=300)

        idx = st.selectbox(
            "Open full chunk",
            options=list(range(len(pipeline.chunks))),
            format_func=lambda i: pipeline.chunks[i].id,
        )
        st.text_area(
            "Full text",
            value=pipeline.chunks[idx].text,
            height=200,
        )

        if vocab:
            with st.expander("Top vocabulary terms (by document frequency)"):
                import numpy as np

                matrix = pipeline.retriever.matrix
                # df_count = number of docs each term appears in
                df_counts = (matrix > 0).sum(axis=0).A1
                feat = pipeline.retriever.vectorizer.get_feature_names_out()
                order = np.argsort(df_counts)[::-1][:40]
                df_view = pd.DataFrame(
                    {
                        "term": [feat[i] for i in order],
                        "doc_freq": [int(df_counts[i]) for i in order],
                    }
                )
                st.dataframe(df_view, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# TAB 4 — Ask (the live, observable query path)
# ---------------------------------------------------------------------------
with tab_ask:
    st.subheader("Ask a question")
    question = st.text_input(
        "Question",
        value=st.session_state.last_question,
        placeholder="e.g. What columns are in the users table and which are foreign keys?",
    )
    ask_clicked = st.button("Run query", type="primary", disabled=not st.session_state.indexed)

    if ask_clicked and question.strip():
        st.markdown("##### Live query trace")
        live = st.container()
        st.session_state.query_trace = []
        recorder = _make_step_recorder(live, "query_trace")
        try:
            answer, contexts = pipeline.ask(question, top_k=int(top_k), on_step=recorder)
            st.session_state.last_question = question
            st.session_state.last_answer = answer
            st.session_state.last_contexts = [
                {"chunk_id": c.chunk_id, "score": c.score, "text": c.text} for c in contexts
            ]
            st.session_state.last_all_scores = pipeline.all_scores_for_query(question)
            st.session_state.last_query_terms = pipeline.top_terms_for_query(question, limit=15)
        except Exception as exc:
            st.error(f"Query failed: {exc}")

    if st.session_state.last_answer:
        st.divider()
        st.markdown("### 💬 Answer")
        st.write(st.session_state.last_answer)

        col_terms, col_scores = st.columns(2)
        with col_terms:
            st.markdown("**Query terms matched (TF-IDF weight)**")
            if st.session_state.last_query_terms:
                df_terms = pd.DataFrame(
                    st.session_state.last_query_terms, columns=["term", "weight"]
                )
                st.dataframe(df_terms, use_container_width=True, hide_index=True)
            else:
                st.caption("No vocab terms matched — try rephrasing.")

        with col_scores:
            st.markdown("**Top-k chunk scores**")
            if st.session_state.last_contexts:
                df_scores = pd.DataFrame(
                    [{"chunk": c["chunk_id"], "score": c["score"]} for c in st.session_state.last_contexts]
                )
                st.bar_chart(df_scores.set_index("chunk"))
            else:
                st.caption("No chunks scored above zero.")

        if st.session_state.last_all_scores:
            with st.expander("Score distribution across ALL chunks"):
                st.line_chart(pd.DataFrame({"score": st.session_state.last_all_scores}))

        st.markdown("### 🔎 Retrieved chunks")
        for ctx in st.session_state.last_contexts:
            with st.expander(f"{ctx['chunk_id']} · score {ctx['score']:.3f}"):
                st.write(ctx["text"])

# ---------------------------------------------------------------------------
# TAB 5 — Trace (full log of last index + last query)
# ---------------------------------------------------------------------------
with tab_trace:
    st.subheader("Pipeline trace")
    st.caption(
        "Each stage of the agent records timings and metrics. "
        "Useful for debugging slow questions or unexpected behaviour."
    )

    left, right = st.columns(2)
    with left:
        st.markdown("##### Last index build")
        _render_trace(st.session_state.index_trace)
    with right:
        st.markdown("##### Last query")
        _render_trace(st.session_state.query_trace)

    with st.expander("Raw trace JSON"):
        st.json(
            {
                "index": st.session_state.index_trace,
                "query": st.session_state.query_trace,
            }
        )
