import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.rag import RAGPipeline

load_dotenv()


def _llm_provider_label() -> tuple[str, str]:
    """Returns (provider_name, model_name) based on configured env vars."""
    token = os.getenv("GITHUB_TOKEN", "").strip()
    if token:
        return "GitHub Models", os.getenv("GITHUB_MODEL", "gpt-4o-mini")
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if key:
        return "OpenAI", os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    return "No LLM (extractive fallback)", ""

st.set_page_config(page_title="JSON RAG Assistant", page_icon="🧠", layout="centered")
st.title("JSON RAG Assistant")
st.caption("Upload JSON, chunk it, retrieve relevant chunks, and answer prompts.")

if "pipeline" not in st.session_state:
    st.session_state.pipeline = RAGPipeline()
if "indexed" not in st.session_state:
    st.session_state.indexed = False
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0
if "pg_status" not in st.session_state:
    st.session_state.pg_status = None  # None | "ok" | "error"
if "pg_version" not in st.session_state:
    st.session_state.pg_version = ""
if "pg_records" not in st.session_state:
    st.session_state.pg_records = []
if "pg_output_path" not in st.session_state:
    st.session_state.pg_output_path = None

# ── Ingest source selector ────────────────────────────────────────────────────
source = st.radio(
    "Ingest source",
    ["Upload JSON file", "Connect to local PostgreSQL"],
    horizontal=True,
)

# ── Branch: JSON upload ───────────────────────────────────────────────────────
if source == "Upload JSON file":
    with st.form("upload_form"):
        uploaded = st.file_uploader("Upload a JSON file", type=["json"])
        index_clicked = st.form_submit_button("Index JSON")

    if index_clicked:
        if uploaded is None:
            st.warning("Upload a JSON file first.")
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as temp:
                temp.write(uploaded.getvalue())
                temp_path = Path(temp.name)

            try:
                count = st.session_state.pipeline.index_json(
                    temp_path
                )
                st.session_state.indexed = True
                st.session_state.chunk_count = count
                st.success(f"Indexed successfully. Created {count} chunks.")
            except Exception as exc:
                st.session_state.indexed = False
                st.error(f"Failed to index JSON: {exc}")
            finally:
                temp_path.unlink(missing_ok=True)

# ── Branch: PostgreSQL ────────────────────────────────────────────────────────
else:
    st.markdown("#### PostgreSQL Connection")

    # Connection check
    check_clicked = st.button("Check Connection")
    if check_clicked:
        try:
            from src.schema.pg_mcp_client import check_connection

            version = check_connection()
            st.session_state.pg_status = "ok"
            st.session_state.pg_version = version
        except Exception as exc:
            st.session_state.pg_status = "error"
            st.session_state.pg_version = str(exc)

    if st.session_state.pg_status == "ok":
        st.success(f"● Connected — {st.session_state.pg_version.splitlines()[0]}")
    elif st.session_state.pg_status == "error":
        st.error(f"● Connection failed — {st.session_state.pg_version}")

    # Index from PostgreSQL
    pg_index_clicked = st.button(
        "Extract Relationships & Index",
        disabled=(st.session_state.pg_status != "ok"),
    )
    if pg_index_clicked:
        with st.spinner("Querying PostgreSQL FK relationships…"):
            try:
                output_path = Path(__file__).parent / "output" / "relationships.json"
                output_path.parent.mkdir(exist_ok=True)
                count, records = st.session_state.pipeline.index_from_postgres(
                    output_path=output_path
                )
                st.session_state.indexed = True
                st.session_state.chunk_count = count
                st.session_state.pg_records = records
                st.session_state.pg_output_path = output_path
                st.success(
                    f"Found {len(records)} FK relationships. "
                    f"Indexed into {count} chunks."
                )
            except Exception as exc:
                st.session_state.indexed = False
                st.error(f"Failed to extract relationships: {exc}")

    # Show preview + download after a successful extraction
    if st.session_state.pg_records and st.session_state.pg_output_path:
        st.subheader("Relationship JSON (preview — first 2 rows)")
        st.json(st.session_state.pg_records[:2])
        raw_json = st.session_state.pg_output_path.read_text(encoding="utf-8")
        st.download_button(
            label="⬇ Download relationships.json",
            data=raw_json,
            file_name="relationships.json",
            mime="application/json",
        )

# ── Q&A section ───────────────────────────────────────────────────────────────
st.divider()

# LLM provider status
_provider, _model = _llm_provider_label()
if _model:
    st.caption(f"LLM: **{_provider}** · model `{_model}`")
else:
    st.caption(f"⚠ {_provider} — set `GITHUB_TOKEN` or `OPENAI_API_KEY` in `.env` to enable LLM answers.")

question = st.text_input("Enter your prompt")
ask_clicked = st.button("Get Answer", type="primary", disabled=not st.session_state.indexed)

if st.session_state.indexed:
    st.info(f"Index active — {st.session_state.chunk_count} chunks.")

if ask_clicked:
    if not question.strip():
        st.warning("Enter a question before asking.")
    else:
        with st.spinner("Running retrieval and generating answer..."):
            try:
                default_top_k = os.getenv("TOP_K", str(5)).strip()
                effective_top_k = min(st.session_state.chunk_count, int(default_top_k))
                answer, contexts = st.session_state.pipeline.ask(question, top_k=effective_top_k)
                st.subheader("Answer")
                st.write(answer)

                st.subheader("Retrieved Chunks")
                if contexts:
                    for item in contexts:
                        st.markdown(f"**{item.chunk_id}** (score: {item.score:.3f})")
                        st.write(item.text)
                        st.divider()
                else:
                    st.write("No relevant chunk found for this query.")
            except Exception as exc:
                st.error(f"Failed to answer: {exc}")
