import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.rag import RAGPipeline

load_dotenv()

st.set_page_config(page_title="JSON RAG Assistant", page_icon="🧠", layout="centered")
st.title("JSON RAG Assistant")
st.caption("Upload JSON, chunk it, retrieve relevant chunks, and answer prompts.")

if "pipeline" not in st.session_state:
    st.session_state.pipeline = RAGPipeline()
if "indexed" not in st.session_state:
    st.session_state.indexed = False
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

with st.form("upload_form"):
    uploaded = st.file_uploader("Upload a JSON file", type=["json"])
    col1, col2 = st.columns(2)
    chunk_size = col1.number_input("Chunk size", min_value=200, max_value=3000, value=900, step=100)
    overlap = col2.number_input("Overlap", min_value=0, max_value=1000, value=150, step=50)
    index_clicked = st.form_submit_button("Index JSON")

if index_clicked:
    if uploaded is None:
        st.warning("Upload a JSON file first.")
    elif overlap >= chunk_size:
        st.warning("Overlap must be smaller than chunk size.")
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as temp:
            temp.write(uploaded.getvalue())
            temp_path = Path(temp.name)

        try:
            count = st.session_state.pipeline.index_json(temp_path, int(chunk_size), int(overlap))
            st.session_state.indexed = True
            st.session_state.chunk_count = count
            st.success(f"Indexed successfully. Created {count} chunks.")
        except Exception as exc:
            st.session_state.indexed = False
            st.error(f"Failed to index JSON: {exc}")
        finally:
            temp_path.unlink(missing_ok=True)

st.divider()

question = st.text_input("Enter your prompt")
ask_clicked = st.button("Get Answer", type="primary", disabled=not st.session_state.indexed)

if st.session_state.indexed:
    st.info(f"JSON is indexed with {st.session_state.chunk_count} chunks.")

if ask_clicked:
    if not question.strip():
        st.warning("Enter a question before asking.")
    else:
        with st.spinner("Running retrieval and generating answer..."):
            try:
                answer, contexts = st.session_state.pipeline.ask(question, top_k=5)
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
