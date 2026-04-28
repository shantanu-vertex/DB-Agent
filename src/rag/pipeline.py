from pathlib import Path

from .chunker import Chunk, chunk_text
from .json_loader import load_json_text
from .llm import generate_answer
from .retriever import RetrievalResult, TfidfRetriever


class RAGPipeline:
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self.retriever: TfidfRetriever | None = None

    def index_json(self, json_path: str | Path, chunk_size: int = 900, overlap: int = 150) -> int:
        text = load_json_text(json_path)
        self.chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        self.retriever = TfidfRetriever(self.chunks) if self.chunks else None
        return len(self.chunks)

    def ask(self, question: str, top_k: int = 5) -> tuple[str, list[RetrievalResult]]:
        if not self.retriever:
            raise RuntimeError("No JSON index found. Upload and index a JSON file first.")

        results = self.retriever.search(question, top_k=top_k)
        answer = generate_answer(question, results)
        return answer, results
