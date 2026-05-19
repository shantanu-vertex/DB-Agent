from pathlib import Path

from .chunker import Chunk, chunk_text, chunk_text_from_relationship_json
from .json_loader import load_json_text
from .llm import generate_answer
from .retriever import RetrievalResult, TfidfRetriever
from src.schema.relationship_extractor import extract_relationships


class RAGPipeline:
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self.retriever: TfidfRetriever | None = None

    def index_json(self, json_path: str | Path) -> int:
        # text = load_json_text(json_path)
        # self.chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        # self.retriever = TfidfRetriever(self.chunks) if self.chunks else None
        records = extract_relationships(json_path)
        self.chunks = chunk_text_from_relationship_json(records)
        self.retriever = TfidfRetriever(self.chunks) if self.chunks else None
        return len(self.chunks)

    def index_from_postgres(
        self,
        output_path: str | Path | None = None
    ) -> tuple[int, list[dict]]:
        """
        1. Extracts FK relationships from the local PostgreSQL database.
        2. Optionally saves the relationship JSON to *output_path*.
        3. Indexes each relationship object as one chunk.

        Returns (chunk_count, relationship_records).
        """
        
        records = extract_relationships(output_path)

        self.chunks = chunk_text_from_relationship_json(records)
        self.retriever = TfidfRetriever(self.chunks) if self.chunks else None
        count = len(self.chunks)

        return count, records

    def ask(self, question: str, top_k: int = 5) -> tuple[str, list[RetrievalResult]]:
        if not self.retriever:
            raise RuntimeError("No JSON index found. Upload and index a JSON file first.")

        results = self.retriever.search(question, top_k=top_k)
        answer = generate_answer(question, results)
        return answer, results
