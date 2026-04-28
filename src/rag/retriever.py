from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .chunker import Chunk


@dataclass
class RetrievalResult:
    chunk_id: str
    text: str
    score: float


class TfidfRetriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform([item.text for item in chunks])

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        query_vec = self.vectorizer.transform([query])
        scores = (self.matrix @ query_vec.T).toarray().ravel()
        if scores.size == 0:
            return []

        top_indices = np.argsort(scores)[::-1][:top_k]
        results: list[RetrievalResult] = []
        for index in top_indices:
            score = float(scores[index])
            if score <= 0:
                continue
            chunk = self.chunks[int(index)]
            results.append(RetrievalResult(chunk_id=chunk.id, text=chunk.text, score=score))
        return results
