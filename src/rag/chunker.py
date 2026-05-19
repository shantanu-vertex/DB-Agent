import json
from dataclasses import dataclass
from typing import Any


@dataclass
class Chunk:
    id: str
    text: str


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be less than chunk_size")

    normalized = " ".join(text.split())
    if not normalized:
        return []

    chunks: list[Chunk] = []
    start = 0
    index = 0
    step = chunk_size - overlap

    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        segment = normalized[start:end].strip()
        if segment:
            chunks.append(Chunk(id=f"chunk-{index}", text=segment))
            index += 1
        start += step

    return chunks


def chunk_text_from_relationship_json(objects: list[dict[str, Any]]) -> list[Chunk]:
    """
    Converts a list of relationship objects into chunks.
    Each top-level object becomes exactly one chunk (no size-based splitting).
    """
    if not objects:
        return []

    chunks: list[Chunk] = []
    for index, item in enumerate(objects):
        chunks.append(
            Chunk(
                id=f"chunk-{index}",
                text=json.dumps(item, ensure_ascii=False),
            )
        )
    return chunks
