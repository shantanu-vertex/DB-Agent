import os
from typing import Iterable

from openai import OpenAI

from .retriever import RetrievalResult


def _context_block(results: Iterable[RetrievalResult]) -> str:
    return "\n\n".join(
        f"[{item.chunk_id} | score={item.score:.3f}]\n{item.text}" for item in results
    )


def _fallback_answer(question: str, contexts: list[RetrievalResult]) -> str:
    if not contexts:
        return "I could not find relevant information in the uploaded JSON."

    snippets = "\n".join(f"- {item.text[:220]}" for item in contexts[:3])
    return (
        "I could not use an LLM key, so here is a grounded extractive response based on retrieved JSON chunks.\n\n"
        f"Question: {question}\n\n"
        f"Most relevant evidence:\n{snippets}"
    )


def generate_answer(question: str, contexts: list[RetrievalResult]) -> str:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

    if not api_key:
        return _fallback_answer(question, contexts)

    client = OpenAI(api_key=api_key)
    context_text = _context_block(contexts)

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": "Answer the user only using the retrieved context. If missing, clearly say it is not present.",
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    f"Retrieved Context:\n{context_text}\n\n"
                    "Return a concise grounded answer."
                ),
            },
        ],
    )

    return response.output_text or _fallback_answer(question, contexts)
