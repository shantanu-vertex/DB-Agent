import os
from typing import Iterable

from openai import OpenAI

from .retriever import RetrievalResult

_GITHUB_MODELS_BASE_URL = "https://models.inference.ai.azure.com"


def _context_block(results: Iterable[RetrievalResult]) -> str:
    return "\n\n".join(
        f"[{item.chunk_id} | score={item.score:.3f}]\n{item.text}" for item in results
    )


def _fallback_answer(question: str, contexts: list[RetrievalResult]) -> str:
    if not contexts:
        return "I could not find relevant information in the uploaded JSON."

    snippets = "\n".join(f"- {item.text[:220]}" for item in contexts[:3])
    return (
        "No LLM key configured. Extractive response from retrieved chunks:\n\n"
        f"Question: {question}\n\n"
        f"Most relevant evidence:\n{snippets}"
    )


def _build_client() -> tuple[OpenAI | None, str]:
    """
    Returns (client, model_name).
    Priority:
      1. GITHUB_TOKEN  → GitHub Models (OpenAI-compatible, free with GitHub account)
      2. OPENAI_API_KEY → Standard OpenAI
      3. Neither set   → return (None, "") to trigger fallback
    """
    github_token = os.getenv("GITHUB_TOKEN", "").strip()
    if github_token:
        model = os.getenv("GITHUB_MODEL", "gpt-4o-mini")
        client = OpenAI(
            api_key=github_token,
            base_url=_GITHUB_MODELS_BASE_URL,
        )
        return client, model

    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if openai_key:
        model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        client = OpenAI(api_key=openai_key)
        return client, model

    return None, ""


def generate_answer(question: str, contexts: list[RetrievalResult]) -> str:
    client, model = _build_client()

    if client is None:
        return _fallback_answer(question, contexts)

    context_text = _context_block(contexts)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Answer the user only using the retrieved context. If the answer is not present, clearly say so.",
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

    return response.choices[0].message.content or _fallback_answer(question, contexts)
