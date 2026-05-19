import os
from typing import Iterable

from openai import OpenAI

from .retriever import RetrievalResult

# GitHub Copilot / GitHub Models uses an OpenAI-compatible endpoint.
# Set GITHUB_TOKEN in your .env to authenticate.
# Optionally override GITHUB_COPILOT_MODEL (default: gpt-4o).
_GITHUB_MODELS_BASE_URL = "https://models.inference.ai.azure.com"

_SYSTEM_PROMPT = (
    "You are a database schema assistant. Answer the user's question using only "
    "the retrieved context provided below. If the answer is not present in the "
    "context, clearly say so — do not invent tables, columns, types, or "
    "relationships. Cite chunk IDs (e.g. `chunk-12`) when referencing evidence."
)


def _context_block(results: Iterable[RetrievalResult]) -> str:
    return "\n\n".join(
        f"[{item.chunk_id} | score={item.score:.3f}]\n{item.text}" for item in results
    )


def _fallback_answer(question: str, contexts: list[RetrievalResult]) -> str:
    if not contexts:
        return (
            "I could not find anything in the indexed schema that matches "
            f"**{question!r}**. Try rephrasing with a table or column name."
        )

    # Each chunk_id is "TableName#N" — pull the unique table names found.
    seen: list[str] = []
    for item in contexts:
        tbl = item.chunk_id.split("#", 1)[0]
        if tbl and tbl not in seen:
            seen.append(tbl)

    table_list = ", ".join(f"**{t}**" for t in seen[:5])
    lines = [
        "_Set `GITHUB_TOKEN` in `.env` to get a natural-language answer. "
        "Below is a grounded extractive summary from the indexed schema:_",
        "",
        f"**Question:** {question}",
        "",
        f"**Relevant tables:** {table_list}",
        "",
        "**Top evidence:**",
    ]
    for item in contexts[:3]:
        # Cleanly summarise the chunk: first ~200 chars without raw inner quotes.
        excerpt = " ".join(item.text.split())[:280]
        lines.append(f"- `{item.chunk_id}` _(score {item.score:.2f})_ — {excerpt}")
    return "\n".join(lines)


def generate_answer(question: str, contexts: list[RetrievalResult]) -> str:
    github_token = os.getenv("GITHUB_TOKEN", "").strip()
    model = os.getenv("GITHUB_COPILOT_MODEL", "gpt-4o")

    if not github_token:
        return _fallback_answer(question, contexts)

    client = OpenAI(
        base_url=_GITHUB_MODELS_BASE_URL,
        api_key=github_token,
    )
    context_text = _context_block(contexts)

    response = client.chat.completions.create(
        model=model,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
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

    text = response.choices[0].message.content if response.choices else ""
    return text or _fallback_answer(question, contexts)
