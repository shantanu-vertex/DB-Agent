import os
from typing import Iterable

from openai import BadRequestError, OpenAI

from .retriever import RetrievalResult

_GITHUB_MODELS_BASE_URL = "https://models.inference.ai.azure.com"

# Conservative defaults for a deployment capped around 8k input tokens.
_DEFAULT_MAX_INPUT_TOKENS = 8000
_DEFAULT_RESPONSE_TOKENS = 500
_DEFAULT_TOKEN_BUFFER = 700
_DEFAULT_PER_CHUNK_TOKENS = 450
_CHARS_PER_TOKEN_ESTIMATE = 3


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

def _safe_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _approx_tokens(text: str) -> int:
    return max(1, (len(text) + (_CHARS_PER_TOKEN_ESTIMATE - 1)) // _CHARS_PER_TOKEN_ESTIMATE)


def _build_budgeted_context(
        contexts: list[RetrievalResult],
        context_budget_tokens: int,
        per_chunk_token_cap: int,
) -> str:
    if context_budget_tokens <= 0:
        return ""

    chunks: list[str] = []
    used_tokens = 0
    capped_chars = max(1, per_chunk_token_cap) * _CHARS_PER_TOKEN_ESTIMATE

    for item in contexts:
        header = f"[{item.chunk_id} | score={item.score:.3f}]\n"
        body = item.text[:capped_chars]
        block = f"{header}{body}"
        block_tokens = _approx_tokens(block) + 2  # separator overhead

        if used_tokens + block_tokens > context_budget_tokens:
            break

        chunks.append(block)
        used_tokens += block_tokens

    return "\n\n".join(chunks)

def _is_token_limit_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "tokens_limit_reached" in message
        or "request body too large" in message
        or "max size" in message
        or "413" in message
    )


def generate_answer_budgeted(question: str, contexts: list[RetrievalResult]) -> str:
    """Budget-aware variant of generate_answer that limits prompt size."""
    client, model = _build_client()
    if client is None:
        return _fallback_answer(question, contexts)

    max_input_tokens = _safe_int_env("LLM_MAX_INPUT_TOKENS", _DEFAULT_MAX_INPUT_TOKENS)
    response_tokens = _safe_int_env("LLM_RESPONSE_TOKENS", _DEFAULT_RESPONSE_TOKENS)
    token_buffer = _safe_int_env("LLM_TOKEN_BUFFER", _DEFAULT_TOKEN_BUFFER)
    per_chunk_token_cap = _safe_int_env("LLM_PER_CHUNK_TOKENS", _DEFAULT_PER_CHUNK_TOKENS)

    system_text = (
        "Answer the user only using the retrieved context. "
        "If the answer is not present, clearly say so."
    )
    # Keep very long questions from consuming the entire budget.
    question_text = question[:4000]
    user_prefix = f"Question:\n{question_text}\n\nRetrieved Context:\n"
    user_suffix = "\n\nReturn a concise grounded answer."

    fixed_tokens = (
        _approx_tokens(system_text)
        + _approx_tokens(user_prefix)
        + _approx_tokens(user_suffix)
        + token_buffer
    )
    base_context_budget = max(0, max_input_tokens - response_tokens - fixed_tokens)

    # Retry once with a smaller context budget if provider still rejects prompt size.
    budget_attempts = [base_context_budget, max(0, base_context_budget // 2)]

    for context_budget_tokens in budget_attempts:
        context_text = _build_budgeted_context(
            contexts=contexts,
            context_budget_tokens=context_budget_tokens,
            per_chunk_token_cap=per_chunk_token_cap,
        )

        if not context_text:
            continue

        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=response_tokens,
                messages=[
                    {"role": "system", "content": system_text},
                    {
                        "role": "user",
                        "content": f"{user_prefix}{context_text}{user_suffix}",
                    },
                ],
            )
            return response.choices[0].message.content or _fallback_answer(question, contexts)
        except BadRequestError as exc:
            if _is_token_limit_error(exc):
                continue
            raise

    return "I could not fit enough retrieved context within the model token limit. Try reducing TOP_K."
