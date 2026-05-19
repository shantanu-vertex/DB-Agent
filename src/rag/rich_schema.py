"""Transform an introspected schema payload + code-derived context
into the rich JSON shape used by the DB Agent portal:

[
  {
    "table_name": "jurtypesetmember",
    "table_description": "...",
    "supplement_details": "...",
    "columns": [
      {
        "column_name": "jurtypesetid",
        "column_description": "",
        "foreign_table": "jurtypeset",
        "fk_description": "",
        "foreign_column": "jurtypesetid"
      },
      ...
    ]
  },
  ...
]
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .code_context import CodeContext


def _split_fk_reference(ref: str) -> tuple[str, str]:
    """``schema.table.column`` or ``table.column`` → (table, column)."""
    if not ref:
        return ("", "")
    parts = ref.split(".")
    if len(parts) >= 2:
        return (parts[-2], parts[-1])
    return (ref, "")


def _fk_index(table: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Map a table's columns to their FK target (if any).

    Returns ``{column_name_lower: {"foreign_table": str, "foreign_column": str}}``.
    """
    idx: dict[str, dict[str, str]] = {}
    for fk in table.get("foreign_keys", []) or []:
        local_cols = (fk.get("column") or "").split(",")
        foreign_table, foreign_col = _split_fk_reference(fk.get("references") or "")
        foreign_cols = foreign_col.split(",") if foreign_col else []
        for i, local in enumerate(local_cols):
            target_col = foreign_cols[i] if i < len(foreign_cols) else foreign_col
            idx[local.strip().lower()] = {
                "foreign_table": foreign_table,
                "foreign_column": target_col,
            }
    return idx


def build_rich_schema(
    payload: dict[str, Any],
    code_ctx: CodeContext | None = None,
) -> list[dict[str, Any]]:
    code_ctx = code_ctx or CodeContext()
    out: list[dict[str, Any]] = []

    for table in payload.get("tables", []):
        name = (table.get("name") or "").strip()
        if not name:
            continue
        key = name.lower()

        table_description = (
            table.get("comment")
            or code_ctx.table_descriptions.get(key)
            or ""
        )
        supplement_details = code_ctx.supplement_details.get(key, "")

        fk_idx = _fk_index(table)

        columns_out: list[dict[str, Any]] = []
        for col in table.get("columns", []) or []:
            cname = (col.get("name") or "").strip()
            if not cname:
                continue
            fk = fk_idx.get(cname.lower(), {})
            columns_out.append({
                "column_name": cname,
                "column_description": (
                    col.get("comment")
                    or code_ctx.column_descriptions.get((key, cname.lower()))
                    or ""
                ),
                "foreign_table": fk.get("foreign_table", ""),
                "fk_description": (
                    code_ctx.fk_descriptions.get((key, cname.lower()))
                    or ""
                ),
                "foreign_column": fk.get("foreign_column", ""),
                # Extra fields preserved for the LLM but optional in spec
                "data_type": col.get("type", ""),
                "nullable": col.get("nullable", True),
            })

        out.append({
            "table_name": name,
            "table_description": table_description,
            "supplement_details": supplement_details,
            "columns": columns_out,
        })

    return out


# ---------------------------------------------------------------------------
# Caching mirror (same shape as pg_loader / oseries_loader cache helpers)
# ---------------------------------------------------------------------------

def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)


def rich_schema_hash(rich: list[dict[str, Any]]) -> str:
    return hashlib.sha256(_canonical_json(rich).encode("utf-8")).hexdigest()


def write_rich_schema(rich: list[dict[str, Any]], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rich, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return out


def default_rich_cache_path(source: str) -> Path:
    base = os.getenv("RICH_CACHE_DIR", "cache")
    return Path(base) / f"rich_{source}.json"
