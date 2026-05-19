"""DB-Agent MCP server.

Exposes read-only tools that let an MCP client (e.g. VS Code's built-in
MCP client used by Copilot Chat agent mode) ask functional questions
about a local Postgres database.

Tools
-----
- ``refresh_index``: ensure the schema JSON cache reflects the live DB
  (hash-based auto-detect) and rebuild the retrieval index.
- ``ask``: run a natural language question against the indexed schema.
- ``list_tables``: cheap direct query for table names.
- ``get_object_ddl``: return the DDL for a table / view / function.

Run with ``python -m src.mcp_server`` (stdio transport).
"""
from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from src.rag import RAGPipeline
from src.rag.pg_loader import PgConfig, _connect  # noqa: F401  (internal reuse)

load_dotenv()

mcp = FastMCP("db-agent")
_pipeline = RAGPipeline()


def _ensure_indexed() -> None:
    if _pipeline.retriever is None:
        _pipeline.index_postgres()


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def refresh_index(force: bool = False) -> dict[str, Any]:
    """Refresh the schema snapshot from Postgres.

    Uses hash-based auto-detect: only re-chunks when the live schema
    differs from the cached JSON. Set ``force=True`` to always rebuild.
    """
    count, changed = _pipeline.index_postgres(force=force)
    return {
        "chunks": count,
        "changed": changed,
        "schema_path": str(_pipeline.schema_path) if _pipeline.schema_path else None,
        "schema_hash": _pipeline.schema_hash,
    }


@mcp.tool()
def ask(question: str, top_k: int = 5) -> dict[str, Any]:
    """Answer a question about the database using the indexed schema."""
    _ensure_indexed()
    answer, ctx = _pipeline.ask(question, top_k=top_k)
    return {
        "answer": answer,
        "chunks": [
            {"id": c.chunk_id, "score": round(c.score, 4), "text": c.text}
            for c in ctx
        ],
    }


@mcp.tool()
def list_tables(schema: str | None = None) -> list[str]:
    """List base table names in the configured (or supplied) schema."""
    cfg = PgConfig.from_env()
    if schema:
        cfg.schema = schema
    with _connect(cfg) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = %s AND table_type = 'BASE TABLE' "
            "ORDER BY table_name;",
            (cfg.schema,),
        )
        return [row[0] for row in cur.fetchall()]


@mcp.tool()
def get_object_ddl(name: str, kind: str = "table", schema: str | None = None) -> str:
    """Return the DDL definition for a table / view / function.

    ``kind`` must be one of ``table``, ``view``, ``function``.
    """
    kind = kind.lower().strip()
    if kind not in {"table", "view", "function"}:
        raise ValueError("kind must be one of: table, view, function")

    cfg = PgConfig.from_env()
    if schema:
        cfg.schema = schema

    with _connect(cfg) as conn, conn.cursor() as cur:
        if kind == "view":
            cur.execute(
                "SELECT pg_get_viewdef(format('%%I.%%I', %s, %s)::regclass, true);",
                (cfg.schema, name),
            )
            row = cur.fetchone()
            return row[0] if row else ""

        if kind == "function":
            cur.execute(
                """
                SELECT pg_get_functiondef(p.oid)
                FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace
                WHERE n.nspname = %s AND p.proname = %s
                LIMIT 1;
                """,
                (cfg.schema, name),
            )
            row = cur.fetchone()
            return row[0] if row else ""

        # table: synthesise a CREATE TABLE-ish description from columns + PK + FK
        cur.execute(
            """
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s
            ORDER BY ordinal_position;
            """,
            (cfg.schema, name),
        )
        cols = cur.fetchall()
        if not cols:
            return ""
        lines = [f"CREATE TABLE {cfg.schema}.{name} ("]
        col_lines = []
        for cname, dtype, nullable, default in cols:
            piece = f"  {cname} {dtype}"
            if nullable == "NO":
                piece += " NOT NULL"
            if default:
                piece += f" DEFAULT {default}"
            col_lines.append(piece)
        lines.append(",\n".join(col_lines))
        lines.append(");")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
