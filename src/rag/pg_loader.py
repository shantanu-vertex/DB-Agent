"""Introspect a Postgres database into a structured JSON document.

The output is designed to give an LLM the *functional details* of each
table: columns, types, nullability, defaults, primary/foreign keys,
indexes and comments, plus DDL for views, functions and triggers.

The JSON is intentionally stable (sorted keys, deterministic ordering)
so its sha256 can be used for hash-based change detection.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

@dataclass
class PgConfig:
    host: str
    port: int
    dbname: str
    user: str
    password: str
    schema: str

    @classmethod
    def from_env(cls) -> "PgConfig":
        return cls(
            host=os.getenv("PG_HOST", "localhost"),
            port=int(os.getenv("PG_PORT", "5432")),
            dbname=os.getenv("PG_DB", "postgres"),
            user=os.getenv("PG_USER", "postgres"),
            password=os.getenv("PG_PASSWORD", ""),
            schema=os.getenv("PG_SCHEMA", "public"),
        )


def _connect(cfg: PgConfig) -> psycopg.Connection:
    return psycopg.connect(
        host=cfg.host,
        port=cfg.port,
        dbname=cfg.dbname,
        user=cfg.user,
        password=cfg.password,
        connect_timeout=5,
    )


# ---------------------------------------------------------------------------
# Introspection queries
# ---------------------------------------------------------------------------

_COLUMNS_SQL = """
SELECT
    c.table_name,
    c.column_name,
    c.ordinal_position,
    c.data_type,
    c.is_nullable,
    c.column_default,
    pgd.description AS column_comment
FROM information_schema.columns c
JOIN pg_catalog.pg_statio_all_tables st
     ON st.schemaname = c.table_schema AND st.relname = c.table_name
LEFT JOIN pg_catalog.pg_description pgd
     ON pgd.objoid = st.relid AND pgd.objsubid = c.ordinal_position
WHERE c.table_schema = %s
ORDER BY c.table_name, c.ordinal_position;
"""

_TABLE_COMMENT_SQL = """
SELECT t.table_name,
       obj_description(format('%%I.%%I', t.table_schema, t.table_name)::regclass, 'pg_class') AS table_comment,
       t.table_type
FROM information_schema.tables t
WHERE t.table_schema = %s
ORDER BY t.table_name;
"""

_PK_SQL = """
SELECT kcu.table_name, kcu.column_name
FROM information_schema.table_constraints tc
JOIN information_schema.key_column_usage kcu
     ON tc.constraint_name = kcu.constraint_name
    AND tc.table_schema = kcu.table_schema
WHERE tc.constraint_type = 'PRIMARY KEY' AND tc.table_schema = %s
ORDER BY kcu.table_name, kcu.ordinal_position;
"""

_FK_SQL = """
SELECT
    kcu.table_name,
    kcu.column_name,
    ccu.table_schema AS ref_schema,
    ccu.table_name   AS ref_table,
    ccu.column_name  AS ref_column,
    tc.constraint_name
FROM information_schema.table_constraints tc
JOIN information_schema.key_column_usage kcu
     ON tc.constraint_name = kcu.constraint_name
    AND tc.table_schema = kcu.table_schema
JOIN information_schema.constraint_column_usage ccu
     ON ccu.constraint_name = tc.constraint_name
    AND ccu.table_schema = tc.table_schema
WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_schema = %s
ORDER BY kcu.table_name, kcu.column_name;
"""

_INDEX_SQL = """
SELECT
    t.relname AS table_name,
    i.relname AS index_name,
    pg_get_indexdef(ix.indexrelid) AS index_def,
    ix.indisunique AS is_unique,
    ix.indisprimary AS is_primary
FROM pg_class t
JOIN pg_index ix ON t.oid = ix.indrelid
JOIN pg_class i  ON i.oid = ix.indexrelid
JOIN pg_namespace n ON n.oid = t.relnamespace
WHERE n.nspname = %s AND t.relkind IN ('r','p')
ORDER BY t.relname, i.relname;
"""

_VIEW_SQL = """
SELECT table_name AS view_name,
       pg_get_viewdef(format('%%I.%%I', table_schema, table_name)::regclass, true) AS definition
FROM information_schema.views
WHERE table_schema = %s
ORDER BY table_name;
"""

_FUNCTION_SQL = """
SELECT p.proname AS function_name,
       pg_get_functiondef(p.oid) AS definition,
       l.lanname AS language
FROM pg_proc p
JOIN pg_namespace n ON n.oid = p.pronamespace
JOIN pg_language l  ON l.oid = p.prolang
WHERE n.nspname = %s
  AND p.prokind IN ('f','p')
ORDER BY p.proname;
"""

_TRIGGER_SQL = """
SELECT event_object_table AS table_name,
       trigger_name,
       action_timing,
       event_manipulation,
       action_statement
FROM information_schema.triggers
WHERE trigger_schema = %s
ORDER BY event_object_table, trigger_name;
"""


# ---------------------------------------------------------------------------
# Build structured dict
# ---------------------------------------------------------------------------

def introspect(cfg: PgConfig | None = None) -> dict[str, Any]:
    cfg = cfg or PgConfig.from_env()

    with _connect(cfg) as conn, conn.cursor() as cur:
        cur.execute(_TABLE_COMMENT_SQL, (cfg.schema,))
        tables: dict[str, dict[str, Any]] = {}
        for name, comment, ttype in cur.fetchall():
            if ttype != "BASE TABLE":
                continue
            tables[name] = {
                "name": name,
                "schema": cfg.schema,
                "comment": comment,
                "columns": [],
                "primary_key": [],
                "foreign_keys": [],
                "indexes": [],
            }

        cur.execute(_COLUMNS_SQL, (cfg.schema,))
        for tname, cname, _pos, dtype, nullable, default, ccomment in cur.fetchall():
            if tname not in tables:
                continue
            tables[tname]["columns"].append({
                "name": cname,
                "type": dtype,
                "nullable": nullable == "YES",
                "default": default,
                "comment": ccomment,
            })

        cur.execute(_PK_SQL, (cfg.schema,))
        for tname, cname in cur.fetchall():
            if tname in tables:
                tables[tname]["primary_key"].append(cname)

        cur.execute(_FK_SQL, (cfg.schema,))
        for tname, cname, ref_schema, ref_table, ref_col, cons in cur.fetchall():
            if tname in tables:
                tables[tname]["foreign_keys"].append({
                    "column": cname,
                    "references": f"{ref_schema}.{ref_table}.{ref_col}",
                    "constraint": cons,
                })

        cur.execute(_INDEX_SQL, (cfg.schema,))
        for tname, iname, idef, is_unique, is_primary in cur.fetchall():
            if tname in tables:
                tables[tname]["indexes"].append({
                    "name": iname,
                    "definition": idef,
                    "unique": bool(is_unique),
                    "primary": bool(is_primary),
                })

        cur.execute(_VIEW_SQL, (cfg.schema,))
        views = [{"name": n, "definition": d.strip()} for n, d in cur.fetchall()]

        cur.execute(_FUNCTION_SQL, (cfg.schema,))
        functions = [
            {"name": n, "language": lang, "definition": d}
            for n, d, lang in cur.fetchall()
        ]

        cur.execute(_TRIGGER_SQL, (cfg.schema,))
        triggers = [
            {
                "table": t,
                "name": tn,
                "timing": timing,
                "event": event,
                "action": action,
            }
            for t, tn, timing, event, action in cur.fetchall()
        ]

    return {
        "database": cfg.dbname,
        "schema": cfg.schema,
        "tables": [tables[k] for k in sorted(tables)],
        "views": views,
        "functions": functions,
        "triggers": triggers,
    }


# ---------------------------------------------------------------------------
# Cache + hash
# ---------------------------------------------------------------------------

def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)


def schema_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def write_schema_cache(payload: dict[str, Any], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_canonical_json(payload), encoding="utf-8")
    return out


def read_schema_cache(path: str | Path) -> dict[str, Any] | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def ensure_schema_json(
    cache_path: str | Path | None = None,
    cfg: PgConfig | None = None,
    force: bool = False,
) -> tuple[Path, dict[str, Any], bool]:
    """Ensure a schema JSON exists at ``cache_path``.

    Behaviour (hash-based auto-detect):
      * If the file is missing or ``force`` is set -> introspect and write.
      * Otherwise compare the sha256 of a fresh introspection with the
        cached file's hash. Only rewrite when they differ.

    Returns ``(path, payload, changed)``.
    """
    cache_path = Path(cache_path or os.getenv("SCHEMA_CACHE_PATH", "cache/schema.json"))

    cached = None if force else read_schema_cache(cache_path)
    if cached is None:
        payload = introspect(cfg)
        write_schema_cache(payload, cache_path)
        return cache_path, payload, True

    fresh = introspect(cfg)
    if schema_hash(fresh) == schema_hash(cached):
        return cache_path, cached, False

    write_schema_cache(fresh, cache_path)
    return cache_path, fresh, True
