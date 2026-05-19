# DB-Agent — PostgreSQL MCP Integration Architecture

## Overview

This document describes the **new flow layer** built on top of the existing DB-Agent RAG codebase.
The goal is to reuse a Python-based MCP PostgreSQL server to connect to a local PostgreSQL instance,
verify the connection, and produce a canonical **database relationship JSON** — feeding the same
enrichment and RAG pipeline already in place.

---

## New Flow at a Glance

```
Developer (VS Code + GitHub Copilot Agent)
        │
        │  MCP protocol (stdio)
        ▼
┌───────────────────────────────────┐
│  mcp-server-postgres (Python)     │  ← reused, not written from scratch
│  Connection: localhost:5432        │
│  Tool: query(), get_schema()       │
└──────────────┬────────────────────┘
               │  SQL queries via MCP tools
               ▼
┌───────────────────────────────────┐
│  Local PostgreSQL (Windows)        │
│  DB: oseries / dev database        │
│  Schema: information_schema        │
└──────────────┬────────────────────┘
               │
               │  Step 1 — Check connection
               │  Step 2 — Query FK relationships
               ▼
┌───────────────────────────────────────────────────────┐
│  Relationship JSON  (canonical output)                 │
│                                                        │
│  [                                                     │
│    {                                                   │
│      "table_name":        "jurtypesetmember",          │
│      "table_description": "",                          │
│      "supplement_details": "",                         │
│      "columns": [                                      │
│        {                                               │
│          "column_name":       "jurtypesetid",          │
│          "column_description": "",                     │
│          "foreign_table":     "jurtypeset",            │
│          "fk_description":    "",                      │
│          "foreign_column":    "jurtypesetid"           │
│        }                                               │
│      ]                                                 │
│    },                                                  │
│    ...                                                 │
│  ]                                                     │
└──────────────┬────────────────────────────────────────┘
               │
               ▼
┌───────────────────────────────────────────────────────┐
│  Existing DB-Agent RAG Pipeline  (unchanged)           │
│  pipeline.index_json(relationship_json_path)           │
│  → chunker → TfidfRetriever → llm → Streamlit UI      │
└───────────────────────────────────────────────────────┘
```

---

## Step 1 — Set Up the Python MCP PostgreSQL Server

### Install

```powershell
pip install mcp-server-postgres
# or via the project venv
.\.venv\Scripts\pip install mcp-server-postgres
```

### Register in VS Code MCP config

File: `%APPDATA%\Code\User\mcp.json`

```json
{
  "servers": {
    "postgres-local": {
      "type": "stdio",
      "command": "python",
      "args": ["-m", "mcp_server_postgres"],
      "env": {
        "POSTGRES_CONNECTION_STRING": "postgresql://postgres:postgres@localhost:5432/your_database"
      }
    }
  }
}
```

> **Security:** Never hard-code passwords. Use `${env:POSTGRES_PASSWORD}` or a `.env` file
> excluded from git.

---

## Step 2 — Check Connection

Once the MCP server is registered and VS Code is reloaded, the connection can be verified
via GitHub Copilot Agent or from Python code in the pipeline:

### Via Copilot Agent (interactive)
Ask in Copilot Chat (Agent mode):
> *"Using the postgres-local MCP, run `SELECT version();` and confirm the connection is active."*

### Via Python (`src/schema/pg_mcp_client.py`) — new file

```python
"""
Thin wrapper that talks to the local PostgreSQL via psycopg2.
Used when running outside Copilot (e.g. Streamlit ingest tab).
"""
import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

def get_connection():
    conn_str = os.getenv(
        "POSTGRES_CONNECTION_STRING",
        "postgresql://postgres:postgres@localhost:5432/postgres"
    )
    return psycopg2.connect(conn_str)

def check_connection() -> str:
    """Returns PostgreSQL server version string or raises on failure."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT version();")
            return cur.fetchone()[0]
```

Add to `.env.example`:
```
POSTGRES_CONNECTION_STRING=postgresql://postgres:postgres@localhost:5432/your_database
```

---

## Step 3 — Generate the Database Relationship JSON

### SQL Queries (via MCP tool or direct psycopg2)

Two queries are now used:

**Query 1 — all columns in every table:**

```sql
SELECT
    c.table_name,
    c.column_name,
    c.ordinal_position
FROM  information_schema.columns c
WHERE c.table_schema = 'public'
  AND c.table_name  IN (
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
          AND table_type   = 'BASE TABLE'
      )
ORDER BY c.table_name, c.ordinal_position;
```

**Query 2 — FK mappings only (to enrich matching columns):**

```sql
SELECT
    kcu.table_name,
    kcu.column_name,
    ccu.table_name  AS foreign_table,
    ccu.column_name AS foreign_column
FROM  information_schema.table_constraints        tc
JOIN  information_schema.key_column_usage         kcu
    ON tc.constraint_name = kcu.constraint_name
    AND tc.table_schema   = kcu.table_schema
JOIN  information_schema.constraint_column_usage  ccu
    ON ccu.constraint_name = tc.constraint_name
    AND ccu.table_schema   = tc.table_schema
WHERE tc.constraint_type = 'FOREIGN KEY'
  AND tc.table_schema    = 'public';
```

### Canonical Output Format

Every table in the schema is included. Columns with FK references carry
`foreign_table` / `foreign_column`; all other columns leave those fields blank.

```json
[
  {
    "table_name":         "jurtypesetmember",
    "table_description":  "",
    "supplement_details": "",
    "columns": [
      {
        "column_name":        "jurtypesetid",
        "column_description": "",
        "foreign_table":      "jurtypeset",
        "fk_description":     "",
        "foreign_column":     "jurtypesetid"
      },
      {
        "column_name":        "sortorder",
        "column_description": "",
        "foreign_table":      "",
        "fk_description":     "",
        "foreign_column":     ""
      }
    ]
  }
]
```

The `*_description` fields start empty and are populated in the enrichment step
(Phase 1A Jira + Confluence MCP enrichment already defined in `ARCHITECTURE.md`).

### New file: `src/schema/relationship_extractor.py`

```python
import json
from pathlib import Path
from .pg_mcp_client import get_connection

ALL_COLUMNS_QUERY = """
SELECT c.table_name, c.column_name, c.ordinal_position
FROM  information_schema.columns c
WHERE c.table_schema = 'public'
  AND c.table_name IN (
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
      )
ORDER BY c.table_name, c.ordinal_position;
"""

FK_QUERY = """
SELECT kcu.table_name, kcu.column_name,
       ccu.table_name AS foreign_table, ccu.column_name AS foreign_column
FROM  information_schema.table_constraints        tc
JOIN  information_schema.key_column_usage         kcu
    ON tc.constraint_name = kcu.constraint_name AND tc.table_schema = kcu.table_schema
JOIN  information_schema.constraint_column_usage  ccu
    ON ccu.constraint_name = tc.constraint_name AND ccu.table_schema = tc.table_schema
WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_schema = 'public';
"""

def extract_relationships(output_path: str | Path | None = None) -> list[dict]:
    """
    Returns every table with all its columns.
    FK columns carry foreign_table/foreign_column; others leave them blank.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(ALL_COLUMNS_QUERY)
            all_col_rows = cur.fetchall()
            cur.execute(FK_QUERY)
            fk_rows = cur.fetchall()

    fk_map = {(r[0], r[1]): (r[2], r[3]) for r in fk_rows}

    tables: dict[str, dict] = {}
    for table_name, column_name, _ in all_col_rows:
        if table_name not in tables:
            tables[table_name] = {
                "table_name": table_name,
                "table_description": "",
                "supplement_details": "",
                "columns": [],
            }
        fk = fk_map.get((table_name, column_name))
        tables[table_name]["columns"].append({
            "column_name":        column_name,
            "column_description": "",
            "foreign_table":      fk[0] if fk else "",
            "fk_description":     "",
            "foreign_column":     fk[1] if fk else "",
        })

    records = list(tables.values())

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(records, indent=2), encoding="utf-8")

    return records
```

---

## Step 4 — Wire Into the Existing RAG Pipeline

`src/rag/pipeline.py` already has `index_json(path)`.
Add one method that generates the relationship JSON first, then indexes it:

```python
# In RAGPipeline (src/rag/pipeline.py) — new method only
from src.schema.relationship_extractor import extract_relationships
import tempfile, json
from pathlib import Path

def index_from_postgres(
    self,
    output_path: str | None = None,
    chunk_size: int = 900,
    overlap: int = 150,
) -> tuple[int, list[dict]]:
    """
    1. Extracts FK relationships from local PostgreSQL.
    2. Saves to output_path (or a temp file).
    3. Indexes via the existing index_json() flow.
    Returns (chunk_count, relationship_records).
    """
    records = extract_relationships(output_path)

    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".json", mode="w", encoding="utf-8"
    ) as tmp:
        json.dump(records, tmp, indent=2)
        tmp_path = Path(tmp.name)

    count = self.index_json(tmp_path, chunk_size, overlap)
    tmp_path.unlink(missing_ok=True)
    return count, records
```

---

## Step 5 — Streamlit UI Addition (`app.py`)

Add a new **"Connect PostgreSQL"** option inside the existing upload form:

```
┌─────────────────────────────────────────────┐
│  Upload a JSON file  OR                      │
│  ┌──────────────────────────────────────┐   │
│  │  [Connect to local PostgreSQL]  btn  │   │
│  │  Status: ● Connected  v14.2          │   │
│  │  Found 42 FK relationships           │   │
│  └──────────────────────────────────────┘   │
│                                             │
│  Chunk size: [900]   Overlap: [150]         │
│  [ Index ]                                  │
└─────────────────────────────────────────────┘
```

Behaviour:
1. Button calls `check_connection()` → shows server version or error.
2. On "Index", calls `pipeline.index_from_postgres()`.
3. The resulting chunks feed the existing Ask tab unchanged.

---

## Files Changed / Created

| File | Action | Purpose |
|------|--------|---------|
| `src/schema/__init__.py` | **Create** | Package marker |
| `src/schema/pg_mcp_client.py` | **Create** | Connection helper + `check_connection()` |
| `src/schema/relationship_extractor.py` | **Create** | FK query → canonical relationship JSON |
| `src/rag/pipeline.py` | **Extend** | Add `index_from_postgres()` method |
| `app.py` | **Extend** | Add PostgreSQL connect button + status |
| `.env.example` | **Extend** | Add `POSTGRES_CONNECTION_STRING` |
| `requirements.txt` | **Extend** | Add `psycopg2-binary`, `mcp-server-postgres` |
| `%APPDATA%\Code\User\mcp.json` | **Configure** | Register `postgres-local` MCP server |

---

## Relationship JSON — Full Field Reference

**Table-level fields:**

| Field | Source | Description |
|-------|--------|-------------|
| `table_name` | `information_schema` | Child table that holds the FK |
| `table_description` | Enrichment (Phase 1A) | AI-generated purpose of the child table |
| `supplement_details` | Enrichment (Phase 1A) | Additional business context for the table |
| `columns` | `information_schema` | Array of FK columns belonging to this table |

**Column-level fields (inside `columns` array):**

| Field | Source | Description |
|-------|--------|-------------|
| `column_name` | `information_schema` | Column on the child table that is the FK |
| `column_description` | Enrichment (Phase 1A) | Meaning of the FK column in context |
| `foreign_table` | `information_schema` | Parent table being referenced |
| `fk_description` | Enrichment (Phase 1A) | Business meaning of the relationship |
| `foreign_column` | `information_schema` | Column on the parent table (usually PK) |

---

## Example Output

```json
[
  {
    "table_name":         "jurtypesetmember",
    "table_description":  "Junction table associating individual jurisdiction members with a type set",
    "supplement_details": "",
    "columns": [
      {
        "column_name":        "jurtypesetid",
        "column_description": "FK to jurtypeset; groups members under the same jurisdiction classification",
        "foreign_table":      "jurtypeset",
        "fk_description":     "Links a jurisdiction type set member to its parent set",
        "foreign_column":     "jurtypesetid"
      },
      {
        "column_name":        "jurtypeid",
        "column_description": "FK to jurtype; specifies the type of jurisdiction for this member entry",
        "foreign_table":      "jurtype",
        "fk_description":     "Identifies the specific jurisdiction type assigned to this member",
        "foreign_column":     "jurtypeid"
      }
    ]
  }
]
```

---

## Dependencies to Add to `requirements.txt`

```
psycopg2-binary>=2.9
mcp-server-postgres>=0.1
```
