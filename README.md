# DB Agent

A Streamlit + MCP assistant that answers **functional questions about a database**.

Two sources are supported from the same UI:

- **Postgres (live)** – introspects a local Postgres database, writes a structured `cache/schema.json`, then indexes it. Uses **hash-based auto-detect**: the cache is only rewritten when the live schema actually changes.
- **JSON file** – upload an existing schema/JSON document and index it directly.

The same retrieval pipeline (TF-IDF + grounded LLM answer) is also exposed as an **MCP server** for use inside VS Code (Copilot Chat agent mode).

## Two ways to run

### 1. Local UI (Streamlit)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env   # then fill in PG_* and OPENAI_API_KEY
streamlit run app.py
```

In the UI:
1. Pick **Postgres (live)** or **JSON file**.
2. Postgres mode: click **Refresh from Postgres**. If `cache/schema.json` does not exist it is generated; if it exists, the live schema hash is compared and the cache is reused when unchanged.
3. Ask a question.

### 2. VS Code MCP server

The repo ships [.vscode/mcp.json](.vscode/mcp.json) which registers a `db-agent` MCP server pointing at [src/mcp_server.py](src/mcp_server.py).

Tools exposed:

| Tool | Purpose |
| --- | --- |
| `refresh_index(force=False)` | Rebuild the schema JSON from Postgres (hash-based; `force=True` to skip the hash check). |
| `ask(question, top_k=5)` | Grounded answer over the indexed schema. |
| `list_tables(schema?)` | Plain list of base tables. |
| `get_object_ddl(name, kind, schema?)` | DDL for a `table`, `view`, or `function`. |

Open the workspace in VS Code, then in Copilot Chat **Agent** mode the `db-agent` tools become available.

## Configuration (`.env`)

```
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4.1-mini

PG_HOST=localhost
PG_PORT=5432
PG_DB=postgres
PG_USER=postgres
PG_PASSWORD=
PG_SCHEMA=public

SCHEMA_CACHE_PATH=cache/schema.json
```

Use a **read-only Postgres role**. The agent never executes free-form SQL.

## Project layout

```
app.py                  # Streamlit UI (JSON or Postgres source)
src/
  mcp_server.py         # FastMCP server (stdio) for VS Code
  rag/
    pg_loader.py        # Postgres introspection -> JSON + sha256 hash
    json_loader.py      # JSON -> flat key.path: value text
    chunker.py          # Overlapping char chunks
    retriever.py        # TF-IDF retriever
    llm.py              # OpenAI Responses call (extractive fallback)
    pipeline.py         # index_json / index_postgres / ask
.vscode/mcp.json        # VS Code MCP registration
```

