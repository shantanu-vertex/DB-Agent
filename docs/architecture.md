# DB-Agent — Schema Intelligence Platform
## Problem → Vision → Plan

---

## Problem Summary

- Hundreds of tables across multiple configuration databases
- No semantic or business understanding of schema
- Impact analysis for features requires manual, error-prone effort
- Significant productivity loss for developers & architects

---

## Innovation Vision

Create an AI Agent that understands database schema the way a seasoned architect does:

- **Structural knowledge** — tables, columns, relationships
- **Business meaning & intent** — what each table actually represents in the domain
- **Feature-to-schema traceability** — which Jira features touch which tables

---

## Solution Phases

### Phase 1A — Schema JSON + Atlassian Enrichment
- ER extraction → canonical JSON schema model
- AI-generated semantic enrichment from Jira, Confluence (MCP already configured)

### Phase 1B — Live PostgreSQL (Fallback / Parallel)
- Direct connection to local running PostgreSQL on Windows
- Extract live table structure and relationships via `information_schema`
- Same enrichment pipeline applied to live-extracted schema

### Phase 2 — Semantic Upgrade (Future)
- Replace TF-IDF with OpenAI vector embeddings
- Multi-database federation
- Table-aware retrieval preserving FK context

---

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│                    Streamlit UI (app.py)                    │
│                                                            │
│   Tab 1: Schema Ingest  │  Tab 2: Ask  │  Tab 3: Trace    │
└──────┬─────────────────────────────────────────────────────┘
       │
┌──────▼──────────────────────────────────────────────────────┐
│              Schema Orchestrator (pipeline.py)               │
│   index_json()  ←─ existing                                  │
│   index_schema() ←─ new: schema-aware chunking               │
└──────┬──────────────────┬──────────────────────────────────┘
       │                  │
┌──────▼──────┐    ┌──────▼──────────────────────────────┐
│ Schema      │    │         Enrichment Layer             │
│ Source      │    │                                      │
│             │    │  • Jira MCP       (already live)     │
│ A: JSON     │    │  • Confluence MCP (already live)     │
│    upload   │    │  • OpenAI for description synthesis  │
│    (Phase   │    └──────────────────────────────────────┘
│     1A)     │
│             │
│ B: Postgres │  ──► psycopg2 / SQLAlchemy  (Phase 1B)
│    live DB  │
└─────────────┘
       │
┌──────▼──────────────────────────────────────────────────────┐
│           Enriched Schema JSON  (canonical format)           │
│                                                              │
│  {                                                           │
│    "table": "tenant_config",                                 │
│    "columns": [...],                                         │
│    "foreign_keys": [...],                                    │
│    "description": "AI-generated business description",       │
│    "jira_refs": ["PROJ-123", "PROJ-456"],                    │
│    "confluence_refs": ["Page: Tax Setup Guide"],             │
│    "feature_tags": ["tax-calculation", "tenant-onboarding"]  │
│  }                                                           │
└──────┬──────────────────────────────────────────────────────┘
       │
┌──────▼──────────────────────────────────────────────────────┐
│         Retrieval Layer                                       │
│   Phase 1: TF-IDF (existing retriever.py — unchanged)        │
│   Phase 2: OpenAI text-embedding-3-small (vector upgrade)    │
└─────────────────────────────────────────────────────────────┘
```

---

## Canonical Schema JSON Format

All schema sources (ER export or live PostgreSQL) are normalised into this shape before enrichment and indexing:

```json
{
  "database": "config_db",
  "extracted_at": "2026-05-08",
  "tables": [
    {
      "name": "tenant_config",
      "schema": "public",
      "columns": [
        { "name": "id",        "type": "uuid",    "nullable": false, "pk": true  },
        { "name": "tenant_id", "type": "varchar", "nullable": false, "pk": false }
      ],
      "foreign_keys": [
        { "column": "tenant_id", "ref_table": "tenants", "ref_column": "id" }
      ],
      "description":      "",
      "jira_refs":        [],
      "confluence_refs":  [],
      "feature_tags":     []
    }
  ]
}
```

Fields populated by enrichment (`description`, `jira_refs`, `confluence_refs`, `feature_tags`) are empty on ingest and filled automatically by `atlassian_enricher.py`.

---

## Implementation Steps

### Step 1 — Canonical Schema Loader  `src/schema/schema_loader.py`
- Accepts any ER tool export (DBeaver, pgAdmin, DataGrip) or hand-crafted JSON
- Validates and normalises into the canonical format above
- Raises clear errors for missing required fields

### Step 2 — Per-Table Schema Chunker  `src/schema/schema_chunker.py`
- Chunks **per table**, not by character count
- Each chunk carries metadata: table name, column list, FK relationships
- Enables queries like *"which tables relate to billing?"* to return structured results

### Step 3 — Atlassian Enricher  `src/enrichment/atlassian_enricher.py`
Uses existing Jira and Confluence MCP connections:

1. For each table name → `searchJiraIssuesUsingJql` with `text ~ "table_name"`
2. For each table name → `searchConfluenceUsingCql` with `text ~ "table_name"`
3. Sends Jira + Confluence results to OpenAI:
   > *"Given these tickets and docs, write a 2-sentence business description of this table."*
4. Writes `description`, `jira_refs`, `confluence_refs` back into the canonical JSON

### Step 4 — Update RAGPipeline  `src/rag/pipeline.py`
- Add `index_schema(schema_json_path)` alongside existing `index_json()`
- Schema-chunked documents passed to the existing `TfidfRetriever` unchanged
- Session state tracks whether current index is raw JSON or enriched schema

### Step 5 — Update Streamlit UI  `app.py`
Replace single-page layout with 3 tabs:

| Tab | Purpose |
|-----|---------|
| **Ingest** | Upload ER JSON or connect to PostgreSQL → run enrichment → show per-table progress |
| **Ask Schema** | Free-form Q&A against enriched schema (existing RAG flow) |
| **Feature Trace** | Input Jira epic key → find impacted tables; or input table → find related Jira features |

### Step 6 — Live PostgreSQL Extractor  `src/schema/pg_extractor.py`  *(Phase 1B)*
```python
# Queries information_schema to extract live structure
# SELECT table_name, column_name, data_type FROM information_schema.columns
# SELECT fk constraints from information_schema.referential_constraints
# Outputs the same canonical JSON — same enrichment pipeline applies
```
- `DB_CONNECTION_STRING` added to `.env` and `.env.example`
- UI "Connect to PostgreSQL" option replaces JSON upload for live data

### Step 7 — PostgreSQL MCP Server  *(optional, Phase 1B)*
- Install `mcp-server-postgres` (Node package) on Windows
- Add to VS Code MCP config
- Allows GitHub Copilot to perform live schema lookups during development

---

## Files to Create / Modify

| File | Action | Purpose |
|------|--------|---------|
| `src/schema/__init__.py` | Create | Module init |
| `src/schema/schema_loader.py` | Create | Normalize any ER JSON → canonical format |
| `src/schema/schema_chunker.py` | Create | Per-table chunking with metadata |
| `src/schema/pg_extractor.py` | Create | Live PostgreSQL schema extraction (Phase 1B) |
| `src/enrichment/__init__.py` | Create | Module init |
| `src/enrichment/atlassian_enricher.py` | Create | Jira + Confluence enrichment per table |
| `src/rag/pipeline.py` | Modify | Add `index_schema()` method |
| `app.py` | Modify | Add 3 tabs: Ingest / Ask / Feature Trace |
| `.env.example` | Modify | Add `DB_CONNECTION_STRING` |
| `requirements.txt` | Modify | Add `psycopg2-binary`, `sqlalchemy` |

---

## Quick-Start Sequence (Recommended)

1. Export your ER diagram from DBeaver / pgAdmin / DataGrip as JSON
2. Implement `schema_loader.py` to normalise it
3. Implement `atlassian_enricher.py` to auto-populate `description` + `jira_refs` per table
4. Update `pipeline.py` with `index_schema()`
5. Update `app.py` with the 3-tab layout
6. Query examples:
   - *"What tables are impacted by the Tax Calculation feature?"*
   - *"Describe the relationship between tenant_config and billing_rules."*
   - *"Which Jira tickets reference the tax_rate table?"*

---

## Dependencies to Add

```
psycopg2-binary     # PostgreSQL connection (Phase 1B)
sqlalchemy          # ORM / connection abstraction (Phase 1B)
```

Existing dependencies (`openai`, `scikit-learn`, `streamlit`) are sufficient for Phase 1A.

---

## Environment Variables

```env
# Existing
OPENAI_API_KEY=your-key-here
OPENAI_MODEL=gpt-4.1-mini

# Phase 1B — PostgreSQL
DB_CONNECTION_STRING=postgresql://user:password@localhost:5432/config_db
```
