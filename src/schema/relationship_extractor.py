"""
Extracts foreign-key relationships from a local PostgreSQL database
and returns them as a list of canonical relationship dicts.
"""
import json
from pathlib import Path

from .pg_mcp_client import get_connection

RELATIONSHIP_QUERY = """
SELECT
    kcu.table_name,
    kcu.column_name,
    ccu.table_name  AS foreign_table,
    ccu.column_name AS foreign_column,
    ''              AS fk_description,
    ''              AS table_description,
    ''              AS column_description
FROM  information_schema.table_constraints        tc
JOIN  information_schema.key_column_usage         kcu
    ON tc.constraint_name = kcu.constraint_name
    AND tc.table_schema   = kcu.table_schema
JOIN  information_schema.constraint_column_usage  ccu
    ON ccu.constraint_name = tc.constraint_name
    AND ccu.table_schema   = tc.table_schema
WHERE tc.constraint_type = 'FOREIGN KEY'
AND   tc.table_schema    = 'public'
ORDER BY kcu.table_name, kcu.column_name;
"""

_ROW_COLUMNS = [
    "table_name",
    "column_name",
    "foreign_table",
    "foreign_column",
    "fk_description",
    "table_description",
    "column_description",
]


def extract_relationships(output_path: str | Path | None = None) -> list[dict]:
    """
    Queries local PostgreSQL for all FK relationships in the public schema.

    Returns a list of table dicts grouped by table in the canonical format:
    [
      {
        "table_name": ...,
        "table_description": ...,
        "supplement_details": "",
        "columns": [
          {
            "column_name": ...,
            "column_description": ...,
            "foreign_table": ...,
            "fk_description": ...,
            "foreign_column": ...
          },
          ...
        ]
      },
      ...
    ]
    If *output_path* is given, the result is also written there as JSON.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(RELATIONSHIP_QUERY)
            rows = cur.fetchall()

    # Group rows by table name
    tables: dict[str, dict] = {}
    for row in rows:
        rec = dict(zip(_ROW_COLUMNS, row))
        t_name = rec["table_name"]
        if t_name not in tables:
            tables[t_name] = {
                "table_name": t_name,
                "table_description": rec["table_description"],
                "supplement_details": "",
                "columns": [],
            }
        tables[t_name]["columns"].append({
            "column_name":        rec["column_name"],
            "column_description": rec["column_description"],
            "foreign_table":      rec["foreign_table"],
            "fk_description":     rec["fk_description"],
            "foreign_column":     rec["foreign_column"],
        })

    records = list(tables.values())

    if output_path:
        Path(output_path).write_text(
            json.dumps(records, indent=2), encoding="utf-8"
        )

    return records
