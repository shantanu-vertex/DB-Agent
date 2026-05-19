"""
Extracts all columns from every table in the public schema.
Columns that are FK references carry foreign-key metadata;
all other columns leave those fields blank.
"""
import json
from pathlib import Path

from .pg_mcp_client import get_connection

# All columns in every table in the public schema
ALL_COLUMNS_QUERY = """
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
"""

# FK mappings: (table_name, column_name) → (foreign_table, foreign_column)
FK_QUERY = """
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
"""


def extract_relationships(output_path: str | Path | None = None) -> list[dict]:
    """
    Returns every table in the public schema with all its columns.
    FK columns include foreign_table / foreign_column / fk_description;
    non-FK columns leave those three fields as empty strings.
    If *output_path* is given the result is also written there as JSON.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(ALL_COLUMNS_QUERY)
            all_col_rows = cur.fetchall()   # (table_name, column_name, ordinal)

            cur.execute(FK_QUERY)
            fk_rows = cur.fetchall()        # (table_name, column_name, foreign_table, foreign_column)

    # Build FK lookup: (table_name, column_name) → (foreign_table, foreign_column)
    fk_map: dict[tuple[str, str], tuple[str, str]] = {
        (r[0], r[1]): (r[2], r[3]) for r in fk_rows
    }

    # Group all columns by table
    tables: dict[str, dict] = {}
    for table_name, column_name, _ in all_col_rows:
        if table_name not in tables:
            tables[table_name] = {
                "table_name":         table_name,
                "table_description":  "",
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
        Path(output_path).write_text(
            json.dumps(records, indent=2), encoding="utf-8"
        )

    return records
