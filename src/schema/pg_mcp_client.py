"""
Thin wrapper that talks to the local PostgreSQL via psycopg2.
Used when running outside Copilot (e.g. Streamlit ingest tab).
"""
import os

import psycopg2
from dotenv import load_dotenv

load_dotenv()

_DEFAULT_CONN = "postgresql://postgres:postgres@localhost:5432/postgres"


def get_connection() -> psycopg2.extensions.connection:
    conn_str = os.getenv("POSTGRES_CONNECTION_STRING", _DEFAULT_CONN)
    return psycopg2.connect(conn_str)


def check_connection() -> str:
    """Returns the PostgreSQL server version string, or raises on failure."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT version();")
            return cur.fetchone()[0]
