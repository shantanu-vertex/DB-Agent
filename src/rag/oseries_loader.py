"""Parse the oseries-database Maven module's DDL XML files into the same
dict shape that ``pg_loader.introspect()`` returns.

The vertexinc/oseries repository defines the canonical O Series schema as
XML under ``oseries-database/oseries-database-*/src/main/resources/schema/xml``.
Each ``<table>`` has ``<column>``, ``<foreign-key>``, ``<index>`` and
``<unique>`` children with rich attributes (domain type, size/scale,
constraint names, PK ordering).

This loader reads those files directly so the agent can answer schema
questions **without a live Postgres connection or credentials**.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class OSeriesConfig:
    """Where to find the oseries-database module on disk.

    ``root`` should point at either:
      * the repo root (``C:/dev/oseries``), or
      * the ``oseries-database`` directory inside it.

    ``modules`` optionally restricts which sub-modules to scan
    (e.g. ``["tps", "util"]``). Defaults to all discovered modules.
    """

    root: Path
    modules: list[str] | None = None

    @classmethod
    def from_env(cls) -> "OSeriesConfig":
        root = os.getenv("OSERIES_ROOT") or "C:/dev/oseries"
        mods_env = os.getenv("OSERIES_MODULES", "").strip()
        modules = [m.strip() for m in mods_env.split(",") if m.strip()] or None
        return cls(root=Path(root), modules=modules)


# ---------------------------------------------------------------------------
# XML helpers
# ---------------------------------------------------------------------------

_NS_PATTERN = re.compile(r"^\{[^}]+\}")


def _localname(tag: str) -> str:
    """Strip the XML namespace, if any."""
    return _NS_PATTERN.sub("", tag)


def _children(elem: ET.Element, name: str) -> list[ET.Element]:
    return [c for c in elem if _localname(c.tag) == name]


def _find_database_dir(root: Path | str) -> Path:
    """Locate the ``oseries-database`` directory.

    Accepts either the repo root or the module dir itself.
    """
    root = Path(root)
    if (root / "oseries-database").is_dir():
        return root / "oseries-database"
    if root.name == "oseries-database":
        return root
    # Fallback: best effort
    return root


def discover_ddl_files(cfg: OSeriesConfig) -> list[tuple[str, Path]]:
    """Return ``(module_short_name, ddl_xml_path)`` pairs.

    ``module_short_name`` is e.g. ``tps``, ``util``, ``rte`` — the part
    after ``oseries-database-``.
    """
    database_dir = _find_database_dir(cfg.root)
    if not database_dir.is_dir():
        raise FileNotFoundError(f"oseries-database not found under {cfg.root!s}")

    pairs: list[tuple[str, Path]] = []
    for child in sorted(database_dir.iterdir()):
        if not child.is_dir() or not child.name.startswith("oseries-database-"):
            continue
        short = child.name[len("oseries-database-"):]
        if cfg.modules and short not in cfg.modules:
            continue
        xml_dir = child / "src" / "main" / "resources" / "schema" / "xml"
        if not xml_dir.is_dir():
            continue
        for xml_file in sorted(xml_dir.glob("*.xml")):
            pairs.append((short, xml_file))
    return pairs


# ---------------------------------------------------------------------------
# Type formatting
# ---------------------------------------------------------------------------

def _format_type(col_elem: ET.Element) -> str:
    """Render the oseries domain type with size/scale, e.g. ``NAME(60)`` or
    ``SCALECOUNTER(18,6)``. Falls back to the raw type if no size.
    """
    base = col_elem.get("type") or ""
    size = col_elem.get("size")
    scale = col_elem.get("scale")
    if size and scale:
        return f"{base}({size},{scale})"
    if size:
        return f"{base}({size})"
    return base


# ---------------------------------------------------------------------------
# Table parsing
# ---------------------------------------------------------------------------

def _parse_table(elem: ET.Element, module: str) -> dict[str, Any]:
    name = elem.get("name") or ""
    pk_constraint = elem.get("primConstraintName")

    # Description (optional)
    desc_el = next(iter(_children(elem, "description")), None)
    description = (desc_el.text or "").strip() if desc_el is not None else None

    # Columns + primary key ordering
    columns: list[dict[str, Any]] = []
    pk_pairs: list[tuple[int, str]] = []
    for col in _children(elem, "column"):
        cname = col.get("name") or ""
        required = (col.get("required") or "").lower() == "true"
        is_pk = (col.get("primaryKey") or "").lower() == "true"
        order_raw = col.get("primaryKeyOrder")
        if is_pk:
            try:
                pk_pairs.append((int(order_raw or 0), cname))
            except ValueError:
                pk_pairs.append((0, cname))
        columns.append({
            "name": cname,
            "type": _format_type(col),
            "nullable": not required,
            "default": None,
            "comment": None,
        })
    primary_key = [name for _order, name in sorted(pk_pairs)]

    # Foreign keys
    foreign_keys: list[dict[str, Any]] = []
    for fk in _children(elem, "foreign-key"):
        constraint = fk.get("constraintName") or ""
        foreign_table = fk.get("foreignTable") or ""
        refs = _children(fk, "reference")
        # Multi-column FK: join with comma so the output stays one row per FK
        local_cols = ",".join((r.get("local") or "") for r in refs)
        foreign_cols = ",".join((r.get("foreign") or "") for r in refs)
        foreign_keys.append({
            "column": local_cols,
            "references": f"{foreign_table}.{foreign_cols}" if foreign_table else foreign_cols,
            "constraint": constraint,
        })

    # Indexes (non-unique <index>, plus <unique>)
    indexes: list[dict[str, Any]] = []
    for idx in _children(elem, "index"):
        iname = idx.get("name") or ""
        cols = [(c.get("name") or "") for c in _children(idx, "index-column")]
        definition = f"INDEX {iname} ON {name} ({', '.join(cols)})"
        indexes.append({
            "name": iname,
            "definition": definition,
            "unique": False,
            "primary": False,
        })
    for uq in _children(elem, "unique"):
        uname = uq.get("name") or ""
        cols = [(c.get("name") or "") for c in _children(uq, "unique-column")]
        definition = f"UNIQUE INDEX {uname} ON {name} ({', '.join(cols)})"
        indexes.append({
            "name": uname,
            "definition": definition,
            "unique": True,
            "primary": False,
        })
    if pk_constraint:
        indexes.append({
            "name": pk_constraint,
            "definition": f"PRIMARY KEY {pk_constraint} ON {name} ({', '.join(primary_key)})",
            "unique": True,
            "primary": True,
        })

    return {
        "name": name,
        "schema": module,
        "comment": description,
        "columns": columns,
        "primary_key": primary_key,
        "foreign_keys": foreign_keys,
        "indexes": indexes,
    }


def _parse_xml(path: Path, module: str) -> list[dict[str, Any]]:
    tree = ET.parse(path)
    root = tree.getroot()
    return [_parse_table(t, module) for t in _children(root, "table")]


# ---------------------------------------------------------------------------
# Public API — same dict shape as pg_loader.introspect()
# ---------------------------------------------------------------------------

def introspect(cfg: OSeriesConfig | None = None) -> dict[str, Any]:
    cfg = cfg or OSeriesConfig.from_env()
    pairs = discover_ddl_files(cfg)
    tables: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for module, xml_path in pairs:
        sources.append({"module": module, "file": str(xml_path)})
        tables.extend(_parse_xml(xml_path, module))

    tables.sort(key=lambda t: (t["schema"], t["name"]))

    return {
        "database": "oseries",
        "schema": ",".join(sorted({t["schema"] for t in tables})) or "(empty)",
        "sources": sources,
        "tables": tables,
        "views": [],
        "functions": [],
        "triggers": [],
    }


# ---------------------------------------------------------------------------
# Cache + hash (mirrors pg_loader)
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
    cfg: OSeriesConfig | None = None,
    force: bool = False,
) -> tuple[Path, dict[str, Any], bool]:
    """Same contract as ``pg_loader.ensure_schema_json``: returns
    ``(path, payload, changed)`` and only rewrites when the hash differs.
    """
    cache_path = Path(
        cache_path or os.getenv("OSERIES_CACHE_PATH", "cache/oseries_schema.json")
    )
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
