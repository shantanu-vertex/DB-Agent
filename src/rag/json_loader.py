import json
from pathlib import Path
from typing import Any


def _flatten_json(value: Any, prefix: str = "") -> list[str]:
    rows: list[str] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            new_prefix = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten_json(nested, new_prefix))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            new_prefix = f"{prefix}[{index}]"
            rows.extend(_flatten_json(nested, new_prefix))
    else:
        rows.append(f"{prefix}: {value}")
    return rows


def load_json_text(source: str | Path) -> str:
    path = Path(source)
    payload = json.loads(path.read_text(encoding="utf-8"))
    lines = _flatten_json(payload)
    return "\n".join(lines)
