from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .context import CliError


def parse_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError as exc:
        raise CliError("validation", f"Invalid datetime: {value}") from exc


def parse_string_list(values: list[str] | None, file: Path | None, field: str) -> list[str]:
    if values:
        return values
    if file is None:
        raise CliError("validation", f"{field} requires at least one value")
    try:
        payload = json.loads(file.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CliError("validation", f"Input file not found: {file}") from exc
    except json.JSONDecodeError as exc:
        raise CliError("validation", f"Input file is not valid JSON: {file}", details={"line": exc.lineno}) from exc
    data = payload.get(field)
    if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
        raise CliError("validation", f"Input JSON must contain `{field}: string[]`")
    return data


def load_json_object(file: Path) -> dict[str, Any]:
    try:
        payload = json.loads(file.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CliError("validation", f"Input file not found: {file}") from exc
    except json.JSONDecodeError as exc:
        raise CliError("validation", f"Input file is not valid JSON: {file}", details={"line": exc.lineno}) from exc
    if not isinstance(payload, dict):
        raise CliError("validation", "Input file must contain a JSON object")
    return payload
