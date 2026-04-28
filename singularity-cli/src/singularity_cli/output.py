from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any

import httpx
import typer
from rich.console import Console
from rich.json import JSON
from rich.panel import Panel

from singularity_client import errors as client_errors
from singularity_client.models.http_validation_error import HTTPValidationError

from .context import CliError

console = Console(stderr=True)


def plain(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, list):
        return [plain(item) for item in value]
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value


def unwrap(result: Any, *, operation: str) -> Any:
    if result is None:
        raise CliError("runtime", f"{operation} returned no parsed payload")
    if isinstance(result, HTTPValidationError):
        raise CliError("validation", f"{operation} failed validation", details=plain(result), status=422)
    return result


def success_envelope(*, operation: str, data: Any, trace_id: str, session_id: str) -> dict[str, Any]:
    return {"ok": True, "operation": operation, "trace_id": trace_id, "session_id": session_id, "data": plain(data)}


def error_envelope(*, operation: str, error: CliError, trace_id: str | None, session_id: str | None) -> dict[str, Any]:
    return {
        "ok": False,
        "operation": operation,
        "trace_id": trace_id,
        "session_id": session_id,
        "error": {"kind": error.kind, "message": error.message, "status": error.status, "details": error.details},
    }


def emit(payload: dict[str, Any], *, human: bool) -> None:
    if payload.get("ok"):
        if human:
            console.print(JSON.from_data(payload["data"]))
            return
        typer.echo(json.dumps(payload["data"], ensure_ascii=False))
        return
    if human:
        console.print(
            Panel.fit(
                payload["error"]["message"],
                title=f"{payload['operation']} [{payload['error']['kind']}]",
                border_style="red",
            )
        )
        details = payload["error"].get("details")
        if details is not None:
            console.print(JSON.from_data(details))
        return
    typer.echo(json.dumps(payload, ensure_ascii=False))


def handle_exception(
    operation: str,
    exc: Exception,
    *,
    human: bool,
    trace_id: str | None = None,
    session_id: str | None = None,
) -> int:
    if isinstance(exc, CliError):
        envelope = error_envelope(operation=operation, error=exc, trace_id=trace_id, session_id=session_id)
    elif isinstance(exc, client_errors.UnexpectedStatus):
        body = exc.content.decode(errors="replace")
        try:
            details: Any = json.loads(body)
        except json.JSONDecodeError:
            details = body
        envelope = error_envelope(
            operation=operation,
            error=CliError("http", f"Backend returned HTTP {exc.status_code}", details=details, status=exc.status_code),
            trace_id=trace_id,
            session_id=session_id,
        )
    elif isinstance(exc, httpx.TimeoutException):
        text = (str(exc).strip() or f"{type(exc).__name__}: request exceeded HTTP client timeout")
        envelope = error_envelope(
            operation=operation, error=CliError("timeout", text), trace_id=trace_id, session_id=session_id
        )
    else:
        text = str(exc) or f"{type(exc).__name__}"
        envelope = error_envelope(
            operation=operation, error=CliError("runtime", text), trace_id=trace_id, session_id=session_id
        )
    emit(envelope, human=human)
    return 1
