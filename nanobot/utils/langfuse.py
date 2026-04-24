"""Minimal optional Langfuse helpers for Nanobot tracing."""

from __future__ import annotations

import importlib
import os
from contextlib import contextmanager, nullcontext
from typing import Any, Callable

_LANGFUSE_READY = False
_get_client: Callable[[], Any] | None = None
_propagate_attributes: Callable[..., Any] | None = None

if os.environ.get("LANGFUSE_SECRET_KEY"):
    try:
        _langfuse = importlib.import_module("langfuse")
        _lf_get_client = getattr(_langfuse, "get_client", None)
        _lf_propagate_attributes = getattr(_langfuse, "propagate_attributes", None)
    except Exception:
        _lf_get_client = None
        _lf_propagate_attributes = None
    else:
        if callable(_lf_get_client) and callable(_lf_propagate_attributes):
            _get_client = _lf_get_client
            _propagate_attributes = _lf_propagate_attributes
            _LANGFUSE_READY = True


def enabled() -> bool:
    """Return True when Langfuse tracing is available and configured."""
    return _LANGFUSE_READY and _get_client is not None and _propagate_attributes is not None


def get_client() -> Any | None:
    """Return the Langfuse client, or None when tracing is unavailable."""
    if not enabled():
        return None
    client_factory = _get_client
    if client_factory is None:
        return None
    try:
        return client_factory()
    except Exception:
        return None


@contextmanager
def start_span(
    *,
    name: str,
    as_type: str = "span",
    input: Any | None = None,
    output: Any | None = None,
    model: str | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
):
    """Start a Langfuse observation if available, else yield None."""
    client = get_client()
    if client is None:
        with nullcontext(None) as span:
            yield span
        return

    params: dict[str, Any] = {
        "name": name,
        "as_type": as_type,
    }
    if input is not None:
        params["input"] = input
    if output is not None:
        params["output"] = output
    if model is not None:
        params["model"] = model
    if metadata:
        params["metadata"] = metadata
    params.update(kwargs)

    with client.start_as_current_observation(**params) as span:
        yield span


@contextmanager
def propagate(
    *,
    session_id: str | None = None,
    user_id: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    trace_name: str | None = None,
):
    """Propagate Langfuse trace/session/user attributes if available."""
    if not enabled():
        with nullcontext():
            yield
        return

    params: dict[str, Any] = {}
    if session_id:
        params["session_id"] = str(session_id)
    if user_id:
        params["user_id"] = str(user_id)
    if tags:
        params["tags"] = [str(tag) for tag in tags if tag]
    if metadata:
        params["metadata"] = metadata
    if trace_name:
        params["trace_name"] = trace_name

    if not params:
        with nullcontext():
            yield
        return

    propagate_fn = _propagate_attributes
    if propagate_fn is None:
        with nullcontext():
            yield
        return

    with propagate_fn(**params):
        yield


def update(
    observation: Any | None,
    *,
    input: Any | None = None,
    output: Any | None = None,
    metadata: dict[str, Any] | None = None,
    model: str | None = None,
    level: str | None = None,
    status_message: str | None = None,
) -> None:
    """Best-effort update of a Langfuse observation."""
    if observation is None:
        return

    payload: dict[str, Any] = {}
    if input is not None:
        payload["input"] = input
    if output is not None:
        payload["output"] = output
    if metadata:
        payload["metadata"] = metadata
    if model is not None:
        payload["model"] = model
    if level is not None:
        payload["level"] = level
    if status_message is not None:
        payload["status_message"] = status_message

    if not payload:
        return

    try:
        observation.update(**payload)
    except Exception:
        return


def observe_agent_turn(func: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap an async message processor in a Langfuse agent-turn observation."""

    async def wrapper(self, msg, session_key=None, *args: Any, **kwargs: Any) -> Any:
        if msg.channel == "system":
            channel = msg.chat_id.split(":", 1)[0] if ":" in msg.chat_id else "cli"
            trace_session_id = msg.chat_id if ":" in msg.chat_id else f"cli:{msg.chat_id}"
            user_id = None
        else:
            channel = msg.channel
            trace_session_id = session_key or msg.session_key
            user_id = None if msg.sender_id in {"user", "subagent"} else str(msg.sender_id)

        metadata = {"model": getattr(self, "model", None), "channel": channel}
        with start_span(name="agent-turn", input=msg.content, metadata=metadata) as span:
            with propagate(
                session_id=trace_session_id,
                user_id=user_id,
                tags=["nanobot", channel],
                trace_name="agent-turn",
            ):
                result = await func(self, msg, session_key, *args, **kwargs)
            update(span, output=getattr(result, "content", None), metadata=metadata)
            return result

    return wrapper
