from __future__ import annotations

import os
from dataclasses import dataclass
from uuid import uuid4


class CliError(RuntimeError):
    def __init__(self, kind: str, message: str, *, details: object | None = None, status: int | None = None) -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.details = details
        self.status = status


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


@dataclass(frozen=True)
class RuntimeContext:
    backend_base_url: str
    service_secret: str
    channel: str | None
    chat_id: str | None
    session_id: str
    trace_id: str
    parent_observation_id: str | None = None

    @classmethod
    def from_sources(
        cls,
        *,
        backend_base_url: str | None = None,
        service_secret: str | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
        session_id: str | None = None,
        trace_id: str | None = None,
        parent_observation_id: str | None = None,
    ) -> RuntimeContext:
        resolved_channel = (
            _clean(channel)
            or _clean(os.getenv("NANOBOT_CHANNEL"))
            or _clean(os.getenv("SG_CHANNEL"))
        )
        resolved_chat_id = (
            _clean(chat_id)
            or _clean(os.getenv("NANOBOT_CHAT_ID"))
            or _clean(os.getenv("SG_CHAT_ID"))
        )
        if bool(resolved_channel) != bool(resolved_chat_id):
            raise CliError(
                "validation",
                "channel and chat_id must be provided together",
                details=[{"field": "channel/chat_id", "message": "Both values are required together."}],
            )

        derived_session_id = (
            f"{resolved_channel}:{resolved_chat_id}"
            if resolved_channel and resolved_chat_id
            else (
                _clean(session_id)
                or _clean(os.getenv("NANOBOT_SESSION_ID"))
                or _clean(os.getenv("SG_SESSION_ID"))
            )
        )
        if not derived_session_id:
            raise CliError(
                "auth",
                "No session context is available",
                details=[{"field": "session_id", "message": "Set SG_CHANNEL+SG_CHAT_ID or SG_SESSION_ID."}],
            )

        resolved_trace_id = (
            _clean(trace_id)
            or _clean(os.getenv("NANOBOT_TRACE_ID"))
            or _clean(os.getenv("SG_TRACE_ID"))
            or f"sg-{uuid4().hex}"
        )
        resolved_parent_observation_id = _clean(parent_observation_id) or _clean(
            os.getenv("NANOBOT_PARENT_OBSERVATION_ID")
        ) or _clean(os.getenv("SG_PARENT_OBSERVATION_ID"))
        resolved_backend = (
            _clean(backend_base_url) or _clean(os.getenv("SG_BACKEND_BASE_URL")) or _clean(os.getenv("BACKEND_BASE_URL"))
        )
        if not resolved_backend:
            raise CliError("validation", "BACKEND_BASE_URL is not set")
        resolved_secret = (
            _clean(service_secret)
            or _clean(os.getenv("BACKEND_SERVICE_SECRET"))
            or _clean(os.getenv("SG_SERVICE_SECRET"))
            or _clean(os.getenv("MCP_SERVICE_SECRET"))
        )
        if not resolved_secret:
            raise CliError("auth", "BACKEND_SERVICE_SECRET is not set")

        return cls(
            backend_base_url=resolved_backend,
            service_secret=resolved_secret,
            channel=resolved_channel,
            chat_id=resolved_chat_id,
            session_id=derived_session_id,
            trace_id=resolved_trace_id,
            parent_observation_id=resolved_parent_observation_id,
        )
