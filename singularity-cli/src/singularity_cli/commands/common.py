from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import typer

from ..auth import aclose, resolve_authenticated_client, state_from_ctx
from ..output import emit, handle_exception, success_envelope, unwrap


async def _run_authenticated_call(operation: str, call: Callable[..., Awaitable[Any]], *, ctx: typer.Context, **kwargs: Any) -> dict[str, Any]:
    state = state_from_ctx(ctx)
    base_client = None
    auth_client = None
    try:
        base_client, auth_client, _ = await resolve_authenticated_client(state.context)
        result = unwrap(await call(client=auth_client, **kwargs), operation=operation)
        return success_envelope(
            operation=operation,
            trace_id=state.context.trace_id,
            session_id=state.context.session_id,
            data=result,
        )
    finally:
        await aclose(auth_client)
        await aclose(base_client)


def run_authenticated_call(operation: str, call: Callable[..., Awaitable[Any]], *, ctx: typer.Context, **kwargs: Any) -> None:
    state = state_from_ctx(ctx)
    try:
        emit(asyncio.run(_run_authenticated_call(operation, call, ctx=ctx, **kwargs)), human=state.human)
    except Exception as exc:
        raise typer.Exit(
            handle_exception(
                operation,
                exc,
                human=state.human,
                trace_id=state.context.trace_id,
                session_id=state.context.session_id,
            )
        )


def run_simple(operation: str, fn: Callable[[], dict[str, Any]], *, ctx: typer.Context) -> None:
    state = state_from_ctx(ctx)
    try:
        emit(fn(), human=state.human)
    except Exception as exc:
        raise typer.Exit(
            handle_exception(
                operation,
                exc,
                human=state.human,
                trace_id=state.context.trace_id,
                session_id=state.context.session_id,
            )
        )
