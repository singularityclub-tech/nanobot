from __future__ import annotations

import asyncio
import sys
import time

import typer
from singularity_client.api.inbox import get_inbox_item, list_inbox_items, send_inbox_item
from singularity_client.models.send_inbox_item_request import SendInboxItemRequest
from singularity_client.types import UNSET

from ..auth import aclose, resolve_authenticated_client, state_from_ctx
from ..output import emit, handle_exception, success_envelope, unwrap
from .common import run_authenticated_call

inbox_app = typer.Typer(no_args_is_help=True)


@inbox_app.command("send")
def send_command(
    ctx: typer.Context,
    content: str = typer.Argument(...),
    answering: int | None = typer.Option(None, "--answering"),
) -> None:
    run_authenticated_call(
        "inbox.send",
        send_inbox_item.asyncio,
        ctx=ctx,
        body=SendInboxItemRequest(content=content, re_outbox_item_id=answering),
    )


@inbox_app.command("status")
def status_command(ctx: typer.Context, item_id: int = typer.Argument(...)) -> None:
    run_authenticated_call("inbox.status", get_inbox_item.asyncio, ctx=ctx, item_id=item_id)


@inbox_app.command("list")
def list_command(ctx: typer.Context, status: str | None = typer.Option(None, "--status")) -> None:
    run_authenticated_call(
        "inbox.list",
        list_inbox_items.asyncio,
        ctx=ctx,
        status=status if status is not None else UNSET,
    )


@inbox_app.command("wait")
def wait_command(
    ctx: typer.Context,
    item_id: int = typer.Argument(...),
    timeout: int = typer.Option(300, "--timeout"),
) -> None:
    state = state_from_ctx(ctx)
    try:
        exit_code = asyncio.run(_wait_for_item(ctx, item_id=item_id, timeout=timeout))
    except Exception as exc:
        raise typer.Exit(
            handle_exception(
                "inbox.wait",
                exc,
                human=state.human,
                trace_id=state.context.trace_id,
                session_id=state.context.session_id,
            )
        )
    raise typer.Exit(exit_code)


# TODO: SSE / websocket / more efficient connection
async def _wait_for_item(ctx: typer.Context, *, item_id: int, timeout: int) -> int:
    state = state_from_ctx(ctx)
    base_client = None
    auth_client = None
    started = time.monotonic()
    try:
        base_client, auth_client, _ = await resolve_authenticated_client(state.context)
        while True:
            result = unwrap(
                await get_inbox_item.asyncio(client=auth_client, item_id=item_id),
                operation="inbox.wait",
            )
            payload = result.to_dict()
            status = payload.get("status")
            if status == "accepted":
                emit(
                    success_envelope(
                        operation="inbox.wait",
                        trace_id=state.context.trace_id,
                        session_id=state.context.session_id,
                        data=payload,
                    ),
                    human=state.human,
                )
                return 0
            if status == "returned":
                emit(
                    success_envelope(
                        operation="inbox.wait",
                        trace_id=state.context.trace_id,
                        session_id=state.context.session_id,
                        data=payload,
                    ),
                    human=state.human,
                )
                return 1
            if time.monotonic() - started >= timeout:
                return 2
            typer.echo(".", nl=False)
            sys.stdout.flush()
            await asyncio.sleep(3)
    finally:
        typer.echo()
        await aclose(auth_client)
        await aclose(base_client)
