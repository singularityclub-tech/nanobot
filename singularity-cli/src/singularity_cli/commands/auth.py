from __future__ import annotations

import asyncio

import typer

from ..auth import issue_token_for_context, state_from_ctx
from ..output import emit, handle_exception, success_envelope, unwrap

auth_app = typer.Typer(no_args_is_help=True)


@auth_app.command("issue-actor-token")
def issue_actor_token_command(ctx: typer.Context) -> None:
    state = state_from_ctx(ctx)

    async def _run() -> dict:
        client = None
        try:
            client, token, resolved = await issue_token_for_context(state.context)
            result = {"token": unwrap(token, operation="issue_actor_token").to_dict(), "resolved_user": resolved.to_dict()}
            return success_envelope(
                operation="auth.issue-actor-token",
                trace_id=state.context.trace_id,
                session_id=state.context.session_id,
                data=result,
            )
        finally:
            if client is not None:
                await client.get_async_httpx_client().aclose()

    try:
        emit(asyncio.run(_run()), human=state.human)
    except Exception as exc:
        raise typer.Exit(
            handle_exception(
                "auth.issue-actor-token",
                exc,
                human=state.human,
                trace_id=state.context.trace_id,
                session_id=state.context.session_id,
            )
        )
