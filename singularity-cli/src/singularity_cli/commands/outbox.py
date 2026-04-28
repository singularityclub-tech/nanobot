from __future__ import annotations

import typer

from singularity_client.api.outbox import answer_outbox_item, claim_outbox_item, fail_outbox_item, poll_outbox
from singularity_client.models.answer_request import AnswerRequest
from singularity_client.models.claim_request import ClaimRequest
from singularity_client.models.fail_request import FailRequest

from .common import run_authenticated_call

outbox_app = typer.Typer(no_args_is_help=True)


@outbox_app.command("list")
def list_command(ctx: typer.Context) -> None:
    run_authenticated_call("outbox.list", poll_outbox.asyncio, ctx=ctx)


@outbox_app.command("claim")
def claim_command(
    ctx: typer.Context,
    item_id: int = typer.Argument(...),
    claim_note: str | None = typer.Option(None, "--claim-note"),
) -> None:
    run_authenticated_call(
        "outbox.claim",
        claim_outbox_item.asyncio,
        ctx=ctx,
        item_id=item_id,
        body=ClaimRequest(claim_note=claim_note),
    )


@outbox_app.command("answer")
def answer_command(
    ctx: typer.Context,
    item_id: int = typer.Argument(...),
    response_text: str | None = typer.Option(None, "--response-text"),
) -> None:
    run_authenticated_call(
        "outbox.answer",
        answer_outbox_item.asyncio,
        ctx=ctx,
        item_id=item_id,
        body=AnswerRequest(response_text=response_text),
    )


@outbox_app.command("fail")
def fail_command(
    ctx: typer.Context,
    item_id: int = typer.Argument(...),
    reason: str = typer.Option(..., "--reason"),
) -> None:
    run_authenticated_call(
        "outbox.fail",
        fail_outbox_item.asyncio,
        ctx=ctx,
        item_id=item_id,
        body=FailRequest(reason=reason),
    )
