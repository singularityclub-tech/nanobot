from __future__ import annotations

import typer

from singularity_client.api.pipelines import create_escalation
from singularity_client.models.escalation_request import EscalationRequest

from .common import run_authenticated_call

escalation_app = typer.Typer(no_args_is_help=True)


@escalation_app.command("create")
def create_command(
    ctx: typer.Context,
    reason: str = typer.Option(..., "--reason"),
    summary: str = typer.Option(..., "--summary"),
) -> None:
    run_authenticated_call(
        "escalation.create",
        create_escalation.asyncio,
        ctx=ctx,
        body=EscalationRequest(reason=reason, summary=summary),
    )
