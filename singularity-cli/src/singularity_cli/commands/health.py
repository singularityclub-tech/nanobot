from __future__ import annotations

import typer

from singularity_client.api.default import health

from .common import run_authenticated_call


def register(app) -> None:
    @app.command("health")
    def health_command(ctx: typer.Context) -> None:
        run_authenticated_call("health", health.asyncio, ctx=ctx)
