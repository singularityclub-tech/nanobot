from __future__ import annotations

import typer

from singularity_client.api.projections import get_all_projection_panels, get_projection_panel, list_projections
from singularity_client.models.get_all_projection_panels_window import GetAllProjectionPanelsWindow
from singularity_client.models.get_projection_panel_window import GetProjectionPanelWindow
from singularity_client.types import UNSET

from ..parsers import parse_datetime
from .common import run_authenticated_call

projection_app = typer.Typer(no_args_is_help=True)


@projection_app.command("list")
def list_command(ctx: typer.Context) -> None:
    run_authenticated_call("projection.list", list_projections.asyncio, ctx=ctx)


@projection_app.command("get")
def get_command(
    ctx: typer.Context,
    panel: str = typer.Option(..., "--panel"),
    since: str | None = typer.Option(None, "--since"),
    until: str | None = typer.Option(None, "--until"),
    tz: str = typer.Option("UTC", "--tz"),
    window: GetProjectionPanelWindow = typer.Option(GetProjectionPanelWindow.LOCAL_DAY, "--window"),
    include_series: bool = typer.Option(True, "--include-series/--no-include-series"),
) -> None:
    run_authenticated_call(
        "projection.get",
        get_projection_panel.asyncio,
        ctx=ctx,
        panel=panel,
        since=parse_datetime(since) if since else UNSET,
        until=parse_datetime(until) if until else UNSET,
        tz=tz,
        window=window,
        include_series=include_series,
    )


@projection_app.command("get-all")
def get_all_command(
    ctx: typer.Context,
    since: str | None = typer.Option(None, "--since"),
    until: str | None = typer.Option(None, "--until"),
    tz: str = typer.Option("UTC", "--tz"),
    window: GetAllProjectionPanelsWindow = typer.Option(GetAllProjectionPanelsWindow.LOCAL_DAY, "--window"),
    include_series: bool = typer.Option(False, "--include-series/--no-include-series"),
) -> None:
    run_authenticated_call(
        "projection.get-all",
        get_all_projection_panels.asyncio,
        ctx=ctx,
        since=parse_datetime(since) if since else UNSET,
        until=parse_datetime(until) if until else UNSET,
        tz=tz,
        window=window,
        include_series=include_series,
    )
