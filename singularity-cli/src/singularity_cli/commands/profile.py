from __future__ import annotations

from pathlib import Path

import typer

from singularity_client.api.pipelines import patch_goals, patch_steering
from singularity_client.models.profile_goals_request import ProfileGoalsRequest
from singularity_client.models.profile_steering_request import ProfileSteeringRequest

from ..parsers import parse_string_list
from .common import run_authenticated_call

profile_app = typer.Typer(no_args_is_help=True)


@profile_app.command("patch-goals")
def patch_goals_command(
    ctx: typer.Context,
    goal: list[str] = typer.Option(None, "--goal"),
    input_file: Path | None = typer.Option(None, "--input-file"),
) -> None:
    goals = parse_string_list(goal, input_file, "goals")
    run_authenticated_call("profile.patch-goals", patch_goals.asyncio, ctx=ctx, body=ProfileGoalsRequest(goals=goals))


@profile_app.command("patch-steering")
def patch_steering_command(
    ctx: typer.Context,
    steering: list[str] = typer.Option(None, "--steering"),
    input_file: Path | None = typer.Option(None, "--input-file"),
) -> None:
    steering_values = parse_string_list(steering, input_file, "steering")
    run_authenticated_call(
        "profile.patch-steering",
        patch_steering.asyncio,
        ctx=ctx,
        body=ProfileSteeringRequest(steering=steering_values),
    )
