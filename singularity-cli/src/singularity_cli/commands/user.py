from __future__ import annotations

import typer

from singularity_client.api.reads import get_active_experiment, get_active_questions, get_profile

from .common import run_authenticated_call

user_app = typer.Typer(no_args_is_help=True)


@user_app.command("get-profile")
def get_profile_command(ctx: typer.Context) -> None:
    run_authenticated_call("user.get-profile", get_profile.asyncio, ctx=ctx)


@user_app.command("get-active-experiment")
def get_active_experiment_command(ctx: typer.Context) -> None:
    run_authenticated_call("user.get-active-experiment", get_active_experiment.asyncio, ctx=ctx)


@user_app.command("get-active-questions")
def get_active_questions_command(ctx: typer.Context) -> None:
    run_authenticated_call("user.get-active-questions", get_active_questions.asyncio, ctx=ctx)
