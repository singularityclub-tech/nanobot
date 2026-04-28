from __future__ import annotations

import typer

from singularity_client.api.pipelines import (
    run_baseline_recalculation,
    run_checkin_response,
    run_data_sync,
    run_experiment_evaluation,
    run_experiment_search,
    run_sensemaking,
    run_user_decision,
)
from singularity_client.models.checkin_response_request import CheckinResponseRequest
from singularity_client.models.experiment_evaluation_request import ExperimentEvaluationRequest
from singularity_client.models.experiment_search_request import ExperimentSearchRequest
from singularity_client.models.sensemaking_request import SensemakingRequest
from singularity_client.models.sensemaking_request_window import SensemakingRequestWindow
from singularity_client.models.user_decision_request import UserDecisionRequest
from singularity_client.models.user_decision_request_decision import UserDecisionRequestDecision
from singularity_client.types import UNSET

from .common import run_authenticated_call

pipeline_app = typer.Typer(no_args_is_help=True)


@pipeline_app.command("checkin-response")
def checkin_response_command(ctx: typer.Context, context_hint: str | None = typer.Option(None, "--context-hint")) -> None:
    run_authenticated_call(
        "pipeline.checkin-response",
        run_checkin_response.asyncio,
        ctx=ctx,
        body=CheckinResponseRequest(context_hint=context_hint if context_hint is not None else UNSET),
    )


@pipeline_app.command("data-sync")
def data_sync_command(ctx: typer.Context) -> None:
    run_authenticated_call("pipeline.data-sync", run_data_sync.asyncio, ctx=ctx)


@pipeline_app.command("experiment-search")
def experiment_search_command(ctx: typer.Context, goal: str = typer.Option(..., "--goal")) -> None:
    run_authenticated_call(
        "pipeline.experiment-search",
        run_experiment_search.asyncio,
        ctx=ctx,
        body=ExperimentSearchRequest(goal=goal),
    )


@pipeline_app.command("user-decision")
def user_decision_command(
    ctx: typer.Context,
    outbox_item_id: int = typer.Option(..., "--outbox-item-id"),
    decision: UserDecisionRequestDecision = typer.Option(..., "--decision"),
    remark: str | None = typer.Option(None, "--remark"),
) -> None:
    run_authenticated_call(
        "pipeline.user-decision",
        run_user_decision.asyncio,
        ctx=ctx,
        body=UserDecisionRequest(
            outbox_item_id=outbox_item_id,
            decision=decision,
            remark=remark if remark is not None else UNSET,
        ),
    )


@pipeline_app.command("experiment-evaluation")
def experiment_evaluation_command(
    ctx: typer.Context,
    early_stop: bool = typer.Option(False, "--early-stop/--no-early-stop"),
    reason: str | None = typer.Option(None, "--reason"),
) -> None:
    run_authenticated_call(
        "pipeline.experiment-evaluation",
        run_experiment_evaluation.asyncio,
        ctx=ctx,
        body=ExperimentEvaluationRequest(early_stop=early_stop, reason=reason if reason is not None else UNSET),
    )


@pipeline_app.command("baseline-recalculation")
def baseline_recalculation_command(ctx: typer.Context) -> None:
    run_authenticated_call("pipeline.baseline-recalculation", run_baseline_recalculation.asyncio, ctx=ctx)


@pipeline_app.command("sensemaking")
def sensemaking_command(
    ctx: typer.Context,
    context_hint: str | None = typer.Option(None, "--context-hint"),
    window: SensemakingRequestWindow | None = typer.Option(
        None,
        "--window",
        help="Projection window: local_day, rolling_7d, rolling_28d",
    ),
) -> None:
    run_authenticated_call(
        "pipeline.sensemaking",
        run_sensemaking.asyncio,
        ctx=ctx,
        body=SensemakingRequest(
            context_hint=context_hint if context_hint is not None else UNSET,
            window=window if window is not None else UNSET,
        ),
    )
