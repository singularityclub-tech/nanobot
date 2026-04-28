from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import typer

from singularity_client.api.observations import write_observation
from singularity_client.models.observation_kind import ObservationKind
from singularity_client.models.observation_write_request import ObservationWriteRequest
from singularity_client.types import UNSET

from ..parsers import load_json_object
from .common import run_authenticated_call

observation_app = typer.Typer(no_args_is_help=True)


@observation_app.command("write")
def write_command(
    ctx: typer.Context,
    measurement_code: str | None = typer.Option(None, "--measurement-code"),
    kind: ObservationKind | None = typer.Option(None, "--kind"),
    value: str | None = typer.Option(None, "--value"),
    observed_at: datetime | None = typer.Option(None, "--observed-at"),
    recorded_at: datetime | None = typer.Option(None, "--recorded-at"),
    input_file: Path | None = typer.Option(None, "--input-file"),
) -> None:
    if input_file is not None:
        payload = load_json_object(input_file)
        body = ObservationWriteRequest.from_dict(payload)
    else:
        if measurement_code is None or kind is None or value is None or observed_at is None:
            raise typer.BadParameter("measurement-code, kind, value and observed-at are required without --input-file")
        body = ObservationWriteRequest(
            measurement_code=measurement_code,
            kind=kind,
            value=value,
            observed_at=observed_at,
            recorded_at=recorded_at if recorded_at is not None else UNSET,
            metadata=UNSET,
        )
    run_authenticated_call("observation.write", write_observation.asyncio, ctx=ctx, body=body)
