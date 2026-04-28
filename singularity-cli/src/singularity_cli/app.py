from __future__ import annotations

import typer

from .auth import AppState
from .commands.auth import auth_app
from .commands.escalation import escalation_app
from .commands.health import register as register_health
from .commands.observation import observation_app
from .commands.outbox import outbox_app
from .commands.pipeline import pipeline_app
from .commands.profile import profile_app
from .commands.projection import projection_app
from .commands.user import user_app
from .context import RuntimeContext

app = typer.Typer(add_completion=False, no_args_is_help=True, pretty_exceptions_enable=False)


@app.callback()
def app_callback(
    ctx: typer.Context,
    human: bool = typer.Option(False, "--human"),
    backend_base_url: str | None = typer.Option(
        None,
        "--backend-base-url",
        envvar=["BACKEND_BASE_URL", "SG_BACKEND_BASE_URL"],
    ),
    service_secret: str | None = typer.Option(
        None,
        "--service-secret",
        # TODO: read from a mounted secret instead because otherwise nanobot should expose the secret in the exec tool, it's dangerous
        envvar=[
            "BACKEND_SERVICE_SECRET",
            "SG_SERVICE_SECRET",
            "MCP_SERVICE_SECRET",
        ],
    ),
    channel: str | None = typer.Option(
        None,
        "--channel",
        envvar=["NANOBOT_CHANNEL", "SG_CHANNEL"],
    ),
    chat_id: str | None = typer.Option(
        None,
        "--chat-id",
        envvar=["NANOBOT_CHAT_ID", "SG_CHAT_ID"],
    ),
    session_id: str | None = typer.Option(
        None,
        "--session-id",
        envvar=["NANOBOT_SESSION_ID", "SG_SESSION_ID"],
    ),
    trace_id: str | None = typer.Option(
        None,
        "--trace-id",
        envvar=["NANOBOT_TRACE_ID", "SG_TRACE_ID"],
    ),
) -> None:
    context = RuntimeContext.from_sources(
        backend_base_url=backend_base_url,
        service_secret=service_secret,
        channel=channel,
        chat_id=chat_id,
        session_id=session_id,
        trace_id=trace_id,
    )
    ctx.obj = AppState(context=context, human=human)


register_health(app)
app.add_typer(auth_app, name="auth")
app.add_typer(user_app, name="user")
app.add_typer(profile_app, name="profile")
# app.add_typer(projection_app, name="projection")
app.add_typer(observation_app, name="observation")
app.add_typer(escalation_app, name="escalation")
app.add_typer(outbox_app, name="outbox")
app.add_typer(pipeline_app, name="pipeline")
