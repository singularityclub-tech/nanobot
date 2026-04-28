from __future__ import annotations

import json
from datetime import datetime

import pytest
from typer.testing import CliRunner

from singularity_client.models.actor_token_response import ActorTokenResponse
from singularity_client.models.profile_response import ProfileResponse
from singularity_client.models.resolve_channel_response import ResolveChannelResponse

from singularity_cli.app import app
from singularity_cli.context import RuntimeContext
from singularity_cli.parsers import parse_datetime

runner = CliRunner()


def test_runtime_context_prefers_channel_and_chat_and_env(monkeypatch):
    monkeypatch.setenv("SG_SESSION_ID", "websocket:old")
    monkeypatch.setenv("BACKEND_BASE_URL", "http://backend")
    monkeypatch.setenv("BACKEND_SERVICE_SECRET", "secret")
    context = RuntimeContext.from_sources(channel="telegram", chat_id="123")
    assert context.session_id == "telegram:123"
    assert context.backend_base_url == "http://backend"
    assert context.trace_id.startswith("sg-")


@pytest.mark.asyncio
async def test_resolve_authenticated_client(monkeypatch):
    async def fake_resolve(*, client, body):
        return ResolveChannelResponse(channel_id=body.channel_id, user_id=7)

    async def fake_issue_token(*, client, body):
        return ActorTokenResponse(access_token="token-123", expires_in=300, token_type="bearer")

    monkeypatch.setattr("singularity_cli.auth.resolve_channel.asyncio", fake_resolve)
    monkeypatch.setattr("singularity_cli.auth.issue_actor_token.asyncio", fake_issue_token)

    context = RuntimeContext.from_sources(
        backend_base_url="http://backend",
        service_secret="secret",
        channel="websocket",
        chat_id="surface-probe",
        trace_id="trace-123",
    )
    from singularity_cli.auth import resolve_authenticated_client

    base_client, auth_client, resolved = await resolve_authenticated_client(context)
    assert resolved.user_id == 7
    assert auth_client.token == "token-123"
    assert auth_client.prefix == "Bearer"
    await base_client.get_async_httpx_client().aclose()
    await auth_client.get_async_httpx_client().aclose()


def test_command_tree_has_layout_groups():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for group in [
        "health",
        "auth",
        "user",
        "profile",
        "projection",
        "observation",
        "escalation",
        "outbox",
        "pipeline",
    ]:
        assert group in result.stdout


def test_user_get_profile(monkeypatch):
    async def fake_resolve(*, client, body):
        return ResolveChannelResponse(channel_id=body.channel_id, user_id=7)

    async def fake_issue_token(*, client, body):
        return ActorTokenResponse(access_token="token-123", expires_in=300, token_type="bearer")

    async def fake_get_profile(*, client, authorization=None):
        return ProfileResponse.from_dict({"user_id": 7, "profile": {"goals": ["energy"]}})

    monkeypatch.setattr("singularity_cli.auth.resolve_channel.asyncio", fake_resolve)
    monkeypatch.setattr("singularity_cli.auth.issue_actor_token.asyncio", fake_issue_token)
    monkeypatch.setattr("singularity_cli.commands.user.get_profile.asyncio", fake_get_profile)

    result = runner.invoke(
        app,
        [
            "user",
            "get-profile",
        ],
        env={
            "BACKEND_BASE_URL": "http://backend",
            "BACKEND_SERVICE_SECRET": "secret",
            "SG_CHANNEL": "websocket",
            "SG_CHAT_ID": "surface-probe",
        },
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["profile"]["goals"] == ["energy"]


def test_profile_patch_goals_uses_string_list(monkeypatch):
    async def fake_resolve(*, client, body):
        return ResolveChannelResponse(channel_id=body.channel_id, user_id=7)

    async def fake_issue_token(*, client, body):
        return ActorTokenResponse(access_token="token-123", expires_in=300, token_type="bearer")

    captured = {}

    async def fake_patch_goals(*, client, body, authorization=None):
        captured["goals"] = body.goals
        return ProfileResponse.from_dict({"user_id": 7, "profile": {"goals": body.goals}})

    monkeypatch.setattr("singularity_cli.auth.resolve_channel.asyncio", fake_resolve)
    monkeypatch.setattr("singularity_cli.auth.issue_actor_token.asyncio", fake_issue_token)
    monkeypatch.setattr("singularity_cli.commands.profile.patch_goals.asyncio", fake_patch_goals)

    result = runner.invoke(
        app,
        [
            "profile",
            "patch-goals",
            "--goal",
            "Daytime energy",
            "--goal",
            "Sleep consistency",
        ],
        env={
            "BACKEND_BASE_URL": "http://backend",
            "BACKEND_SERVICE_SECRET": "secret",
            "SG_CHANNEL": "websocket",
            "SG_CHAT_ID": "surface-probe",
        },
    )
    assert result.exit_code == 0
    assert captured["goals"] == ["Daytime energy", "Sleep consistency"]


def test_projection_get_parses_datetime(monkeypatch):
    async def fake_resolve(*, client, body):
        return ResolveChannelResponse(channel_id=body.channel_id, user_id=7)

    async def fake_issue_token(*, client, body):
        return ActorTokenResponse(access_token="token-123", expires_in=300, token_type="bearer")

    captured = {}

    async def fake_projection(*, client, panel, since, until, tz, window, include_series, authorization=None):
        captured["panel"] = panel
        captured["since"] = since
        captured["until"] = until
        return {"panel": panel}

    monkeypatch.setattr("singularity_cli.auth.resolve_channel.asyncio", fake_resolve)
    monkeypatch.setattr("singularity_cli.auth.issue_actor_token.asyncio", fake_issue_token)
    monkeypatch.setattr("singularity_cli.commands.projection.get_projection_panel.asyncio", fake_projection)

    result = runner.invoke(
        app,
        [
            "projection",
            "get",
            "--panel",
            "energy",
            "--since",
            "2026-04-27T10:00:00+03:00",
            "--until",
            "2026-04-27T12:00:00+03:00",
        ],
        env={
            "BACKEND_BASE_URL": "http://backend",
            "BACKEND_SERVICE_SECRET": "secret",
            "SG_CHANNEL": "websocket",
            "SG_CHAT_ID": "surface-probe",
        },
    )
    assert result.exit_code == 0
    assert captured["panel"] == "energy"
    assert isinstance(captured["since"], datetime)
    assert isinstance(captured["until"], datetime)


def test_parse_datetime_rejects_invalid():
    with pytest.raises(Exception):
        parse_datetime("not-a-datetime")


def test_runtime_context_reads_nanobot_env(monkeypatch):
    monkeypatch.setenv("BACKEND_BASE_URL", "http://backend")
    monkeypatch.setenv("BACKEND_SERVICE_SECRET", "secret")
    monkeypatch.setenv("NANOBOT_CHANNEL", "telegram")
    monkeypatch.setenv("NANOBOT_CHAT_ID", "42")
    monkeypatch.setenv("NANOBOT_TRACE_ID", "trace-abc")

    context = RuntimeContext.from_sources()
    assert context.session_id == "telegram:42"
    assert context.trace_id == "trace-abc"
