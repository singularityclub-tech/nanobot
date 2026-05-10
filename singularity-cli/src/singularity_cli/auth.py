from __future__ import annotations

from typing import Any

import httpx

from singularity_client import AuthenticatedClient, Client
from singularity_client.api.auth import issue_actor_token
from singularity_client.api.user_channels import resolve_channel
from singularity_client.models.actor_token_request import ActorTokenRequest
from singularity_client.models.actor_token_response import ActorTokenResponse
from singularity_client.models.resolve_channel_request import ResolveChannelRequest
from singularity_client.models.resolve_channel_response import ResolveChannelResponse
from singularity_client.types import UNSET

from .context import CliError, RuntimeContext
from .output import unwrap


def base_client(base_url: str) -> Client:
    return Client(
        base_url=base_url.rstrip("/"),
        timeout=httpx.Timeout(30.0),
        follow_redirects=True,
        raise_on_unexpected_status=True,
    )


def authenticated_client(base_url: str, token: ActorTokenResponse) -> AuthenticatedClient:
    token_type = getattr(token, "token_type", "Bearer")
    prefix = str(token_type).capitalize() if token_type else "Bearer"
    return AuthenticatedClient(
        base_url=base_url.rstrip("/"),
        token=token.access_token,
        prefix=prefix,
        # Pipelines block until LLM work completes; 30s is too short (httpx ReadTimeout on sensemaking, etc.)
        timeout=httpx.Timeout(600.0),
        follow_redirects=True,
        raise_on_unexpected_status=True,
    )


async def aclose(client: Client | AuthenticatedClient | None) -> None:
    if client is None:
        return
    try:
        await client.get_async_httpx_client().aclose()
    except Exception:
        return


async def resolve_authenticated_client(context: RuntimeContext) -> tuple[Client, AuthenticatedClient, ResolveChannelResponse]:
    base = base_client(context.backend_base_url)
    try:
        resolved = unwrap(
            await resolve_channel.asyncio(client=base, body=ResolveChannelRequest(channel_id=context.session_id)),
            operation="resolve_channel",
        )
        if not isinstance(resolved, ResolveChannelResponse):
            raise CliError("runtime", "resolve_channel returned an unexpected payload")
        if resolved.user_id is None:
            raise CliError(
                "auth",
                f"No user mapping exists for session {context.session_id}",
                details={"channel_id": resolved.channel_id, "user_id": resolved.user_id},
            )
        token = unwrap(
            await issue_actor_token.asyncio(
                client=base,
                body=ActorTokenRequest(
                    service_secret=context.service_secret,
                    service="singularity-cli",
                    user_id=resolved.user_id,
                    scopes=["mcp"],
                    trace_id=context.trace_id,
                    session_id=context.session_id,
                ),
            ),
            operation="issue_actor_token",
        )
        if not isinstance(token, ActorTokenResponse):
            raise CliError("runtime", "issue_actor_token returned an unexpected payload")
        return base, authenticated_client(context.backend_base_url, token), resolved
    except Exception:
        await aclose(base)
        raise


async def issue_token_for_context(context: RuntimeContext) -> tuple[Client, ActorTokenResponse, ResolveChannelResponse]:
    base = base_client(context.backend_base_url)
    try:
        resolved = unwrap(
            await resolve_channel.asyncio(client=base, body=ResolveChannelRequest(channel_id=context.session_id)),
            operation="resolve_channel",
        )
        if not isinstance(resolved, ResolveChannelResponse) or resolved.user_id is None:
            raise CliError("auth", "Unable to resolve user for session", details={"session_id": context.session_id})
        token = unwrap(
            await issue_actor_token.asyncio(
                client=base,
                body=ActorTokenRequest(
                    service_secret=context.service_secret,
                    service="singularity-cli",
                    user_id=resolved.user_id,
                    scopes=["mcp"],
                    trace_id=context.trace_id,
                    session_id=context.session_id,
                ),
            ),
            operation="issue_actor_token",
        )
        if not isinstance(token, ActorTokenResponse):
            raise CliError("runtime", "issue_actor_token returned an unexpected payload")
        return base, token, resolved
    except Exception:
        await aclose(base)
        raise


class AppState:
    def __init__(self, *, context: RuntimeContext, human: bool) -> None:
        self.context = context
        self.human = human


def state_from_ctx(ctx: Any) -> AppState:
    state = ctx.obj
    if not isinstance(state, AppState):
        raise CliError("runtime", "CLI state was not initialized")
    return state
