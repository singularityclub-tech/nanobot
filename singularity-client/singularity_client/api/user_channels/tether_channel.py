from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.tether_channel_request import TetherChannelRequest
from ...models.tether_channel_response import TetherChannelResponse
from ...types import Response


def _get_kwargs(
    *,
    body: TetherChannelRequest,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/internal/user-channels/tether",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | TetherChannelResponse | None:
    if response.status_code == 200:
        response_200 = TetherChannelResponse.from_dict(response.json())

        return response_200

    if response.status_code == 422:
        response_422 = HTTPValidationError.from_dict(response.json())

        return response_422

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[HTTPValidationError | TetherChannelResponse]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: TetherChannelRequest,
) -> Response[HTTPValidationError | TetherChannelResponse]:
    """Tether Channel

     Tether a channel identifier to a user account.

    TODO: Add authentication/authorization. This endpoint is currently insecure
    and should not be exposed in production without proper security measures.

    This creates a mapping between a channel-specific user identifier
    (e.g., Telegram user ID, WebSocket session ID) and the canonical
    backend user ID.

    Args:
        body (TetherChannelRequest): Request to tether a channel to a user.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | TetherChannelResponse]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient | Client,
    body: TetherChannelRequest,
) -> HTTPValidationError | TetherChannelResponse | None:
    """Tether Channel

     Tether a channel identifier to a user account.

    TODO: Add authentication/authorization. This endpoint is currently insecure
    and should not be exposed in production without proper security measures.

    This creates a mapping between a channel-specific user identifier
    (e.g., Telegram user ID, WebSocket session ID) and the canonical
    backend user ID.

    Args:
        body (TetherChannelRequest): Request to tether a channel to a user.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | TetherChannelResponse
    """

    return sync_detailed(
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: TetherChannelRequest,
) -> Response[HTTPValidationError | TetherChannelResponse]:
    """Tether Channel

     Tether a channel identifier to a user account.

    TODO: Add authentication/authorization. This endpoint is currently insecure
    and should not be exposed in production without proper security measures.

    This creates a mapping between a channel-specific user identifier
    (e.g., Telegram user ID, WebSocket session ID) and the canonical
    backend user ID.

    Args:
        body (TetherChannelRequest): Request to tether a channel to a user.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | TetherChannelResponse]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    body: TetherChannelRequest,
) -> HTTPValidationError | TetherChannelResponse | None:
    """Tether Channel

     Tether a channel identifier to a user account.

    TODO: Add authentication/authorization. This endpoint is currently insecure
    and should not be exposed in production without proper security measures.

    This creates a mapping between a channel-specific user identifier
    (e.g., Telegram user ID, WebSocket session ID) and the canonical
    backend user ID.

    Args:
        body (TetherChannelRequest): Request to tether a channel to a user.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | TetherChannelResponse
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
        )
    ).parsed
