import datetime
from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.intent import Intent
from ...models.manager_get_panel_pack_response_manager_get_panel_pack import (
    ManagerGetPanelPackResponseManagerGetPanelPack,
)
from ...types import UNSET, Response, Unset


def _get_kwargs(
    user_id: int,
    panel: str,
    *,
    evaluated_at: datetime.datetime | None | Unset = UNSET,
    tz: str | Unset = "UTC",
    intent: Intent | Unset = UNSET,
    x_manager_secret: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(x_manager_secret, Unset):
        headers["x-manager-secret"] = x_manager_secret

    params: dict[str, Any] = {}

    json_evaluated_at: None | str | Unset
    if isinstance(evaluated_at, Unset):
        json_evaluated_at = UNSET
    elif isinstance(evaluated_at, datetime.datetime):
        json_evaluated_at = evaluated_at.isoformat()
    else:
        json_evaluated_at = evaluated_at
    params["evaluated_at"] = json_evaluated_at

    params["tz"] = tz

    json_intent: str | Unset = UNSET
    if not isinstance(intent, Unset):
        json_intent = intent.value

    params["intent"] = json_intent

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/internal/manager/users/{user_id}/panels/{panel}".format(
            user_id=quote(str(user_id), safe=""),
            panel=quote(str(panel), safe=""),
        ),
        "params": params,
    }

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | ManagerGetPanelPackResponseManagerGetPanelPack | None:
    if response.status_code == 200:
        response_200 = ManagerGetPanelPackResponseManagerGetPanelPack.from_dict(response.json())

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
) -> Response[HTTPValidationError | ManagerGetPanelPackResponseManagerGetPanelPack]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    user_id: int,
    panel: str,
    *,
    client: AuthenticatedClient | Client,
    evaluated_at: datetime.datetime | None | Unset = UNSET,
    tz: str | Unset = "UTC",
    intent: Intent | Unset = UNSET,
    x_manager_secret: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | ManagerGetPanelPackResponseManagerGetPanelPack]:
    """Manager Get Panel Pack

    Args:
        user_id (int):
        panel (str):
        evaluated_at (datetime.datetime | None | Unset):
        tz (str | Unset):  Default: 'UTC'.
        intent (Intent | Unset):
        x_manager_secret (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | ManagerGetPanelPackResponseManagerGetPanelPack]
    """

    kwargs = _get_kwargs(
        user_id=user_id,
        panel=panel,
        evaluated_at=evaluated_at,
        tz=tz,
        intent=intent,
        x_manager_secret=x_manager_secret,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    user_id: int,
    panel: str,
    *,
    client: AuthenticatedClient | Client,
    evaluated_at: datetime.datetime | None | Unset = UNSET,
    tz: str | Unset = "UTC",
    intent: Intent | Unset = UNSET,
    x_manager_secret: None | str | Unset = UNSET,
) -> HTTPValidationError | ManagerGetPanelPackResponseManagerGetPanelPack | None:
    """Manager Get Panel Pack

    Args:
        user_id (int):
        panel (str):
        evaluated_at (datetime.datetime | None | Unset):
        tz (str | Unset):  Default: 'UTC'.
        intent (Intent | Unset):
        x_manager_secret (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | ManagerGetPanelPackResponseManagerGetPanelPack
    """

    return sync_detailed(
        user_id=user_id,
        panel=panel,
        client=client,
        evaluated_at=evaluated_at,
        tz=tz,
        intent=intent,
        x_manager_secret=x_manager_secret,
    ).parsed


async def asyncio_detailed(
    user_id: int,
    panel: str,
    *,
    client: AuthenticatedClient | Client,
    evaluated_at: datetime.datetime | None | Unset = UNSET,
    tz: str | Unset = "UTC",
    intent: Intent | Unset = UNSET,
    x_manager_secret: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | ManagerGetPanelPackResponseManagerGetPanelPack]:
    """Manager Get Panel Pack

    Args:
        user_id (int):
        panel (str):
        evaluated_at (datetime.datetime | None | Unset):
        tz (str | Unset):  Default: 'UTC'.
        intent (Intent | Unset):
        x_manager_secret (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | ManagerGetPanelPackResponseManagerGetPanelPack]
    """

    kwargs = _get_kwargs(
        user_id=user_id,
        panel=panel,
        evaluated_at=evaluated_at,
        tz=tz,
        intent=intent,
        x_manager_secret=x_manager_secret,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    user_id: int,
    panel: str,
    *,
    client: AuthenticatedClient | Client,
    evaluated_at: datetime.datetime | None | Unset = UNSET,
    tz: str | Unset = "UTC",
    intent: Intent | Unset = UNSET,
    x_manager_secret: None | str | Unset = UNSET,
) -> HTTPValidationError | ManagerGetPanelPackResponseManagerGetPanelPack | None:
    """Manager Get Panel Pack

    Args:
        user_id (int):
        panel (str):
        evaluated_at (datetime.datetime | None | Unset):
        tz (str | Unset):  Default: 'UTC'.
        intent (Intent | Unset):
        x_manager_secret (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | ManagerGetPanelPackResponseManagerGetPanelPack
    """

    return (
        await asyncio_detailed(
            user_id=user_id,
            panel=panel,
            client=client,
            evaluated_at=evaluated_at,
            tz=tz,
            intent=intent,
            x_manager_secret=x_manager_secret,
        )
    ).parsed
