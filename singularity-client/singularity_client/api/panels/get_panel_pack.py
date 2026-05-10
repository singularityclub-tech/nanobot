import datetime
from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.get_panel_pack_response_get_panel_pack import GetPanelPackResponseGetPanelPack
from ...models.http_validation_error import HTTPValidationError
from ...models.intent import Intent
from ...types import UNSET, Response, Unset


def _get_kwargs(
    panel: str,
    *,
    intent: Intent | Unset = UNSET,
    tz: str | Unset = "UTC",
    evaluated_at: datetime.datetime | None | Unset = UNSET,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["authorization"] = authorization

    params: dict[str, Any] = {}

    json_intent: str | Unset = UNSET
    if not isinstance(intent, Unset):
        json_intent = intent.value

    params["intent"] = json_intent

    params["tz"] = tz

    json_evaluated_at: None | str | Unset
    if isinstance(evaluated_at, Unset):
        json_evaluated_at = UNSET
    elif isinstance(evaluated_at, datetime.datetime):
        json_evaluated_at = evaluated_at.isoformat()
    else:
        json_evaluated_at = evaluated_at
    params["evaluated_at"] = json_evaluated_at

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/internal/users/me/panels/{panel}".format(
            panel=quote(str(panel), safe=""),
        ),
        "params": params,
    }

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> GetPanelPackResponseGetPanelPack | HTTPValidationError | None:
    if response.status_code == 200:
        response_200 = GetPanelPackResponseGetPanelPack.from_dict(response.json())

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
) -> Response[GetPanelPackResponseGetPanelPack | HTTPValidationError]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    panel: str,
    *,
    client: AuthenticatedClient | Client,
    intent: Intent | Unset = UNSET,
    tz: str | Unset = "UTC",
    evaluated_at: datetime.datetime | None | Unset = UNSET,
    authorization: None | str | Unset = UNSET,
) -> Response[GetPanelPackResponseGetPanelPack | HTTPValidationError]:
    """Get User Panel Pack

    Args:
        panel (str):
        intent (Intent | Unset):
        tz (str | Unset):  Default: 'UTC'.
        evaluated_at (datetime.datetime | None | Unset):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[GetPanelPackResponseGetPanelPack | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        panel=panel,
        intent=intent,
        tz=tz,
        evaluated_at=evaluated_at,
        authorization=authorization,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    panel: str,
    *,
    client: AuthenticatedClient | Client,
    intent: Intent | Unset = UNSET,
    tz: str | Unset = "UTC",
    evaluated_at: datetime.datetime | None | Unset = UNSET,
    authorization: None | str | Unset = UNSET,
) -> GetPanelPackResponseGetPanelPack | HTTPValidationError | None:
    """Get User Panel Pack

    Args:
        panel (str):
        intent (Intent | Unset):
        tz (str | Unset):  Default: 'UTC'.
        evaluated_at (datetime.datetime | None | Unset):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        GetPanelPackResponseGetPanelPack | HTTPValidationError
    """

    return sync_detailed(
        panel=panel,
        client=client,
        intent=intent,
        tz=tz,
        evaluated_at=evaluated_at,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    panel: str,
    *,
    client: AuthenticatedClient | Client,
    intent: Intent | Unset = UNSET,
    tz: str | Unset = "UTC",
    evaluated_at: datetime.datetime | None | Unset = UNSET,
    authorization: None | str | Unset = UNSET,
) -> Response[GetPanelPackResponseGetPanelPack | HTTPValidationError]:
    """Get User Panel Pack

    Args:
        panel (str):
        intent (Intent | Unset):
        tz (str | Unset):  Default: 'UTC'.
        evaluated_at (datetime.datetime | None | Unset):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[GetPanelPackResponseGetPanelPack | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        panel=panel,
        intent=intent,
        tz=tz,
        evaluated_at=evaluated_at,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    panel: str,
    *,
    client: AuthenticatedClient | Client,
    intent: Intent | Unset = UNSET,
    tz: str | Unset = "UTC",
    evaluated_at: datetime.datetime | None | Unset = UNSET,
    authorization: None | str | Unset = UNSET,
) -> GetPanelPackResponseGetPanelPack | HTTPValidationError | None:
    """Get User Panel Pack

    Args:
        panel (str):
        intent (Intent | Unset):
        tz (str | Unset):  Default: 'UTC'.
        evaluated_at (datetime.datetime | None | Unset):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        GetPanelPackResponseGetPanelPack | HTTPValidationError
    """

    return (
        await asyncio_detailed(
            panel=panel,
            client=client,
            intent=intent,
            tz=tz,
            evaluated_at=evaluated_at,
            authorization=authorization,
        )
    ).parsed
