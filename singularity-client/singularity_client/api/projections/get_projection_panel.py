import datetime
from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.get_projection_panel_window import GetProjectionPanelWindow
from ...models.http_validation_error import HTTPValidationError
from ...models.projection_panel_response import ProjectionPanelResponse
from ...types import UNSET, Response, Unset


def _get_kwargs(
    panel: str,
    *,
    until: datetime.datetime | None | Unset = UNSET,
    since: datetime.datetime | None | Unset = UNSET,
    tz: str | Unset = "UTC",
    window: GetProjectionPanelWindow | Unset = GetProjectionPanelWindow.LOCAL_DAY,
    include_series: bool | Unset = True,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["authorization"] = authorization

    params: dict[str, Any] = {}

    json_until: None | str | Unset
    if isinstance(until, Unset):
        json_until = UNSET
    elif isinstance(until, datetime.datetime):
        json_until = until.isoformat()
    else:
        json_until = until
    params["until"] = json_until

    json_since: None | str | Unset
    if isinstance(since, Unset):
        json_since = UNSET
    elif isinstance(since, datetime.datetime):
        json_since = since.isoformat()
    else:
        json_since = since
    params["since"] = json_since

    params["tz"] = tz

    json_window: str | Unset = UNSET
    if not isinstance(window, Unset):
        json_window = window.value

    params["window"] = json_window

    params["include_series"] = include_series

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/internal/users/me/projections/{panel}".format(
            panel=quote(str(panel), safe=""),
        ),
        "params": params,
    }

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | ProjectionPanelResponse | None:
    if response.status_code == 200:
        response_200 = ProjectionPanelResponse.from_dict(response.json())

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
) -> Response[HTTPValidationError | ProjectionPanelResponse]:
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
    until: datetime.datetime | None | Unset = UNSET,
    since: datetime.datetime | None | Unset = UNSET,
    tz: str | Unset = "UTC",
    window: GetProjectionPanelWindow | Unset = GetProjectionPanelWindow.LOCAL_DAY,
    include_series: bool | Unset = True,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | ProjectionPanelResponse]:
    """Get Projection Panel

    Args:
        panel (str):
        until (datetime.datetime | None | Unset):
        since (datetime.datetime | None | Unset):
        tz (str | Unset):  Default: 'UTC'.
        window (GetProjectionPanelWindow | Unset):  Default: GetProjectionPanelWindow.LOCAL_DAY.
        include_series (bool | Unset):  Default: True.
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | ProjectionPanelResponse]
    """

    kwargs = _get_kwargs(
        panel=panel,
        until=until,
        since=since,
        tz=tz,
        window=window,
        include_series=include_series,
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
    until: datetime.datetime | None | Unset = UNSET,
    since: datetime.datetime | None | Unset = UNSET,
    tz: str | Unset = "UTC",
    window: GetProjectionPanelWindow | Unset = GetProjectionPanelWindow.LOCAL_DAY,
    include_series: bool | Unset = True,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | ProjectionPanelResponse | None:
    """Get Projection Panel

    Args:
        panel (str):
        until (datetime.datetime | None | Unset):
        since (datetime.datetime | None | Unset):
        tz (str | Unset):  Default: 'UTC'.
        window (GetProjectionPanelWindow | Unset):  Default: GetProjectionPanelWindow.LOCAL_DAY.
        include_series (bool | Unset):  Default: True.
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | ProjectionPanelResponse
    """

    return sync_detailed(
        panel=panel,
        client=client,
        until=until,
        since=since,
        tz=tz,
        window=window,
        include_series=include_series,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    panel: str,
    *,
    client: AuthenticatedClient | Client,
    until: datetime.datetime | None | Unset = UNSET,
    since: datetime.datetime | None | Unset = UNSET,
    tz: str | Unset = "UTC",
    window: GetProjectionPanelWindow | Unset = GetProjectionPanelWindow.LOCAL_DAY,
    include_series: bool | Unset = True,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | ProjectionPanelResponse]:
    """Get Projection Panel

    Args:
        panel (str):
        until (datetime.datetime | None | Unset):
        since (datetime.datetime | None | Unset):
        tz (str | Unset):  Default: 'UTC'.
        window (GetProjectionPanelWindow | Unset):  Default: GetProjectionPanelWindow.LOCAL_DAY.
        include_series (bool | Unset):  Default: True.
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | ProjectionPanelResponse]
    """

    kwargs = _get_kwargs(
        panel=panel,
        until=until,
        since=since,
        tz=tz,
        window=window,
        include_series=include_series,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    panel: str,
    *,
    client: AuthenticatedClient | Client,
    until: datetime.datetime | None | Unset = UNSET,
    since: datetime.datetime | None | Unset = UNSET,
    tz: str | Unset = "UTC",
    window: GetProjectionPanelWindow | Unset = GetProjectionPanelWindow.LOCAL_DAY,
    include_series: bool | Unset = True,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | ProjectionPanelResponse | None:
    """Get Projection Panel

    Args:
        panel (str):
        until (datetime.datetime | None | Unset):
        since (datetime.datetime | None | Unset):
        tz (str | Unset):  Default: 'UTC'.
        window (GetProjectionPanelWindow | Unset):  Default: GetProjectionPanelWindow.LOCAL_DAY.
        include_series (bool | Unset):  Default: True.
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | ProjectionPanelResponse
    """

    return (
        await asyncio_detailed(
            panel=panel,
            client=client,
            until=until,
            since=since,
            tz=tz,
            window=window,
            include_series=include_series,
            authorization=authorization,
        )
    ).parsed
