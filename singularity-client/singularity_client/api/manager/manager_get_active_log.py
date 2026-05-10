from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.manager_active_log_response import ManagerActiveLogResponse
from ...types import UNSET, Response, Unset


def _get_kwargs(
    user_id: int,
    *,
    limit: int | Unset = 100,
    x_manager_secret: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(x_manager_secret, Unset):
        headers["x-manager-secret"] = x_manager_secret

    params: dict[str, Any] = {}

    params["limit"] = limit

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/internal/manager/users/{user_id}/active-log".format(
            user_id=quote(str(user_id), safe=""),
        ),
        "params": params,
    }

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | ManagerActiveLogResponse | None:
    if response.status_code == 200:
        response_200 = ManagerActiveLogResponse.from_dict(response.json())

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
) -> Response[HTTPValidationError | ManagerActiveLogResponse]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    user_id: int,
    *,
    client: AuthenticatedClient | Client,
    limit: int | Unset = 100,
    x_manager_secret: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | ManagerActiveLogResponse]:
    """Manager Get Active Log

    Args:
        user_id (int):
        limit (int | Unset):  Default: 100.
        x_manager_secret (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | ManagerActiveLogResponse]
    """

    kwargs = _get_kwargs(
        user_id=user_id,
        limit=limit,
        x_manager_secret=x_manager_secret,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    user_id: int,
    *,
    client: AuthenticatedClient | Client,
    limit: int | Unset = 100,
    x_manager_secret: None | str | Unset = UNSET,
) -> HTTPValidationError | ManagerActiveLogResponse | None:
    """Manager Get Active Log

    Args:
        user_id (int):
        limit (int | Unset):  Default: 100.
        x_manager_secret (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | ManagerActiveLogResponse
    """

    return sync_detailed(
        user_id=user_id,
        client=client,
        limit=limit,
        x_manager_secret=x_manager_secret,
    ).parsed


async def asyncio_detailed(
    user_id: int,
    *,
    client: AuthenticatedClient | Client,
    limit: int | Unset = 100,
    x_manager_secret: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | ManagerActiveLogResponse]:
    """Manager Get Active Log

    Args:
        user_id (int):
        limit (int | Unset):  Default: 100.
        x_manager_secret (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | ManagerActiveLogResponse]
    """

    kwargs = _get_kwargs(
        user_id=user_id,
        limit=limit,
        x_manager_secret=x_manager_secret,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    user_id: int,
    *,
    client: AuthenticatedClient | Client,
    limit: int | Unset = 100,
    x_manager_secret: None | str | Unset = UNSET,
) -> HTTPValidationError | ManagerActiveLogResponse | None:
    """Manager Get Active Log

    Args:
        user_id (int):
        limit (int | Unset):  Default: 100.
        x_manager_secret (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | ManagerActiveLogResponse
    """

    return (
        await asyncio_detailed(
            user_id=user_id,
            client=client,
            limit=limit,
            x_manager_secret=x_manager_secret,
        )
    ).parsed
