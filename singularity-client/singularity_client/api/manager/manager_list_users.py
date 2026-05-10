from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.manager_user_list_response import ManagerUserListResponse
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    x_manager_secret: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(x_manager_secret, Unset):
        headers["x-manager-secret"] = x_manager_secret

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/internal/manager/users",
    }

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | ManagerUserListResponse | None:
    if response.status_code == 200:
        response_200 = ManagerUserListResponse.from_dict(response.json())

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
) -> Response[HTTPValidationError | ManagerUserListResponse]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    x_manager_secret: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | ManagerUserListResponse]:
    """Manager List Users

    Args:
        x_manager_secret (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | ManagerUserListResponse]
    """

    kwargs = _get_kwargs(
        x_manager_secret=x_manager_secret,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient | Client,
    x_manager_secret: None | str | Unset = UNSET,
) -> HTTPValidationError | ManagerUserListResponse | None:
    """Manager List Users

    Args:
        x_manager_secret (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | ManagerUserListResponse
    """

    return sync_detailed(
        client=client,
        x_manager_secret=x_manager_secret,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    x_manager_secret: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | ManagerUserListResponse]:
    """Manager List Users

    Args:
        x_manager_secret (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | ManagerUserListResponse]
    """

    kwargs = _get_kwargs(
        x_manager_secret=x_manager_secret,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    x_manager_secret: None | str | Unset = UNSET,
) -> HTTPValidationError | ManagerUserListResponse | None:
    """Manager List Users

    Args:
        x_manager_secret (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | ManagerUserListResponse
    """

    return (
        await asyncio_detailed(
            client=client,
            x_manager_secret=x_manager_secret,
        )
    ).parsed
