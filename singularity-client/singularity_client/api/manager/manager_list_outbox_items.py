from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.manager_outbox_list_response import ManagerOutboxListResponse
from ...types import UNSET, Response, Unset


def _get_kwargs(
    user_id: int,
    *,
    state: None | str | Unset = UNSET,
    limit: int | Unset = 100,
    x_manager_secret: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(x_manager_secret, Unset):
        headers["x-manager-secret"] = x_manager_secret

    params: dict[str, Any] = {}

    json_state: None | str | Unset
    if isinstance(state, Unset):
        json_state = UNSET
    else:
        json_state = state
    params["state"] = json_state

    params["limit"] = limit

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/internal/manager/users/{user_id}/outbox".format(
            user_id=quote(str(user_id), safe=""),
        ),
        "params": params,
    }

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | ManagerOutboxListResponse | None:
    if response.status_code == 200:
        response_200 = ManagerOutboxListResponse.from_dict(response.json())

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
) -> Response[HTTPValidationError | ManagerOutboxListResponse]:
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
    state: None | str | Unset = UNSET,
    limit: int | Unset = 100,
    x_manager_secret: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | ManagerOutboxListResponse]:
    """Manager List Outbox Items

    Args:
        user_id (int):
        state (None | str | Unset):
        limit (int | Unset):  Default: 100.
        x_manager_secret (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | ManagerOutboxListResponse]
    """

    kwargs = _get_kwargs(
        user_id=user_id,
        state=state,
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
    state: None | str | Unset = UNSET,
    limit: int | Unset = 100,
    x_manager_secret: None | str | Unset = UNSET,
) -> HTTPValidationError | ManagerOutboxListResponse | None:
    """Manager List Outbox Items

    Args:
        user_id (int):
        state (None | str | Unset):
        limit (int | Unset):  Default: 100.
        x_manager_secret (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | ManagerOutboxListResponse
    """

    return sync_detailed(
        user_id=user_id,
        client=client,
        state=state,
        limit=limit,
        x_manager_secret=x_manager_secret,
    ).parsed


async def asyncio_detailed(
    user_id: int,
    *,
    client: AuthenticatedClient | Client,
    state: None | str | Unset = UNSET,
    limit: int | Unset = 100,
    x_manager_secret: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | ManagerOutboxListResponse]:
    """Manager List Outbox Items

    Args:
        user_id (int):
        state (None | str | Unset):
        limit (int | Unset):  Default: 100.
        x_manager_secret (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | ManagerOutboxListResponse]
    """

    kwargs = _get_kwargs(
        user_id=user_id,
        state=state,
        limit=limit,
        x_manager_secret=x_manager_secret,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    user_id: int,
    *,
    client: AuthenticatedClient | Client,
    state: None | str | Unset = UNSET,
    limit: int | Unset = 100,
    x_manager_secret: None | str | Unset = UNSET,
) -> HTTPValidationError | ManagerOutboxListResponse | None:
    """Manager List Outbox Items

    Args:
        user_id (int):
        state (None | str | Unset):
        limit (int | Unset):  Default: 100.
        x_manager_secret (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | ManagerOutboxListResponse
    """

    return (
        await asyncio_detailed(
            user_id=user_id,
            client=client,
            state=state,
            limit=limit,
            x_manager_secret=x_manager_secret,
        )
    ).parsed
