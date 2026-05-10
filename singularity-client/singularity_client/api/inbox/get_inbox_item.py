from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.inbox_item_response import InboxItemResponse
from ...types import UNSET, Response, Unset


def _get_kwargs(
    item_id: int,
    *,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/internal/mcp/users/me/inbox/{item_id}".format(
            item_id=quote(str(item_id), safe=""),
        ),
    }

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | InboxItemResponse | None:
    if response.status_code == 200:
        response_200 = InboxItemResponse.from_dict(response.json())

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
) -> Response[HTTPValidationError | InboxItemResponse]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    item_id: int,
    *,
    client: AuthenticatedClient | Client,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | InboxItemResponse]:
    """Get Inbox Item

    Args:
        item_id (int):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | InboxItemResponse]
    """

    kwargs = _get_kwargs(
        item_id=item_id,
        authorization=authorization,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    item_id: int,
    *,
    client: AuthenticatedClient | Client,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | InboxItemResponse | None:
    """Get Inbox Item

    Args:
        item_id (int):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | InboxItemResponse
    """

    return sync_detailed(
        item_id=item_id,
        client=client,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    item_id: int,
    *,
    client: AuthenticatedClient | Client,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | InboxItemResponse]:
    """Get Inbox Item

    Args:
        item_id (int):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | InboxItemResponse]
    """

    kwargs = _get_kwargs(
        item_id=item_id,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    item_id: int,
    *,
    client: AuthenticatedClient | Client,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | InboxItemResponse | None:
    """Get Inbox Item

    Args:
        item_id (int):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | InboxItemResponse
    """

    return (
        await asyncio_detailed(
            item_id=item_id,
            client=client,
            authorization=authorization,
        )
    ).parsed
