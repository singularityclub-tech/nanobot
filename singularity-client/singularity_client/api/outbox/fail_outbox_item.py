from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.fail_request import FailRequest
from ...models.http_validation_error import HTTPValidationError
from ...models.outbox_state_response import OutboxStateResponse
from ...types import UNSET, Response, Unset


def _get_kwargs(
    item_id: int,
    *,
    body: FailRequest,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/internal/mcp/users/me/outbox/{item_id}:fail".format(
            item_id=quote(str(item_id), safe=""),
        ),
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | OutboxStateResponse | None:
    if response.status_code == 200:
        response_200 = OutboxStateResponse.from_dict(response.json())

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
) -> Response[HTTPValidationError | OutboxStateResponse]:
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
    body: FailRequest,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | OutboxStateResponse]:
    """Fail Outbox Item

    Args:
        item_id (int):
        authorization (None | str | Unset):
        body (FailRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | OutboxStateResponse]
    """

    kwargs = _get_kwargs(
        item_id=item_id,
        body=body,
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
    body: FailRequest,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | OutboxStateResponse | None:
    """Fail Outbox Item

    Args:
        item_id (int):
        authorization (None | str | Unset):
        body (FailRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | OutboxStateResponse
    """

    return sync_detailed(
        item_id=item_id,
        client=client,
        body=body,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    item_id: int,
    *,
    client: AuthenticatedClient | Client,
    body: FailRequest,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | OutboxStateResponse]:
    """Fail Outbox Item

    Args:
        item_id (int):
        authorization (None | str | Unset):
        body (FailRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | OutboxStateResponse]
    """

    kwargs = _get_kwargs(
        item_id=item_id,
        body=body,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    item_id: int,
    *,
    client: AuthenticatedClient | Client,
    body: FailRequest,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | OutboxStateResponse | None:
    """Fail Outbox Item

    Args:
        item_id (int):
        authorization (None | str | Unset):
        body (FailRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | OutboxStateResponse
    """

    return (
        await asyncio_detailed(
            item_id=item_id,
            client=client,
            body=body,
            authorization=authorization,
        )
    ).parsed
