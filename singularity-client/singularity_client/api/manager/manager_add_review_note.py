from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.manager_review_note_request import ManagerReviewNoteRequest
from ...models.manager_review_note_response import ManagerReviewNoteResponse
from ...types import UNSET, Response, Unset


def _get_kwargs(
    user_id: int,
    *,
    body: ManagerReviewNoteRequest,
    x_manager_secret: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(x_manager_secret, Unset):
        headers["x-manager-secret"] = x_manager_secret

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/internal/manager/users/{user_id}/active-log/review-note".format(
            user_id=quote(str(user_id), safe=""),
        ),
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | ManagerReviewNoteResponse | None:
    if response.status_code == 200:
        response_200 = ManagerReviewNoteResponse.from_dict(response.json())

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
) -> Response[HTTPValidationError | ManagerReviewNoteResponse]:
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
    body: ManagerReviewNoteRequest,
    x_manager_secret: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | ManagerReviewNoteResponse]:
    """Manager Add Review Note

    Args:
        user_id (int):
        x_manager_secret (None | str | Unset):
        body (ManagerReviewNoteRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | ManagerReviewNoteResponse]
    """

    kwargs = _get_kwargs(
        user_id=user_id,
        body=body,
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
    body: ManagerReviewNoteRequest,
    x_manager_secret: None | str | Unset = UNSET,
) -> HTTPValidationError | ManagerReviewNoteResponse | None:
    """Manager Add Review Note

    Args:
        user_id (int):
        x_manager_secret (None | str | Unset):
        body (ManagerReviewNoteRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | ManagerReviewNoteResponse
    """

    return sync_detailed(
        user_id=user_id,
        client=client,
        body=body,
        x_manager_secret=x_manager_secret,
    ).parsed


async def asyncio_detailed(
    user_id: int,
    *,
    client: AuthenticatedClient | Client,
    body: ManagerReviewNoteRequest,
    x_manager_secret: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | ManagerReviewNoteResponse]:
    """Manager Add Review Note

    Args:
        user_id (int):
        x_manager_secret (None | str | Unset):
        body (ManagerReviewNoteRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | ManagerReviewNoteResponse]
    """

    kwargs = _get_kwargs(
        user_id=user_id,
        body=body,
        x_manager_secret=x_manager_secret,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    user_id: int,
    *,
    client: AuthenticatedClient | Client,
    body: ManagerReviewNoteRequest,
    x_manager_secret: None | str | Unset = UNSET,
) -> HTTPValidationError | ManagerReviewNoteResponse | None:
    """Manager Add Review Note

    Args:
        user_id (int):
        x_manager_secret (None | str | Unset):
        body (ManagerReviewNoteRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | ManagerReviewNoteResponse
    """

    return (
        await asyncio_detailed(
            user_id=user_id,
            client=client,
            body=body,
            x_manager_secret=x_manager_secret,
        )
    ).parsed
