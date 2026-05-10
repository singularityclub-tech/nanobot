from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define

from ..types import UNSET, Unset

T = TypeVar("T", bound="ActorTokenRequest")


@_attrs_define
class ActorTokenRequest:
    """
    Attributes:
        service_secret (str):
        service (str):
        user_id (int):
        scopes (list[str] | Unset):
        trace_id (None | str | Unset):
        session_id (None | str | Unset):
    """

    service_secret: str
    service: str
    user_id: int
    scopes: list[str] | Unset = UNSET
    trace_id: None | str | Unset = UNSET
    session_id: None | str | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        service_secret = self.service_secret

        service = self.service

        user_id = self.user_id

        scopes: list[str] | Unset = UNSET
        if not isinstance(self.scopes, Unset):
            scopes = self.scopes

        trace_id: None | str | Unset
        if isinstance(self.trace_id, Unset):
            trace_id = UNSET
        else:
            trace_id = self.trace_id

        session_id: None | str | Unset
        if isinstance(self.session_id, Unset):
            session_id = UNSET
        else:
            session_id = self.session_id

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "service_secret": service_secret,
                "service": service,
                "user_id": user_id,
            }
        )
        if scopes is not UNSET:
            field_dict["scopes"] = scopes
        if trace_id is not UNSET:
            field_dict["trace_id"] = trace_id
        if session_id is not UNSET:
            field_dict["session_id"] = session_id

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        service_secret = d.pop("service_secret")

        service = d.pop("service")

        user_id = d.pop("user_id")

        scopes = cast(list[str], d.pop("scopes", UNSET))

        def _parse_trace_id(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        trace_id = _parse_trace_id(d.pop("trace_id", UNSET))

        def _parse_session_id(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        session_id = _parse_session_id(d.pop("session_id", UNSET))

        actor_token_request = cls(
            service_secret=service_secret,
            service=service,
            user_id=user_id,
            scopes=scopes,
            trace_id=trace_id,
            session_id=session_id,
        )

        return actor_token_request
