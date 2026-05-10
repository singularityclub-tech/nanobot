from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from dateutil.parser import isoparse

from ..types import UNSET, Unset

T = TypeVar("T", bound="AnswerRequest")


@_attrs_define
class AnswerRequest:
    """
    Attributes:
        response_text (None | str | Unset):
        answered_at (datetime.datetime | None | Unset):
    """

    response_text: None | str | Unset = UNSET
    answered_at: datetime.datetime | None | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        response_text: None | str | Unset
        if isinstance(self.response_text, Unset):
            response_text = UNSET
        else:
            response_text = self.response_text

        answered_at: None | str | Unset
        if isinstance(self.answered_at, Unset):
            answered_at = UNSET
        elif isinstance(self.answered_at, datetime.datetime):
            answered_at = self.answered_at.isoformat()
        else:
            answered_at = self.answered_at

        field_dict: dict[str, Any] = {}

        field_dict.update({})
        if response_text is not UNSET:
            field_dict["response_text"] = response_text
        if answered_at is not UNSET:
            field_dict["answered_at"] = answered_at

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)

        def _parse_response_text(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        response_text = _parse_response_text(d.pop("response_text", UNSET))

        def _parse_answered_at(data: object) -> datetime.datetime | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                answered_at_type_0 = isoparse(data)

                return answered_at_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(datetime.datetime | None | Unset, data)

        answered_at = _parse_answered_at(d.pop("answered_at", UNSET))

        answer_request = cls(
            response_text=response_text,
            answered_at=answered_at,
        )

        return answer_request
