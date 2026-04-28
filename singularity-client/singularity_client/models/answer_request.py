from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define

from ..types import UNSET, Unset

T = TypeVar("T", bound="AnswerRequest")


@_attrs_define
class AnswerRequest:
    """
    Attributes:
        response_text (None | str | Unset):
    """

    response_text: None | str | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        response_text: None | str | Unset
        if isinstance(self.response_text, Unset):
            response_text = UNSET
        else:
            response_text = self.response_text

        field_dict: dict[str, Any] = {}

        field_dict.update({})
        if response_text is not UNSET:
            field_dict["response_text"] = response_text

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

        answer_request = cls(
            response_text=response_text,
        )

        return answer_request
