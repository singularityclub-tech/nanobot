from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define

from ..types import UNSET, Unset

T = TypeVar("T", bound="CheckinResponseRequest")


@_attrs_define
class CheckinResponseRequest:
    """
    Attributes:
        context_hint (None | str | Unset):
    """

    context_hint: None | str | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        context_hint: None | str | Unset
        if isinstance(self.context_hint, Unset):
            context_hint = UNSET
        else:
            context_hint = self.context_hint

        field_dict: dict[str, Any] = {}

        field_dict.update({})
        if context_hint is not UNSET:
            field_dict["context_hint"] = context_hint

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)

        def _parse_context_hint(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        context_hint = _parse_context_hint(d.pop("context_hint", UNSET))

        checkin_response_request = cls(
            context_hint=context_hint,
        )

        return checkin_response_request
