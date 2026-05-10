from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define

from ..types import UNSET, Unset

T = TypeVar("T", bound="ManagerRunSensemakingRequest")


@_attrs_define
class ManagerRunSensemakingRequest:
    """
    Attributes:
        context_hint (None | str | Unset):
        timezone (str | Unset):  Default: 'UTC'.
    """

    context_hint: None | str | Unset = UNSET
    timezone: str | Unset = "UTC"

    def to_dict(self) -> dict[str, Any]:
        context_hint: None | str | Unset
        if isinstance(self.context_hint, Unset):
            context_hint = UNSET
        else:
            context_hint = self.context_hint

        timezone = self.timezone

        field_dict: dict[str, Any] = {}

        field_dict.update({})
        if context_hint is not UNSET:
            field_dict["context_hint"] = context_hint
        if timezone is not UNSET:
            field_dict["timezone"] = timezone

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

        timezone = d.pop("timezone", UNSET)

        manager_run_sensemaking_request = cls(
            context_hint=context_hint,
            timezone=timezone,
        )

        return manager_run_sensemaking_request
