from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..types import UNSET, Unset

T = TypeVar("T", bound="InboxAttemptSummary")


@_attrs_define
class InboxAttemptSummary:
    """
    Attributes:
        id (int):
        status (str):
        created_at (datetime.datetime):
        return_reason (None | str | Unset):
    """

    id: int
    status: str
    created_at: datetime.datetime
    return_reason: None | str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        id = self.id

        status = self.status

        created_at = self.created_at.isoformat()

        return_reason: None | str | Unset
        if isinstance(self.return_reason, Unset):
            return_reason = UNSET
        else:
            return_reason = self.return_reason

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "status": status,
                "created_at": created_at,
            }
        )
        if return_reason is not UNSET:
            field_dict["return_reason"] = return_reason

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        id = d.pop("id")

        status = d.pop("status")

        created_at = isoparse(d.pop("created_at"))

        def _parse_return_reason(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        return_reason = _parse_return_reason(d.pop("return_reason", UNSET))

        inbox_attempt_summary = cls(
            id=id,
            status=status,
            created_at=created_at,
            return_reason=return_reason,
        )

        inbox_attempt_summary.additional_properties = d
        return inbox_attempt_summary

    @property
    def additional_keys(self) -> list[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> Any:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
