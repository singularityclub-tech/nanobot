from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define

if TYPE_CHECKING:
    from ..models.manager_active_log_entry import ManagerActiveLogEntry


T = TypeVar("T", bound="ManagerActiveLogResponse")


@_attrs_define
class ManagerActiveLogResponse:
    """
    Attributes:
        user_id (int):
        entries (list[ManagerActiveLogEntry]):
    """

    user_id: int
    entries: list[ManagerActiveLogEntry]

    def to_dict(self) -> dict[str, Any]:
        user_id = self.user_id

        entries = []
        for entries_item_data in self.entries:
            entries_item = entries_item_data.to_dict()
            entries.append(entries_item)

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "user_id": user_id,
                "entries": entries,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.manager_active_log_entry import ManagerActiveLogEntry

        d = dict(src_dict)
        user_id = d.pop("user_id")

        entries = []
        _entries = d.pop("entries")
        for entries_item_data in _entries:
            entries_item = ManagerActiveLogEntry.from_dict(entries_item_data)

            entries.append(entries_item)

        manager_active_log_response = cls(
            user_id=user_id,
            entries=entries,
        )

        return manager_active_log_response
