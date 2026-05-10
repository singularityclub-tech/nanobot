from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define

from ..types import UNSET, Unset

T = TypeVar("T", bound="ManagerReviewNoteRequest")


@_attrs_define
class ManagerReviewNoteRequest:
    """
    Attributes:
        note (str):
        related_entry_id (int | None | Unset):
    """

    note: str
    related_entry_id: int | None | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        note = self.note

        related_entry_id: int | None | Unset
        if isinstance(self.related_entry_id, Unset):
            related_entry_id = UNSET
        else:
            related_entry_id = self.related_entry_id

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "note": note,
            }
        )
        if related_entry_id is not UNSET:
            field_dict["related_entry_id"] = related_entry_id

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        note = d.pop("note")

        def _parse_related_entry_id(data: object) -> int | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(int | None | Unset, data)

        related_entry_id = _parse_related_entry_id(d.pop("related_entry_id", UNSET))

        manager_review_note_request = cls(
            note=note,
            related_entry_id=related_entry_id,
        )

        return manager_review_note_request
