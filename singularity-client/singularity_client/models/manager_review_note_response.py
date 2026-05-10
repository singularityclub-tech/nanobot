from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define

T = TypeVar("T", bound="ManagerReviewNoteResponse")


@_attrs_define
class ManagerReviewNoteResponse:
    """
    Attributes:
        id (int):
        created_at (str):
    """

    id: int
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        id = self.id

        created_at = self.created_at

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "id": id,
                "created_at": created_at,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        id = d.pop("id")

        created_at = d.pop("created_at")

        manager_review_note_response = cls(
            id=id,
            created_at=created_at,
        )

        return manager_review_note_response
