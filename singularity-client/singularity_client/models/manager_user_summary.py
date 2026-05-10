from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define

if TYPE_CHECKING:
    from ..models.manager_user_summary_profile import ManagerUserSummaryProfile


T = TypeVar("T", bound="ManagerUserSummary")


@_attrs_define
class ManagerUserSummary:
    """
    Attributes:
        user_id (int):
        created_at (str):
        profile (ManagerUserSummaryProfile):
    """

    user_id: int
    created_at: str
    profile: ManagerUserSummaryProfile

    def to_dict(self) -> dict[str, Any]:
        user_id = self.user_id

        created_at = self.created_at

        profile = self.profile.to_dict()

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "user_id": user_id,
                "created_at": created_at,
                "profile": profile,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.manager_user_summary_profile import ManagerUserSummaryProfile

        d = dict(src_dict)
        user_id = d.pop("user_id")

        created_at = d.pop("created_at")

        profile = ManagerUserSummaryProfile.from_dict(d.pop("profile"))

        manager_user_summary = cls(
            user_id=user_id,
            created_at=created_at,
            profile=profile,
        )

        return manager_user_summary
