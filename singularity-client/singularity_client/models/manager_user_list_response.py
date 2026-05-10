from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define

if TYPE_CHECKING:
    from ..models.manager_user_summary import ManagerUserSummary


T = TypeVar("T", bound="ManagerUserListResponse")


@_attrs_define
class ManagerUserListResponse:
    """
    Attributes:
        users (list[ManagerUserSummary]):
    """

    users: list[ManagerUserSummary]

    def to_dict(self) -> dict[str, Any]:
        users = []
        for users_item_data in self.users:
            users_item = users_item_data.to_dict()
            users.append(users_item)

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "users": users,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.manager_user_summary import ManagerUserSummary

        d = dict(src_dict)
        users = []
        _users = d.pop("users")
        for users_item_data in _users:
            users_item = ManagerUserSummary.from_dict(users_item_data)

            users.append(users_item)

        manager_user_list_response = cls(
            users=users,
        )

        return manager_user_list_response
