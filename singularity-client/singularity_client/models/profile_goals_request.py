from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define

T = TypeVar("T", bound="ProfileGoalsRequest")


@_attrs_define
class ProfileGoalsRequest:
    """
    Attributes:
        goals (list[str]):
    """

    goals: list[str]

    def to_dict(self) -> dict[str, Any]:
        goals = self.goals

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "goals": goals,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        goals = cast(list[str], d.pop("goals"))

        profile_goals_request = cls(
            goals=goals,
        )

        return profile_goals_request
