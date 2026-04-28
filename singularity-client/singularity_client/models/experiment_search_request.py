from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define

T = TypeVar("T", bound="ExperimentSearchRequest")


@_attrs_define
class ExperimentSearchRequest:
    """
    Attributes:
        goal (str):
    """

    goal: str

    def to_dict(self) -> dict[str, Any]:
        goal = self.goal

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "goal": goal,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        goal = d.pop("goal")

        experiment_search_request = cls(
            goal=goal,
        )

        return experiment_search_request
