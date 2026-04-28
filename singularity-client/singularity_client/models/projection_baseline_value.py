from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define

T = TypeVar("T", bound="ProjectionBaselineValue")


@_attrs_define
class ProjectionBaselineValue:
    """
    Attributes:
        window_days (int):
        value (float):
        n (int):
    """

    window_days: int
    value: float
    n: int

    def to_dict(self) -> dict[str, Any]:
        window_days = self.window_days

        value = self.value

        n = self.n

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "window_days": window_days,
                "value": value,
                "n": n,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        window_days = d.pop("window_days")

        value = d.pop("value")

        n = d.pop("n")

        projection_baseline_value = cls(
            window_days=window_days,
            value=value,
            n=n,
        )

        return projection_baseline_value
