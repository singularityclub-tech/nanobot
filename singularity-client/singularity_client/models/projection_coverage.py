from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define

T = TypeVar("T", bound="ProjectionCoverage")


@_attrs_define
class ProjectionCoverage:
    """
    Attributes:
        observed_metrics (int):
        total_metrics (int):
    """

    observed_metrics: int
    total_metrics: int

    def to_dict(self) -> dict[str, Any]:
        observed_metrics = self.observed_metrics

        total_metrics = self.total_metrics

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "observed_metrics": observed_metrics,
                "total_metrics": total_metrics,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        observed_metrics = d.pop("observed_metrics")

        total_metrics = d.pop("total_metrics")

        projection_coverage = cls(
            observed_metrics=observed_metrics,
            total_metrics=total_metrics,
        )

        return projection_coverage
