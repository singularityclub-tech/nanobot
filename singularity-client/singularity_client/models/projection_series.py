from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define

if TYPE_CHECKING:
    from ..models.projection_series_metrics import ProjectionSeriesMetrics


T = TypeVar("T", bound="ProjectionSeries")


@_attrs_define
class ProjectionSeries:
    """
    Attributes:
        metrics (ProjectionSeriesMetrics):
    """

    metrics: ProjectionSeriesMetrics

    def to_dict(self) -> dict[str, Any]:
        metrics = self.metrics.to_dict()

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "metrics": metrics,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.projection_series_metrics import ProjectionSeriesMetrics

        d = dict(src_dict)
        metrics = ProjectionSeriesMetrics.from_dict(d.pop("metrics"))

        projection_series = cls(
            metrics=metrics,
        )

        return projection_series
