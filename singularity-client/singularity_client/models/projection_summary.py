from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define

if TYPE_CHECKING:
    from ..models.projection_coverage import ProjectionCoverage
    from ..models.projection_summary_metrics import ProjectionSummaryMetrics


T = TypeVar("T", bound="ProjectionSummary")


@_attrs_define
class ProjectionSummary:
    """
    Attributes:
        metrics (ProjectionSummaryMetrics):
        coverage (ProjectionCoverage):
    """

    metrics: ProjectionSummaryMetrics
    coverage: ProjectionCoverage

    def to_dict(self) -> dict[str, Any]:
        metrics = self.metrics.to_dict()

        coverage = self.coverage.to_dict()

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "metrics": metrics,
                "coverage": coverage,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.projection_coverage import ProjectionCoverage
        from ..models.projection_summary_metrics import ProjectionSummaryMetrics

        d = dict(src_dict)
        metrics = ProjectionSummaryMetrics.from_dict(d.pop("metrics"))

        coverage = ProjectionCoverage.from_dict(d.pop("coverage"))

        projection_summary = cls(
            metrics=metrics,
            coverage=coverage,
        )

        return projection_summary
