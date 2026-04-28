from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.projection_baseline_value import ProjectionBaselineValue
    from ..models.projection_metric_source_value import ProjectionMetricSourceValue
    from ..models.projection_metric_value import ProjectionMetricValue


T = TypeVar("T", bound="ProjectionMetricSummary")


@_attrs_define
class ProjectionMetricSummary:
    """
    Attributes:
        latest (None | ProjectionMetricValue | Unset):
        baseline (None | ProjectionBaselineValue | Unset):
        sources (list[ProjectionMetricSourceValue] | Unset):
    """

    latest: None | ProjectionMetricValue | Unset = UNSET
    baseline: None | ProjectionBaselineValue | Unset = UNSET
    sources: list[ProjectionMetricSourceValue] | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        from ..models.projection_baseline_value import ProjectionBaselineValue
        from ..models.projection_metric_value import ProjectionMetricValue

        latest: dict[str, Any] | None | Unset
        if isinstance(self.latest, Unset):
            latest = UNSET
        elif isinstance(self.latest, ProjectionMetricValue):
            latest = self.latest.to_dict()
        else:
            latest = self.latest

        baseline: dict[str, Any] | None | Unset
        if isinstance(self.baseline, Unset):
            baseline = UNSET
        elif isinstance(self.baseline, ProjectionBaselineValue):
            baseline = self.baseline.to_dict()
        else:
            baseline = self.baseline

        sources: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.sources, Unset):
            sources = []
            for sources_item_data in self.sources:
                sources_item = sources_item_data.to_dict()
                sources.append(sources_item)

        field_dict: dict[str, Any] = {}

        field_dict.update({})
        if latest is not UNSET:
            field_dict["latest"] = latest
        if baseline is not UNSET:
            field_dict["baseline"] = baseline
        if sources is not UNSET:
            field_dict["sources"] = sources

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.projection_baseline_value import ProjectionBaselineValue
        from ..models.projection_metric_source_value import ProjectionMetricSourceValue
        from ..models.projection_metric_value import ProjectionMetricValue

        d = dict(src_dict)

        def _parse_latest(data: object) -> None | ProjectionMetricValue | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                latest_type_0 = ProjectionMetricValue.from_dict(data)

                return latest_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | ProjectionMetricValue | Unset, data)

        latest = _parse_latest(d.pop("latest", UNSET))

        def _parse_baseline(data: object) -> None | ProjectionBaselineValue | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                baseline_type_0 = ProjectionBaselineValue.from_dict(data)

                return baseline_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | ProjectionBaselineValue | Unset, data)

        baseline = _parse_baseline(d.pop("baseline", UNSET))

        _sources = d.pop("sources", UNSET)
        sources: list[ProjectionMetricSourceValue] | Unset = UNSET
        if _sources is not UNSET:
            sources = []
            for sources_item_data in _sources:
                sources_item = ProjectionMetricSourceValue.from_dict(sources_item_data)

                sources.append(sources_item)

        projection_metric_summary = cls(
            latest=latest,
            baseline=baseline,
            sources=sources,
        )

        return projection_metric_summary
