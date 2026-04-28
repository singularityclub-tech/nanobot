from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.projection_metric_descriptor import ProjectionMetricDescriptor
    from ..models.projection_panel_response_derived_type_0 import ProjectionPanelResponseDerivedType0
    from ..models.projection_series import ProjectionSeries
    from ..models.projection_summary import ProjectionSummary
    from ..models.projection_window import ProjectionWindow


T = TypeVar("T", bound="ProjectionPanelResponse")


@_attrs_define
class ProjectionPanelResponse:
    """
    Attributes:
        panel (str):
        label (str):
        window (ProjectionWindow):
        metrics (list[ProjectionMetricDescriptor]):
        summary (ProjectionSummary):
        series (None | ProjectionSeries | Unset):
        derived (None | ProjectionPanelResponseDerivedType0 | Unset):
    """

    panel: str
    label: str
    window: ProjectionWindow
    metrics: list[ProjectionMetricDescriptor]
    summary: ProjectionSummary
    series: None | ProjectionSeries | Unset = UNSET
    derived: None | ProjectionPanelResponseDerivedType0 | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        from ..models.projection_panel_response_derived_type_0 import ProjectionPanelResponseDerivedType0
        from ..models.projection_series import ProjectionSeries

        panel = self.panel

        label = self.label

        window = self.window.to_dict()

        metrics = []
        for metrics_item_data in self.metrics:
            metrics_item = metrics_item_data.to_dict()
            metrics.append(metrics_item)

        summary = self.summary.to_dict()

        series: dict[str, Any] | None | Unset
        if isinstance(self.series, Unset):
            series = UNSET
        elif isinstance(self.series, ProjectionSeries):
            series = self.series.to_dict()
        else:
            series = self.series

        derived: dict[str, Any] | None | Unset
        if isinstance(self.derived, Unset):
            derived = UNSET
        elif isinstance(self.derived, ProjectionPanelResponseDerivedType0):
            derived = self.derived.to_dict()
        else:
            derived = self.derived

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "panel": panel,
                "label": label,
                "window": window,
                "metrics": metrics,
                "summary": summary,
            }
        )
        if series is not UNSET:
            field_dict["series"] = series
        if derived is not UNSET:
            field_dict["derived"] = derived

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.projection_metric_descriptor import ProjectionMetricDescriptor
        from ..models.projection_panel_response_derived_type_0 import ProjectionPanelResponseDerivedType0
        from ..models.projection_series import ProjectionSeries
        from ..models.projection_summary import ProjectionSummary
        from ..models.projection_window import ProjectionWindow

        d = dict(src_dict)
        panel = d.pop("panel")

        label = d.pop("label")

        window = ProjectionWindow.from_dict(d.pop("window"))

        metrics = []
        _metrics = d.pop("metrics")
        for metrics_item_data in _metrics:
            metrics_item = ProjectionMetricDescriptor.from_dict(metrics_item_data)

            metrics.append(metrics_item)

        summary = ProjectionSummary.from_dict(d.pop("summary"))

        def _parse_series(data: object) -> None | ProjectionSeries | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                series_type_0 = ProjectionSeries.from_dict(data)

                return series_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | ProjectionSeries | Unset, data)

        series = _parse_series(d.pop("series", UNSET))

        def _parse_derived(data: object) -> None | ProjectionPanelResponseDerivedType0 | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                derived_type_0 = ProjectionPanelResponseDerivedType0.from_dict(data)

                return derived_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | ProjectionPanelResponseDerivedType0 | Unset, data)

        derived = _parse_derived(d.pop("derived", UNSET))

        projection_panel_response = cls(
            panel=panel,
            label=label,
            window=window,
            metrics=metrics,
            summary=summary,
            series=series,
            derived=derived,
        )

        return projection_panel_response
