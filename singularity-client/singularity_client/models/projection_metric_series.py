from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.projection_series_point import ProjectionSeriesPoint


T = TypeVar("T", bound="ProjectionMetricSeries")


@_attrs_define
class ProjectionMetricSeries:
    """
    Attributes:
        points (list[ProjectionSeriesPoint] | Unset):
    """

    points: list[ProjectionSeriesPoint] | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        points: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.points, Unset):
            points = []
            for points_item_data in self.points:
                points_item = points_item_data.to_dict()
                points.append(points_item)

        field_dict: dict[str, Any] = {}

        field_dict.update({})
        if points is not UNSET:
            field_dict["points"] = points

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.projection_series_point import ProjectionSeriesPoint

        d = dict(src_dict)
        _points = d.pop("points", UNSET)
        points: list[ProjectionSeriesPoint] | Unset = UNSET
        if _points is not UNSET:
            points = []
            for points_item_data in _points:
                points_item = ProjectionSeriesPoint.from_dict(points_item_data)

                points.append(points_item)

        projection_metric_series = cls(
            points=points,
        )

        return projection_metric_series
