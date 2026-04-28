from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define

from ..models.projection_metric_descriptor_aggregation import ProjectionMetricDescriptorAggregation

T = TypeVar("T", bound="ProjectionMetricDescriptor")


@_attrs_define
class ProjectionMetricDescriptor:
    """
    Attributes:
        key (str):
        code (str):
        label (str):
        aggregation (ProjectionMetricDescriptorAggregation):
        baseline_days (int):
    """

    key: str
    code: str
    label: str
    aggregation: ProjectionMetricDescriptorAggregation
    baseline_days: int

    def to_dict(self) -> dict[str, Any]:
        key = self.key

        code = self.code

        label = self.label

        aggregation = self.aggregation.value

        baseline_days = self.baseline_days

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "key": key,
                "code": code,
                "label": label,
                "aggregation": aggregation,
                "baseline_days": baseline_days,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        key = d.pop("key")

        code = d.pop("code")

        label = d.pop("label")

        aggregation = ProjectionMetricDescriptorAggregation(d.pop("aggregation"))

        baseline_days = d.pop("baseline_days")

        projection_metric_descriptor = cls(
            key=key,
            code=code,
            label=label,
            aggregation=aggregation,
            baseline_days=baseline_days,
        )

        return projection_metric_descriptor
