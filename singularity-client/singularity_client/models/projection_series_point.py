from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from dateutil.parser import isoparse

T = TypeVar("T", bound="ProjectionSeriesPoint")


@_attrs_define
class ProjectionSeriesPoint:
    """
    Attributes:
        date (datetime.date):
        value (float):
    """

    date: datetime.date
    value: float

    def to_dict(self) -> dict[str, Any]:
        date = self.date.isoformat()

        value = self.value

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "date": date,
                "value": value,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        date = isoparse(d.pop("date")).date()

        value = d.pop("value")

        projection_series_point = cls(
            date=date,
            value=value,
        )

        return projection_series_point
