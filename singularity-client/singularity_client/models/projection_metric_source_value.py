from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from dateutil.parser import isoparse

from ..types import UNSET, Unset

T = TypeVar("T", bound="ProjectionMetricSourceValue")


@_attrs_define
class ProjectionMetricSourceValue:
    """
    Attributes:
        value (float):
        observed_at (datetime.datetime):
        source_vendor (None | str | Unset):
        source_kind (None | str | Unset):
        observation_id (int | None | Unset):
        observation_set_id (None | str | Unset):
        raw_ref (None | str | Unset):
    """

    value: float
    observed_at: datetime.datetime
    source_vendor: None | str | Unset = UNSET
    source_kind: None | str | Unset = UNSET
    observation_id: int | None | Unset = UNSET
    observation_set_id: None | str | Unset = UNSET
    raw_ref: None | str | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        value = self.value

        observed_at = self.observed_at.isoformat()

        source_vendor: None | str | Unset
        if isinstance(self.source_vendor, Unset):
            source_vendor = UNSET
        else:
            source_vendor = self.source_vendor

        source_kind: None | str | Unset
        if isinstance(self.source_kind, Unset):
            source_kind = UNSET
        else:
            source_kind = self.source_kind

        observation_id: int | None | Unset
        if isinstance(self.observation_id, Unset):
            observation_id = UNSET
        else:
            observation_id = self.observation_id

        observation_set_id: None | str | Unset
        if isinstance(self.observation_set_id, Unset):
            observation_set_id = UNSET
        else:
            observation_set_id = self.observation_set_id

        raw_ref: None | str | Unset
        if isinstance(self.raw_ref, Unset):
            raw_ref = UNSET
        else:
            raw_ref = self.raw_ref

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "value": value,
                "observed_at": observed_at,
            }
        )
        if source_vendor is not UNSET:
            field_dict["source_vendor"] = source_vendor
        if source_kind is not UNSET:
            field_dict["source_kind"] = source_kind
        if observation_id is not UNSET:
            field_dict["observation_id"] = observation_id
        if observation_set_id is not UNSET:
            field_dict["observation_set_id"] = observation_set_id
        if raw_ref is not UNSET:
            field_dict["raw_ref"] = raw_ref

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        value = d.pop("value")

        observed_at = isoparse(d.pop("observed_at"))

        def _parse_source_vendor(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        source_vendor = _parse_source_vendor(d.pop("source_vendor", UNSET))

        def _parse_source_kind(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        source_kind = _parse_source_kind(d.pop("source_kind", UNSET))

        def _parse_observation_id(data: object) -> int | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(int | None | Unset, data)

        observation_id = _parse_observation_id(d.pop("observation_id", UNSET))

        def _parse_observation_set_id(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        observation_set_id = _parse_observation_set_id(d.pop("observation_set_id", UNSET))

        def _parse_raw_ref(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        raw_ref = _parse_raw_ref(d.pop("raw_ref", UNSET))

        projection_metric_source_value = cls(
            value=value,
            observed_at=observed_at,
            source_vendor=source_vendor,
            source_kind=source_kind,
            observation_id=observation_id,
            observation_set_id=observation_set_id,
            raw_ref=raw_ref,
        )

        return projection_metric_source_value
