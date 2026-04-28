from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from dateutil.parser import isoparse

from ..models.observation_kind import ObservationKind
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.observation_write_request_metadata_type_0 import ObservationWriteRequestMetadataType0


T = TypeVar("T", bound="ObservationWriteRequest")


@_attrs_define
class ObservationWriteRequest:
    """
    Attributes:
        measurement_code (str):
        kind (ObservationKind):
        value (bool | float | int | str):
        observed_at (datetime.datetime):
        recorded_at (datetime.datetime | None | Unset):
        metadata (None | ObservationWriteRequestMetadataType0 | Unset):
    """

    measurement_code: str
    kind: ObservationKind
    value: bool | float | int | str
    observed_at: datetime.datetime
    recorded_at: datetime.datetime | None | Unset = UNSET
    metadata: None | ObservationWriteRequestMetadataType0 | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        from ..models.observation_write_request_metadata_type_0 import ObservationWriteRequestMetadataType0

        measurement_code = self.measurement_code

        kind = self.kind.value

        value: bool | float | int | str
        value = self.value

        observed_at = self.observed_at.isoformat()

        recorded_at: None | str | Unset
        if isinstance(self.recorded_at, Unset):
            recorded_at = UNSET
        elif isinstance(self.recorded_at, datetime.datetime):
            recorded_at = self.recorded_at.isoformat()
        else:
            recorded_at = self.recorded_at

        metadata: dict[str, Any] | None | Unset
        if isinstance(self.metadata, Unset):
            metadata = UNSET
        elif isinstance(self.metadata, ObservationWriteRequestMetadataType0):
            metadata = self.metadata.to_dict()
        else:
            metadata = self.metadata

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "measurement_code": measurement_code,
                "kind": kind,
                "value": value,
                "observed_at": observed_at,
            }
        )
        if recorded_at is not UNSET:
            field_dict["recorded_at"] = recorded_at
        if metadata is not UNSET:
            field_dict["metadata"] = metadata

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.observation_write_request_metadata_type_0 import ObservationWriteRequestMetadataType0

        d = dict(src_dict)
        measurement_code = d.pop("measurement_code")

        kind = ObservationKind(d.pop("kind"))

        def _parse_value(data: object) -> bool | float | int | str:
            return cast(bool | float | int | str, data)

        value = _parse_value(d.pop("value"))

        observed_at = isoparse(d.pop("observed_at"))

        def _parse_recorded_at(data: object) -> datetime.datetime | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                recorded_at_type_0 = isoparse(data)

                return recorded_at_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(datetime.datetime | None | Unset, data)

        recorded_at = _parse_recorded_at(d.pop("recorded_at", UNSET))

        def _parse_metadata(data: object) -> None | ObservationWriteRequestMetadataType0 | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                metadata_type_0 = ObservationWriteRequestMetadataType0.from_dict(data)

                return metadata_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | ObservationWriteRequestMetadataType0 | Unset, data)

        metadata = _parse_metadata(d.pop("metadata", UNSET))

        observation_write_request = cls(
            measurement_code=measurement_code,
            kind=kind,
            value=value,
            observed_at=observed_at,
            recorded_at=recorded_at,
            metadata=metadata,
        )

        return observation_write_request
