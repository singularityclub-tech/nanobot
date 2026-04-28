from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.observation_kind import ObservationKind

T = TypeVar("T", bound="ObservationWriteResponse")


@_attrs_define
class ObservationWriteResponse:
    """
    Attributes:
        observation_id (int):
        user_id (int):
        measurement_code (str):
        kind (ObservationKind):
    """

    observation_id: int
    user_id: int
    measurement_code: str
    kind: ObservationKind
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        observation_id = self.observation_id

        user_id = self.user_id

        measurement_code = self.measurement_code

        kind = self.kind.value

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "observation_id": observation_id,
                "user_id": user_id,
                "measurement_code": measurement_code,
                "kind": kind,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        observation_id = d.pop("observation_id")

        user_id = d.pop("user_id")

        measurement_code = d.pop("measurement_code")

        kind = ObservationKind(d.pop("kind"))

        observation_write_response = cls(
            observation_id=observation_id,
            user_id=user_id,
            measurement_code=measurement_code,
            kind=kind,
        )

        observation_write_response.additional_properties = d
        return observation_write_response

    @property
    def additional_keys(self) -> list[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> Any:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
