from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define

T = TypeVar("T", bound="ProfileSteeringRequest")


@_attrs_define
class ProfileSteeringRequest:
    """
    Attributes:
        steering (list[str]):
    """

    steering: list[str]

    def to_dict(self) -> dict[str, Any]:
        steering = self.steering

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "steering": steering,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        steering = cast(list[str], d.pop("steering"))

        profile_steering_request = cls(
            steering=steering,
        )

        return profile_steering_request
