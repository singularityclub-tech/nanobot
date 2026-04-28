from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="TetherChannelResponse")


@_attrs_define
class TetherChannelResponse:
    """Response from tethering a channel.

    Attributes:
        id (int):
        channel_id (str):
        user_id (int):
        channel_type (str):
        channel_user_id (str):
        created_at (str):
        last_seen_at (None | str):
    """

    id: int
    channel_id: str
    user_id: int
    channel_type: str
    channel_user_id: str
    created_at: str
    last_seen_at: None | str
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        id = self.id

        channel_id = self.channel_id

        user_id = self.user_id

        channel_type = self.channel_type

        channel_user_id = self.channel_user_id

        created_at = self.created_at

        last_seen_at: None | str
        last_seen_at = self.last_seen_at

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "channel_id": channel_id,
                "user_id": user_id,
                "channel_type": channel_type,
                "channel_user_id": channel_user_id,
                "created_at": created_at,
                "last_seen_at": last_seen_at,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        id = d.pop("id")

        channel_id = d.pop("channel_id")

        user_id = d.pop("user_id")

        channel_type = d.pop("channel_type")

        channel_user_id = d.pop("channel_user_id")

        created_at = d.pop("created_at")

        def _parse_last_seen_at(data: object) -> None | str:
            if data is None:
                return data
            return cast(None | str, data)

        last_seen_at = _parse_last_seen_at(d.pop("last_seen_at"))

        tether_channel_response = cls(
            id=id,
            channel_id=channel_id,
            user_id=user_id,
            channel_type=channel_type,
            channel_user_id=channel_user_id,
            created_at=created_at,
            last_seen_at=last_seen_at,
        )

        tether_channel_response.additional_properties = d
        return tether_channel_response

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
