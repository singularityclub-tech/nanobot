from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define

T = TypeVar("T", bound="TetherChannelRequest")


@_attrs_define
class TetherChannelRequest:
    """Request to tether a channel to a user.

    Attributes:
        channel_type (str): Type of channel (e.g., 'telegram', 'websocket', 'whatsapp')
        channel_user_id (str): Platform-specific user identifier
        user_id (int): Canonical backend user ID
    """

    channel_type: str
    channel_user_id: str
    user_id: int

    def to_dict(self) -> dict[str, Any]:
        channel_type = self.channel_type

        channel_user_id = self.channel_user_id

        user_id = self.user_id

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "channel_type": channel_type,
                "channel_user_id": channel_user_id,
                "user_id": user_id,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        channel_type = d.pop("channel_type")

        channel_user_id = d.pop("channel_user_id")

        user_id = d.pop("user_id")

        tether_channel_request = cls(
            channel_type=channel_type,
            channel_user_id=channel_user_id,
            user_id=user_id,
        )

        return tether_channel_request
