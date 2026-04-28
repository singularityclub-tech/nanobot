from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define

T = TypeVar("T", bound="ResolveChannelRequest")


@_attrs_define
class ResolveChannelRequest:
    """Request to resolve a channel ID to a user ID.

    Attributes:
        channel_id (str): Channel identifier in format {channel_type}:{user_identifier}
    """

    channel_id: str

    def to_dict(self) -> dict[str, Any]:
        channel_id = self.channel_id

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "channel_id": channel_id,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        channel_id = d.pop("channel_id")

        resolve_channel_request = cls(
            channel_id=channel_id,
        )

        return resolve_channel_request
