from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.outbox_state import OutboxState
from ..types import UNSET, Unset

T = TypeVar("T", bound="OutboxStateResponse")


@_attrs_define
class OutboxStateResponse:
    """
    Attributes:
        id (int):
        state (OutboxState):
        active_log_id (int | None | Unset):
        active_log_ids (list[int] | Unset):
        raw_asset_id (int | None | Unset):
    """

    id: int
    state: OutboxState
    active_log_id: int | None | Unset = UNSET
    active_log_ids: list[int] | Unset = UNSET
    raw_asset_id: int | None | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        id = self.id

        state = self.state.value

        active_log_id: int | None | Unset
        if isinstance(self.active_log_id, Unset):
            active_log_id = UNSET
        else:
            active_log_id = self.active_log_id

        active_log_ids: list[int] | Unset = UNSET
        if not isinstance(self.active_log_ids, Unset):
            active_log_ids = self.active_log_ids

        raw_asset_id: int | None | Unset
        if isinstance(self.raw_asset_id, Unset):
            raw_asset_id = UNSET
        else:
            raw_asset_id = self.raw_asset_id

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "state": state,
            }
        )
        if active_log_id is not UNSET:
            field_dict["active_log_id"] = active_log_id
        if active_log_ids is not UNSET:
            field_dict["active_log_ids"] = active_log_ids
        if raw_asset_id is not UNSET:
            field_dict["raw_asset_id"] = raw_asset_id

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        id = d.pop("id")

        state = OutboxState(d.pop("state"))

        def _parse_active_log_id(data: object) -> int | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(int | None | Unset, data)

        active_log_id = _parse_active_log_id(d.pop("active_log_id", UNSET))

        active_log_ids = cast(list[int], d.pop("active_log_ids", UNSET))

        def _parse_raw_asset_id(data: object) -> int | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(int | None | Unset, data)

        raw_asset_id = _parse_raw_asset_id(d.pop("raw_asset_id", UNSET))

        outbox_state_response = cls(
            id=id,
            state=state,
            active_log_id=active_log_id,
            active_log_ids=active_log_ids,
            raw_asset_id=raw_asset_id,
        )

        outbox_state_response.additional_properties = d
        return outbox_state_response

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
