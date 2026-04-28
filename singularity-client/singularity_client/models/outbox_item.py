from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..models.outbox_state import OutboxState
from ..models.outbox_type import OutboxType
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.outbox_item_payload import OutboxItemPayload


T = TypeVar("T", bound="OutboxItem")


@_attrs_define
class OutboxItem:
    """
    Attributes:
        id (int):
        type_ (OutboxType):
        state (OutboxState):
        due_at (datetime.datetime):
        payload (OutboxItemPayload):
        expires_at (datetime.datetime | None | Unset):
    """

    id: int
    type_: OutboxType
    state: OutboxState
    due_at: datetime.datetime
    payload: OutboxItemPayload
    expires_at: datetime.datetime | None | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        id = self.id

        type_ = self.type_.value

        state = self.state.value

        due_at = self.due_at.isoformat()

        payload = self.payload.to_dict()

        expires_at: None | str | Unset
        if isinstance(self.expires_at, Unset):
            expires_at = UNSET
        elif isinstance(self.expires_at, datetime.datetime):
            expires_at = self.expires_at.isoformat()
        else:
            expires_at = self.expires_at

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "type": type_,
                "state": state,
                "due_at": due_at,
                "payload": payload,
            }
        )
        if expires_at is not UNSET:
            field_dict["expires_at"] = expires_at

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.outbox_item_payload import OutboxItemPayload

        d = dict(src_dict)
        id = d.pop("id")

        type_ = OutboxType(d.pop("type"))

        state = OutboxState(d.pop("state"))

        due_at = isoparse(d.pop("due_at"))

        payload = OutboxItemPayload.from_dict(d.pop("payload"))

        def _parse_expires_at(data: object) -> datetime.datetime | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                expires_at_type_0 = isoparse(data)

                return expires_at_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(datetime.datetime | None | Unset, data)

        expires_at = _parse_expires_at(d.pop("expires_at", UNSET))

        outbox_item = cls(
            id=id,
            type_=type_,
            state=state,
            due_at=due_at,
            payload=payload,
            expires_at=expires_at,
        )

        outbox_item.additional_properties = d
        return outbox_item

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
