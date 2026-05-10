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
    from ..models.inbox_attempt_summary import InboxAttemptSummary
    from ..models.outbox_item_from_address_type_0 import OutboxItemFromAddressType0
    from ..models.outbox_item_payload import OutboxItemPayload
    from ..models.outbox_item_reply_to_address_type_0 import OutboxItemReplyToAddressType0


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
        from_address (None | OutboxItemFromAddressType0 | Unset):
        reply_to_address (None | OutboxItemReplyToAddressType0 | Unset):
        inbox_attempts (list[InboxAttemptSummary] | Unset):
    """

    id: int
    type_: OutboxType
    state: OutboxState
    due_at: datetime.datetime
    payload: OutboxItemPayload
    expires_at: datetime.datetime | None | Unset = UNSET
    from_address: None | OutboxItemFromAddressType0 | Unset = UNSET
    reply_to_address: None | OutboxItemReplyToAddressType0 | Unset = UNSET
    inbox_attempts: list[InboxAttemptSummary] | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.outbox_item_from_address_type_0 import OutboxItemFromAddressType0
        from ..models.outbox_item_reply_to_address_type_0 import OutboxItemReplyToAddressType0

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

        from_address: dict[str, Any] | None | Unset
        if isinstance(self.from_address, Unset):
            from_address = UNSET
        elif isinstance(self.from_address, OutboxItemFromAddressType0):
            from_address = self.from_address.to_dict()
        else:
            from_address = self.from_address

        reply_to_address: dict[str, Any] | None | Unset
        if isinstance(self.reply_to_address, Unset):
            reply_to_address = UNSET
        elif isinstance(self.reply_to_address, OutboxItemReplyToAddressType0):
            reply_to_address = self.reply_to_address.to_dict()
        else:
            reply_to_address = self.reply_to_address

        inbox_attempts: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.inbox_attempts, Unset):
            inbox_attempts = []
            for inbox_attempts_item_data in self.inbox_attempts:
                inbox_attempts_item = inbox_attempts_item_data.to_dict()
                inbox_attempts.append(inbox_attempts_item)

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
        if from_address is not UNSET:
            field_dict["from_address"] = from_address
        if reply_to_address is not UNSET:
            field_dict["reply_to_address"] = reply_to_address
        if inbox_attempts is not UNSET:
            field_dict["inbox_attempts"] = inbox_attempts

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.inbox_attempt_summary import InboxAttemptSummary
        from ..models.outbox_item_from_address_type_0 import OutboxItemFromAddressType0
        from ..models.outbox_item_payload import OutboxItemPayload
        from ..models.outbox_item_reply_to_address_type_0 import OutboxItemReplyToAddressType0

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

        def _parse_from_address(data: object) -> None | OutboxItemFromAddressType0 | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                from_address_type_0 = OutboxItemFromAddressType0.from_dict(data)

                return from_address_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | OutboxItemFromAddressType0 | Unset, data)

        from_address = _parse_from_address(d.pop("from_address", UNSET))

        def _parse_reply_to_address(data: object) -> None | OutboxItemReplyToAddressType0 | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                reply_to_address_type_0 = OutboxItemReplyToAddressType0.from_dict(data)

                return reply_to_address_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | OutboxItemReplyToAddressType0 | Unset, data)

        reply_to_address = _parse_reply_to_address(d.pop("reply_to_address", UNSET))

        _inbox_attempts = d.pop("inbox_attempts", UNSET)
        inbox_attempts: list[InboxAttemptSummary] | Unset = UNSET
        if _inbox_attempts is not UNSET:
            inbox_attempts = []
            for inbox_attempts_item_data in _inbox_attempts:
                inbox_attempts_item = InboxAttemptSummary.from_dict(inbox_attempts_item_data)

                inbox_attempts.append(inbox_attempts_item)

        outbox_item = cls(
            id=id,
            type_=type_,
            state=state,
            due_at=due_at,
            payload=payload,
            expires_at=expires_at,
            from_address=from_address,
            reply_to_address=reply_to_address,
            inbox_attempts=inbox_attempts,
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
