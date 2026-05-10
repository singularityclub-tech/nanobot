from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define

if TYPE_CHECKING:
    from ..models.manager_outbox_item_response_from_address_type_0 import ManagerOutboxItemResponseFromAddressType0
    from ..models.manager_outbox_item_response_payload import ManagerOutboxItemResponsePayload
    from ..models.manager_outbox_item_response_reply_to_address_type_0 import (
        ManagerOutboxItemResponseReplyToAddressType0,
    )


T = TypeVar("T", bound="ManagerOutboxItemResponse")


@_attrs_define
class ManagerOutboxItemResponse:
    """
    Attributes:
        id (int):
        type_ (str):
        state (str):
        due_at (str):
        expires_at (None | str):
        payload (ManagerOutboxItemResponsePayload):
        from_address (ManagerOutboxItemResponseFromAddressType0 | None):
        reply_to_address (ManagerOutboxItemResponseReplyToAddressType0 | None):
        created_at (str):
    """

    id: int
    type_: str
    state: str
    due_at: str
    expires_at: None | str
    payload: ManagerOutboxItemResponsePayload
    from_address: ManagerOutboxItemResponseFromAddressType0 | None
    reply_to_address: ManagerOutboxItemResponseReplyToAddressType0 | None
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        from ..models.manager_outbox_item_response_from_address_type_0 import ManagerOutboxItemResponseFromAddressType0
        from ..models.manager_outbox_item_response_reply_to_address_type_0 import (
            ManagerOutboxItemResponseReplyToAddressType0,
        )

        id = self.id

        type_ = self.type_

        state = self.state

        due_at = self.due_at

        expires_at: None | str
        expires_at = self.expires_at

        payload = self.payload.to_dict()

        from_address: dict[str, Any] | None
        if isinstance(self.from_address, ManagerOutboxItemResponseFromAddressType0):
            from_address = self.from_address.to_dict()
        else:
            from_address = self.from_address

        reply_to_address: dict[str, Any] | None
        if isinstance(self.reply_to_address, ManagerOutboxItemResponseReplyToAddressType0):
            reply_to_address = self.reply_to_address.to_dict()
        else:
            reply_to_address = self.reply_to_address

        created_at = self.created_at

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "id": id,
                "type": type_,
                "state": state,
                "due_at": due_at,
                "expires_at": expires_at,
                "payload": payload,
                "from_address": from_address,
                "reply_to_address": reply_to_address,
                "created_at": created_at,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.manager_outbox_item_response_from_address_type_0 import ManagerOutboxItemResponseFromAddressType0
        from ..models.manager_outbox_item_response_payload import ManagerOutboxItemResponsePayload
        from ..models.manager_outbox_item_response_reply_to_address_type_0 import (
            ManagerOutboxItemResponseReplyToAddressType0,
        )

        d = dict(src_dict)
        id = d.pop("id")

        type_ = d.pop("type")

        state = d.pop("state")

        due_at = d.pop("due_at")

        def _parse_expires_at(data: object) -> None | str:
            if data is None:
                return data
            return cast(None | str, data)

        expires_at = _parse_expires_at(d.pop("expires_at"))

        payload = ManagerOutboxItemResponsePayload.from_dict(d.pop("payload"))

        def _parse_from_address(data: object) -> ManagerOutboxItemResponseFromAddressType0 | None:
            if data is None:
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                from_address_type_0 = ManagerOutboxItemResponseFromAddressType0.from_dict(data)

                return from_address_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(ManagerOutboxItemResponseFromAddressType0 | None, data)

        from_address = _parse_from_address(d.pop("from_address"))

        def _parse_reply_to_address(data: object) -> ManagerOutboxItemResponseReplyToAddressType0 | None:
            if data is None:
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                reply_to_address_type_0 = ManagerOutboxItemResponseReplyToAddressType0.from_dict(data)

                return reply_to_address_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(ManagerOutboxItemResponseReplyToAddressType0 | None, data)

        reply_to_address = _parse_reply_to_address(d.pop("reply_to_address"))

        created_at = d.pop("created_at")

        manager_outbox_item_response = cls(
            id=id,
            type_=type_,
            state=state,
            due_at=due_at,
            expires_at=expires_at,
            payload=payload,
            from_address=from_address,
            reply_to_address=reply_to_address,
            created_at=created_at,
        )

        return manager_outbox_item_response
