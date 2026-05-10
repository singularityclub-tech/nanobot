from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.send_inbox_item_request_to_address_type_0 import SendInboxItemRequestToAddressType0


T = TypeVar("T", bound="SendInboxItemRequest")


@_attrs_define
class SendInboxItemRequest:
    """
    Attributes:
        content (str):
        re_outbox_item_id (int | None | Unset):
        to_address (None | SendInboxItemRequestToAddressType0 | Unset):
    """

    content: str
    re_outbox_item_id: int | None | Unset = UNSET
    to_address: None | SendInboxItemRequestToAddressType0 | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        from ..models.send_inbox_item_request_to_address_type_0 import SendInboxItemRequestToAddressType0

        content = self.content

        re_outbox_item_id: int | None | Unset
        if isinstance(self.re_outbox_item_id, Unset):
            re_outbox_item_id = UNSET
        else:
            re_outbox_item_id = self.re_outbox_item_id

        to_address: dict[str, Any] | None | Unset
        if isinstance(self.to_address, Unset):
            to_address = UNSET
        elif isinstance(self.to_address, SendInboxItemRequestToAddressType0):
            to_address = self.to_address.to_dict()
        else:
            to_address = self.to_address

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "content": content,
            }
        )
        if re_outbox_item_id is not UNSET:
            field_dict["re_outbox_item_id"] = re_outbox_item_id
        if to_address is not UNSET:
            field_dict["to_address"] = to_address

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.send_inbox_item_request_to_address_type_0 import SendInboxItemRequestToAddressType0

        d = dict(src_dict)
        content = d.pop("content")

        def _parse_re_outbox_item_id(data: object) -> int | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(int | None | Unset, data)

        re_outbox_item_id = _parse_re_outbox_item_id(d.pop("re_outbox_item_id", UNSET))

        def _parse_to_address(data: object) -> None | SendInboxItemRequestToAddressType0 | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                to_address_type_0 = SendInboxItemRequestToAddressType0.from_dict(data)

                return to_address_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | SendInboxItemRequestToAddressType0 | Unset, data)

        to_address = _parse_to_address(d.pop("to_address", UNSET))

        send_inbox_item_request = cls(
            content=content,
            re_outbox_item_id=re_outbox_item_id,
            to_address=to_address,
        )

        return send_inbox_item_request
