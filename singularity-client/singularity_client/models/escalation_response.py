from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.outbox_state import OutboxState

T = TypeVar("T", bound="EscalationResponse")


@_attrs_define
class EscalationResponse:
    """
    Attributes:
        outbox_item_id (int):
        state (OutboxState):
    """

    outbox_item_id: int
    state: OutboxState
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        outbox_item_id = self.outbox_item_id

        state = self.state.value

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "outbox_item_id": outbox_item_id,
                "state": state,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        outbox_item_id = d.pop("outbox_item_id")

        state = OutboxState(d.pop("state"))

        escalation_response = cls(
            outbox_item_id=outbox_item_id,
            state=state,
        )

        escalation_response.additional_properties = d
        return escalation_response

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
