from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define

from ..models.user_decision_request_decision import UserDecisionRequestDecision
from ..types import UNSET, Unset

T = TypeVar("T", bound="UserDecisionRequest")


@_attrs_define
class UserDecisionRequest:
    """
    Attributes:
        outbox_item_id (int):
        decision (UserDecisionRequestDecision):
        remark (None | str | Unset):
    """

    outbox_item_id: int
    decision: UserDecisionRequestDecision
    remark: None | str | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        outbox_item_id = self.outbox_item_id

        decision = self.decision.value

        remark: None | str | Unset
        if isinstance(self.remark, Unset):
            remark = UNSET
        else:
            remark = self.remark

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "outbox_item_id": outbox_item_id,
                "decision": decision,
            }
        )
        if remark is not UNSET:
            field_dict["remark"] = remark

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        outbox_item_id = d.pop("outbox_item_id")

        decision = UserDecisionRequestDecision(d.pop("decision"))

        def _parse_remark(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        remark = _parse_remark(d.pop("remark", UNSET))

        user_decision_request = cls(
            outbox_item_id=outbox_item_id,
            decision=decision,
            remark=remark,
        )

        return user_decision_request
