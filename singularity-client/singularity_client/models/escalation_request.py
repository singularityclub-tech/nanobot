from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define

T = TypeVar("T", bound="EscalationRequest")


@_attrs_define
class EscalationRequest:
    """
    Attributes:
        reason (str):
        summary (str):
    """

    reason: str
    summary: str

    def to_dict(self) -> dict[str, Any]:
        reason = self.reason

        summary = self.summary

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "reason": reason,
                "summary": summary,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        reason = d.pop("reason")

        summary = d.pop("summary")

        escalation_request = cls(
            reason=reason,
            summary=summary,
        )

        return escalation_request
