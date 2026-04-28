from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define

from ..types import UNSET, Unset

T = TypeVar("T", bound="ExperimentEvaluationRequest")


@_attrs_define
class ExperimentEvaluationRequest:
    """
    Attributes:
        early_stop (bool | Unset):  Default: False.
        reason (None | str | Unset):
    """

    early_stop: bool | Unset = False
    reason: None | str | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        early_stop = self.early_stop

        reason: None | str | Unset
        if isinstance(self.reason, Unset):
            reason = UNSET
        else:
            reason = self.reason

        field_dict: dict[str, Any] = {}

        field_dict.update({})
        if early_stop is not UNSET:
            field_dict["early_stop"] = early_stop
        if reason is not UNSET:
            field_dict["reason"] = reason

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        early_stop = d.pop("early_stop", UNSET)

        def _parse_reason(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        reason = _parse_reason(d.pop("reason", UNSET))

        experiment_evaluation_request = cls(
            early_stop=early_stop,
            reason=reason,
        )

        return experiment_evaluation_request
