from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define

from ..types import UNSET, Unset

T = TypeVar("T", bound="ClaimRequest")


@_attrs_define
class ClaimRequest:
    """
    Attributes:
        claim_note (None | str | Unset):
    """

    claim_note: None | str | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        claim_note: None | str | Unset
        if isinstance(self.claim_note, Unset):
            claim_note = UNSET
        else:
            claim_note = self.claim_note

        field_dict: dict[str, Any] = {}

        field_dict.update({})
        if claim_note is not UNSET:
            field_dict["claim_note"] = claim_note

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)

        def _parse_claim_note(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        claim_note = _parse_claim_note(d.pop("claim_note", UNSET))

        claim_request = cls(
            claim_note=claim_note,
        )

        return claim_request
