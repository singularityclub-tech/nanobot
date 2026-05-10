from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define

from ..types import UNSET, Unset

T = TypeVar("T", bound="ManagerRunSensemakingResponse")


@_attrs_define
class ManagerRunSensemakingResponse:
    """
    Attributes:
        produced_entries (int):
        iterations (int):
        narrative_preview (None | str | Unset):
    """

    produced_entries: int
    iterations: int
    narrative_preview: None | str | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        produced_entries = self.produced_entries

        iterations = self.iterations

        narrative_preview: None | str | Unset
        if isinstance(self.narrative_preview, Unset):
            narrative_preview = UNSET
        else:
            narrative_preview = self.narrative_preview

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "produced_entries": produced_entries,
                "iterations": iterations,
            }
        )
        if narrative_preview is not UNSET:
            field_dict["narrative_preview"] = narrative_preview

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        produced_entries = d.pop("produced_entries")

        iterations = d.pop("iterations")

        def _parse_narrative_preview(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        narrative_preview = _parse_narrative_preview(d.pop("narrative_preview", UNSET))

        manager_run_sensemaking_response = cls(
            produced_entries=produced_entries,
            iterations=iterations,
            narrative_preview=narrative_preview,
        )

        return manager_run_sensemaking_response
