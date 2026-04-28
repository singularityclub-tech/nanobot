from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="PipelineAck")


@_attrs_define
class PipelineAck:
    """
    Attributes:
        pipeline (str):
        active_log_id (int | None | Unset):
        active_log_ids (list[int] | Unset):
        outbox_item_id (int | None | Unset):
        narrator_text (None | str | Unset):
    """

    pipeline: str
    active_log_id: int | None | Unset = UNSET
    active_log_ids: list[int] | Unset = UNSET
    outbox_item_id: int | None | Unset = UNSET
    narrator_text: None | str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        pipeline = self.pipeline

        active_log_id: int | None | Unset
        if isinstance(self.active_log_id, Unset):
            active_log_id = UNSET
        else:
            active_log_id = self.active_log_id

        active_log_ids: list[int] | Unset = UNSET
        if not isinstance(self.active_log_ids, Unset):
            active_log_ids = self.active_log_ids

        outbox_item_id: int | None | Unset
        if isinstance(self.outbox_item_id, Unset):
            outbox_item_id = UNSET
        else:
            outbox_item_id = self.outbox_item_id

        narrator_text: None | str | Unset
        if isinstance(self.narrator_text, Unset):
            narrator_text = UNSET
        else:
            narrator_text = self.narrator_text

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "pipeline": pipeline,
            }
        )
        if active_log_id is not UNSET:
            field_dict["active_log_id"] = active_log_id
        if active_log_ids is not UNSET:
            field_dict["active_log_ids"] = active_log_ids
        if outbox_item_id is not UNSET:
            field_dict["outbox_item_id"] = outbox_item_id
        if narrator_text is not UNSET:
            field_dict["narrator_text"] = narrator_text

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        pipeline = d.pop("pipeline")

        def _parse_active_log_id(data: object) -> int | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(int | None | Unset, data)

        active_log_id = _parse_active_log_id(d.pop("active_log_id", UNSET))

        active_log_ids = cast(list[int], d.pop("active_log_ids", UNSET))

        def _parse_outbox_item_id(data: object) -> int | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(int | None | Unset, data)

        outbox_item_id = _parse_outbox_item_id(d.pop("outbox_item_id", UNSET))

        def _parse_narrator_text(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        narrator_text = _parse_narrator_text(d.pop("narrator_text", UNSET))

        pipeline_ack = cls(
            pipeline=pipeline,
            active_log_id=active_log_id,
            active_log_ids=active_log_ids,
            outbox_item_id=outbox_item_id,
            narrator_text=narrator_text,
        )

        pipeline_ack.additional_properties = d
        return pipeline_ack

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
