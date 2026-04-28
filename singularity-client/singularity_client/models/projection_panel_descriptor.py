from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define

T = TypeVar("T", bound="ProjectionPanelDescriptor")


@_attrs_define
class ProjectionPanelDescriptor:
    """
    Attributes:
        key (str):
        label (str):
    """

    key: str
    label: str

    def to_dict(self) -> dict[str, Any]:
        key = self.key

        label = self.label

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "key": key,
                "label": label,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        key = d.pop("key")

        label = d.pop("label")

        projection_panel_descriptor = cls(
            key=key,
            label=label,
        )

        return projection_panel_descriptor
