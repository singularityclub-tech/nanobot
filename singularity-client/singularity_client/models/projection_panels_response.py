from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define

if TYPE_CHECKING:
    from ..models.projection_panel_response import ProjectionPanelResponse
    from ..models.projection_window import ProjectionWindow


T = TypeVar("T", bound="ProjectionPanelsResponse")


@_attrs_define
class ProjectionPanelsResponse:
    """
    Attributes:
        window (ProjectionWindow):
        panels (list[ProjectionPanelResponse]):
    """

    window: ProjectionWindow
    panels: list[ProjectionPanelResponse]

    def to_dict(self) -> dict[str, Any]:
        window = self.window.to_dict()

        panels = []
        for panels_item_data in self.panels:
            panels_item = panels_item_data.to_dict()
            panels.append(panels_item)

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "window": window,
                "panels": panels,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.projection_panel_response import ProjectionPanelResponse
        from ..models.projection_window import ProjectionWindow

        d = dict(src_dict)
        window = ProjectionWindow.from_dict(d.pop("window"))

        panels = []
        _panels = d.pop("panels")
        for panels_item_data in _panels:
            panels_item = ProjectionPanelResponse.from_dict(panels_item_data)

            panels.append(panels_item)

        projection_panels_response = cls(
            window=window,
            panels=panels,
        )

        return projection_panels_response
