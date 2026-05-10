from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define

if TYPE_CHECKING:
    from ..models.manager_protocol_item_response_kind_specific_metadata import (
        ManagerProtocolItemResponseKindSpecificMetadata,
    )


T = TypeVar("T", bound="ManagerProtocolItemResponse")


@_attrs_define
class ManagerProtocolItemResponse:
    """
    Attributes:
        id (int):
        created_by (str):
        kind (str):
        name (str):
        instructions (None | str):
        kind_specific_metadata (ManagerProtocolItemResponseKindSpecificMetadata):
        status (str):
        start_date (None | str):
        end_date (None | str):
        created_at (str):
    """

    id: int
    created_by: str
    kind: str
    name: str
    instructions: None | str
    kind_specific_metadata: ManagerProtocolItemResponseKindSpecificMetadata
    status: str
    start_date: None | str
    end_date: None | str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        id = self.id

        created_by = self.created_by

        kind = self.kind

        name = self.name

        instructions: None | str
        instructions = self.instructions

        kind_specific_metadata = self.kind_specific_metadata.to_dict()

        status = self.status

        start_date: None | str
        start_date = self.start_date

        end_date: None | str
        end_date = self.end_date

        created_at = self.created_at

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "id": id,
                "created_by": created_by,
                "kind": kind,
                "name": name,
                "instructions": instructions,
                "kind_specific_metadata": kind_specific_metadata,
                "status": status,
                "start_date": start_date,
                "end_date": end_date,
                "created_at": created_at,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.manager_protocol_item_response_kind_specific_metadata import (
            ManagerProtocolItemResponseKindSpecificMetadata,
        )

        d = dict(src_dict)
        id = d.pop("id")

        created_by = d.pop("created_by")

        kind = d.pop("kind")

        name = d.pop("name")

        def _parse_instructions(data: object) -> None | str:
            if data is None:
                return data
            return cast(None | str, data)

        instructions = _parse_instructions(d.pop("instructions"))

        kind_specific_metadata = ManagerProtocolItemResponseKindSpecificMetadata.from_dict(
            d.pop("kind_specific_metadata")
        )

        status = d.pop("status")

        def _parse_start_date(data: object) -> None | str:
            if data is None:
                return data
            return cast(None | str, data)

        start_date = _parse_start_date(d.pop("start_date"))

        def _parse_end_date(data: object) -> None | str:
            if data is None:
                return data
            return cast(None | str, data)

        end_date = _parse_end_date(d.pop("end_date"))

        created_at = d.pop("created_at")

        manager_protocol_item_response = cls(
            id=id,
            created_by=created_by,
            kind=kind,
            name=name,
            instructions=instructions,
            kind_specific_metadata=kind_specific_metadata,
            status=status,
            start_date=start_date,
            end_date=end_date,
            created_at=created_at,
        )

        return manager_protocol_item_response
