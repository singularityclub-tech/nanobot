from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define

if TYPE_CHECKING:
    from ..models.manager_active_log_entry_details_type_0 import ManagerActiveLogEntryDetailsType0
    from ..models.manager_active_log_entry_payload import ManagerActiveLogEntryPayload


T = TypeVar("T", bound="ManagerActiveLogEntry")


@_attrs_define
class ManagerActiveLogEntry:
    """
    Attributes:
        id (int):
        user_id (int):
        kind (str):
        record (str):
        details (ManagerActiveLogEntryDetailsType0 | None):
        payload (ManagerActiveLogEntryPayload):
        created_at (str):
    """

    id: int
    user_id: int
    kind: str
    record: str
    details: ManagerActiveLogEntryDetailsType0 | None
    payload: ManagerActiveLogEntryPayload
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        from ..models.manager_active_log_entry_details_type_0 import ManagerActiveLogEntryDetailsType0

        id = self.id

        user_id = self.user_id

        kind = self.kind

        record = self.record

        details: dict[str, Any] | None
        if isinstance(self.details, ManagerActiveLogEntryDetailsType0):
            details = self.details.to_dict()
        else:
            details = self.details

        payload = self.payload.to_dict()

        created_at = self.created_at

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "id": id,
                "user_id": user_id,
                "kind": kind,
                "record": record,
                "details": details,
                "payload": payload,
                "created_at": created_at,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.manager_active_log_entry_details_type_0 import ManagerActiveLogEntryDetailsType0
        from ..models.manager_active_log_entry_payload import ManagerActiveLogEntryPayload

        d = dict(src_dict)
        id = d.pop("id")

        user_id = d.pop("user_id")

        kind = d.pop("kind")

        record = d.pop("record")

        def _parse_details(data: object) -> ManagerActiveLogEntryDetailsType0 | None:
            if data is None:
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                details_type_0 = ManagerActiveLogEntryDetailsType0.from_dict(data)

                return details_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(ManagerActiveLogEntryDetailsType0 | None, data)

        details = _parse_details(d.pop("details"))

        payload = ManagerActiveLogEntryPayload.from_dict(d.pop("payload"))

        created_at = d.pop("created_at")

        manager_active_log_entry = cls(
            id=id,
            user_id=user_id,
            kind=kind,
            record=record,
            details=details,
            payload=payload,
            created_at=created_at,
        )

        return manager_active_log_entry
