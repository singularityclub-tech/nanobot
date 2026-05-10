from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.inbox_item_response_result_type_0 import InboxItemResponseResultType0


T = TypeVar("T", bound="InboxItemResponse")


@_attrs_define
class InboxItemResponse:
    """
    Attributes:
        id (int):
        status (str):
        created_at (datetime.datetime):
        return_reason (None | str | Unset):
        result (InboxItemResponseResultType0 | None | Unset):
        processed_at (datetime.datetime | None | Unset):
    """

    id: int
    status: str
    created_at: datetime.datetime
    return_reason: None | str | Unset = UNSET
    result: InboxItemResponseResultType0 | None | Unset = UNSET
    processed_at: datetime.datetime | None | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.inbox_item_response_result_type_0 import InboxItemResponseResultType0

        id = self.id

        status = self.status

        created_at = self.created_at.isoformat()

        return_reason: None | str | Unset
        if isinstance(self.return_reason, Unset):
            return_reason = UNSET
        else:
            return_reason = self.return_reason

        result: dict[str, Any] | None | Unset
        if isinstance(self.result, Unset):
            result = UNSET
        elif isinstance(self.result, InboxItemResponseResultType0):
            result = self.result.to_dict()
        else:
            result = self.result

        processed_at: None | str | Unset
        if isinstance(self.processed_at, Unset):
            processed_at = UNSET
        elif isinstance(self.processed_at, datetime.datetime):
            processed_at = self.processed_at.isoformat()
        else:
            processed_at = self.processed_at

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "status": status,
                "created_at": created_at,
            }
        )
        if return_reason is not UNSET:
            field_dict["return_reason"] = return_reason
        if result is not UNSET:
            field_dict["result"] = result
        if processed_at is not UNSET:
            field_dict["processed_at"] = processed_at

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.inbox_item_response_result_type_0 import InboxItemResponseResultType0

        d = dict(src_dict)
        id = d.pop("id")

        status = d.pop("status")

        created_at = isoparse(d.pop("created_at"))

        def _parse_return_reason(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        return_reason = _parse_return_reason(d.pop("return_reason", UNSET))

        def _parse_result(data: object) -> InboxItemResponseResultType0 | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                result_type_0 = InboxItemResponseResultType0.from_dict(data)

                return result_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(InboxItemResponseResultType0 | None | Unset, data)

        result = _parse_result(d.pop("result", UNSET))

        def _parse_processed_at(data: object) -> datetime.datetime | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                processed_at_type_0 = isoparse(data)

                return processed_at_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(datetime.datetime | None | Unset, data)

        processed_at = _parse_processed_at(d.pop("processed_at", UNSET))

        inbox_item_response = cls(
            id=id,
            status=status,
            created_at=created_at,
            return_reason=return_reason,
            result=result,
            processed_at=processed_at,
        )

        inbox_item_response.additional_properties = d
        return inbox_item_response

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
