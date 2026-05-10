from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define

if TYPE_CHECKING:
    from ..models.manager_inbox_item_response_from_address import ManagerInboxItemResponseFromAddress
    from ..models.manager_inbox_item_response_result_type_0 import ManagerInboxItemResponseResultType0
    from ..models.manager_inbox_item_response_to_address import ManagerInboxItemResponseToAddress


T = TypeVar("T", bound="ManagerInboxItemResponse")


@_attrs_define
class ManagerInboxItemResponse:
    """
    Attributes:
        id (int):
        from_address (ManagerInboxItemResponseFromAddress):
        to_address (ManagerInboxItemResponseToAddress):
        re_outbox_item_id (int | None):
        content (str):
        status (str):
        return_reason (None | str):
        result (ManagerInboxItemResponseResultType0 | None):
        created_at (str):
        processed_at (None | str):
    """

    id: int
    from_address: ManagerInboxItemResponseFromAddress
    to_address: ManagerInboxItemResponseToAddress
    re_outbox_item_id: int | None
    content: str
    status: str
    return_reason: None | str
    result: ManagerInboxItemResponseResultType0 | None
    created_at: str
    processed_at: None | str

    def to_dict(self) -> dict[str, Any]:
        from ..models.manager_inbox_item_response_result_type_0 import ManagerInboxItemResponseResultType0

        id = self.id

        from_address = self.from_address.to_dict()

        to_address = self.to_address.to_dict()

        re_outbox_item_id: int | None
        re_outbox_item_id = self.re_outbox_item_id

        content = self.content

        status = self.status

        return_reason: None | str
        return_reason = self.return_reason

        result: dict[str, Any] | None
        if isinstance(self.result, ManagerInboxItemResponseResultType0):
            result = self.result.to_dict()
        else:
            result = self.result

        created_at = self.created_at

        processed_at: None | str
        processed_at = self.processed_at

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "id": id,
                "from_address": from_address,
                "to_address": to_address,
                "re_outbox_item_id": re_outbox_item_id,
                "content": content,
                "status": status,
                "return_reason": return_reason,
                "result": result,
                "created_at": created_at,
                "processed_at": processed_at,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.manager_inbox_item_response_from_address import ManagerInboxItemResponseFromAddress
        from ..models.manager_inbox_item_response_result_type_0 import ManagerInboxItemResponseResultType0
        from ..models.manager_inbox_item_response_to_address import ManagerInboxItemResponseToAddress

        d = dict(src_dict)
        id = d.pop("id")

        from_address = ManagerInboxItemResponseFromAddress.from_dict(d.pop("from_address"))

        to_address = ManagerInboxItemResponseToAddress.from_dict(d.pop("to_address"))

        def _parse_re_outbox_item_id(data: object) -> int | None:
            if data is None:
                return data
            return cast(int | None, data)

        re_outbox_item_id = _parse_re_outbox_item_id(d.pop("re_outbox_item_id"))

        content = d.pop("content")

        status = d.pop("status")

        def _parse_return_reason(data: object) -> None | str:
            if data is None:
                return data
            return cast(None | str, data)

        return_reason = _parse_return_reason(d.pop("return_reason"))

        def _parse_result(data: object) -> ManagerInboxItemResponseResultType0 | None:
            if data is None:
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                result_type_0 = ManagerInboxItemResponseResultType0.from_dict(data)

                return result_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(ManagerInboxItemResponseResultType0 | None, data)

        result = _parse_result(d.pop("result"))

        created_at = d.pop("created_at")

        def _parse_processed_at(data: object) -> None | str:
            if data is None:
                return data
            return cast(None | str, data)

        processed_at = _parse_processed_at(d.pop("processed_at"))

        manager_inbox_item_response = cls(
            id=id,
            from_address=from_address,
            to_address=to_address,
            re_outbox_item_id=re_outbox_item_id,
            content=content,
            status=status,
            return_reason=return_reason,
            result=result,
            created_at=created_at,
            processed_at=processed_at,
        )

        return manager_inbox_item_response
