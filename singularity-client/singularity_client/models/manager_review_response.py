from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define

from ..types import UNSET, Unset

T = TypeVar("T", bound="ManagerReviewResponse")


@_attrs_define
class ManagerReviewResponse:
    """
    Attributes:
        id (int):
        review_status (str):
        created_protocol_item_ids (list[int] | Unset):
        created_observation_target_ids (list[int] | Unset):
    """

    id: int
    review_status: str
    created_protocol_item_ids: list[int] | Unset = UNSET
    created_observation_target_ids: list[int] | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        id = self.id

        review_status = self.review_status

        created_protocol_item_ids: list[int] | Unset = UNSET
        if not isinstance(self.created_protocol_item_ids, Unset):
            created_protocol_item_ids = self.created_protocol_item_ids

        created_observation_target_ids: list[int] | Unset = UNSET
        if not isinstance(self.created_observation_target_ids, Unset):
            created_observation_target_ids = self.created_observation_target_ids

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "id": id,
                "review_status": review_status,
            }
        )
        if created_protocol_item_ids is not UNSET:
            field_dict["created_protocol_item_ids"] = created_protocol_item_ids
        if created_observation_target_ids is not UNSET:
            field_dict["created_observation_target_ids"] = created_observation_target_ids

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        id = d.pop("id")

        review_status = d.pop("review_status")

        created_protocol_item_ids = cast(list[int], d.pop("created_protocol_item_ids", UNSET))

        created_observation_target_ids = cast(list[int], d.pop("created_observation_target_ids", UNSET))

        manager_review_response = cls(
            id=id,
            review_status=review_status,
            created_protocol_item_ids=created_protocol_item_ids,
            created_observation_target_ids=created_observation_target_ids,
        )

        return manager_review_response
