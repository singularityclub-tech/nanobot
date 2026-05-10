from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.manager_review_request_recommended_actions_type_0_item import (
        ManagerReviewRequestRecommendedActionsType0Item,
    )


T = TypeVar("T", bound="ManagerReviewRequest")


@_attrs_define
class ManagerReviewRequest:
    """
    Attributes:
        action (str):
        recommended_actions (list[ManagerReviewRequestRecommendedActionsType0Item] | None | Unset):
    """

    action: str
    recommended_actions: list[ManagerReviewRequestRecommendedActionsType0Item] | None | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        action = self.action

        recommended_actions: list[dict[str, Any]] | None | Unset
        if isinstance(self.recommended_actions, Unset):
            recommended_actions = UNSET
        elif isinstance(self.recommended_actions, list):
            recommended_actions = []
            for recommended_actions_type_0_item_data in self.recommended_actions:
                recommended_actions_type_0_item = recommended_actions_type_0_item_data.to_dict()
                recommended_actions.append(recommended_actions_type_0_item)

        else:
            recommended_actions = self.recommended_actions

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "action": action,
            }
        )
        if recommended_actions is not UNSET:
            field_dict["recommended_actions"] = recommended_actions

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.manager_review_request_recommended_actions_type_0_item import (
            ManagerReviewRequestRecommendedActionsType0Item,
        )

        d = dict(src_dict)
        action = d.pop("action")

        def _parse_recommended_actions(
            data: object,
        ) -> list[ManagerReviewRequestRecommendedActionsType0Item] | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                recommended_actions_type_0 = []
                _recommended_actions_type_0 = data
                for recommended_actions_type_0_item_data in _recommended_actions_type_0:
                    recommended_actions_type_0_item = ManagerReviewRequestRecommendedActionsType0Item.from_dict(
                        recommended_actions_type_0_item_data
                    )

                    recommended_actions_type_0.append(recommended_actions_type_0_item)

                return recommended_actions_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[ManagerReviewRequestRecommendedActionsType0Item] | None | Unset, data)

        recommended_actions = _parse_recommended_actions(d.pop("recommended_actions", UNSET))

        manager_review_request = cls(
            action=action,
            recommended_actions=recommended_actions,
        )

        return manager_review_request
