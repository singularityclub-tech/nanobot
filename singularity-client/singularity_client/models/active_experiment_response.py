from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.active_experiment_response_active_experiment_type_0 import (
        ActiveExperimentResponseActiveExperimentType0,
    )


T = TypeVar("T", bound="ActiveExperimentResponse")


@_attrs_define
class ActiveExperimentResponse:
    """
    Attributes:
        active_experiment (ActiveExperimentResponseActiveExperimentType0 | None | Unset):
    """

    active_experiment: ActiveExperimentResponseActiveExperimentType0 | None | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.active_experiment_response_active_experiment_type_0 import (
            ActiveExperimentResponseActiveExperimentType0,
        )

        active_experiment: dict[str, Any] | None | Unset
        if isinstance(self.active_experiment, Unset):
            active_experiment = UNSET
        elif isinstance(self.active_experiment, ActiveExperimentResponseActiveExperimentType0):
            active_experiment = self.active_experiment.to_dict()
        else:
            active_experiment = self.active_experiment

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if active_experiment is not UNSET:
            field_dict["active_experiment"] = active_experiment

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.active_experiment_response_active_experiment_type_0 import (
            ActiveExperimentResponseActiveExperimentType0,
        )

        d = dict(src_dict)

        def _parse_active_experiment(data: object) -> ActiveExperimentResponseActiveExperimentType0 | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                active_experiment_type_0 = ActiveExperimentResponseActiveExperimentType0.from_dict(data)

                return active_experiment_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(ActiveExperimentResponseActiveExperimentType0 | None | Unset, data)

        active_experiment = _parse_active_experiment(d.pop("active_experiment", UNSET))

        active_experiment_response = cls(
            active_experiment=active_experiment,
        )

        active_experiment_response.additional_properties = d
        return active_experiment_response

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
