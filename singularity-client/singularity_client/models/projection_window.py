from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from dateutil.parser import isoparse

from ..models.projection_window_kind import ProjectionWindowKind
from ..types import UNSET, Unset

T = TypeVar("T", bound="ProjectionWindow")


@_attrs_define
class ProjectionWindow:
    """
    Attributes:
        kind (ProjectionWindowKind):
        since (datetime.datetime):
        timezone (str):
        until (datetime.datetime | None | Unset):
    """

    kind: ProjectionWindowKind
    since: datetime.datetime
    timezone: str
    until: datetime.datetime | None | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        kind = self.kind.value

        since = self.since.isoformat()

        timezone = self.timezone

        until: None | str | Unset
        if isinstance(self.until, Unset):
            until = UNSET
        elif isinstance(self.until, datetime.datetime):
            until = self.until.isoformat()
        else:
            until = self.until

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "kind": kind,
                "since": since,
                "timezone": timezone,
            }
        )
        if until is not UNSET:
            field_dict["until"] = until

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        kind = ProjectionWindowKind(d.pop("kind"))

        since = isoparse(d.pop("since"))

        timezone = d.pop("timezone")

        def _parse_until(data: object) -> datetime.datetime | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                until_type_0 = isoparse(data)

                return until_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(datetime.datetime | None | Unset, data)

        until = _parse_until(d.pop("until", UNSET))

        projection_window = cls(
            kind=kind,
            since=since,
            timezone=timezone,
            until=until,
        )

        return projection_window
