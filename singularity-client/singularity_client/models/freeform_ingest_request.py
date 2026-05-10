from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from dateutil.parser import isoparse

from ..models.freeform_ingest_request_source import FreeformIngestRequestSource
from ..models.freeform_ingest_request_source_type import FreeformIngestRequestSourceType
from ..types import UNSET, Unset

T = TypeVar("T", bound="FreeformIngestRequest")


@_attrs_define
class FreeformIngestRequest:
    """
    Attributes:
        text (str):
        source (FreeformIngestRequestSource | Unset):  Default: FreeformIngestRequestSource.CHAT.
        source_type (FreeformIngestRequestSourceType | Unset):  Default: FreeformIngestRequestSourceType.NOTE.
        captured_at (datetime.datetime | None | Unset):
    """

    text: str
    source: FreeformIngestRequestSource | Unset = FreeformIngestRequestSource.CHAT
    source_type: FreeformIngestRequestSourceType | Unset = FreeformIngestRequestSourceType.NOTE
    captured_at: datetime.datetime | None | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        text = self.text

        source: str | Unset = UNSET
        if not isinstance(self.source, Unset):
            source = self.source.value

        source_type: str | Unset = UNSET
        if not isinstance(self.source_type, Unset):
            source_type = self.source_type.value

        captured_at: None | str | Unset
        if isinstance(self.captured_at, Unset):
            captured_at = UNSET
        elif isinstance(self.captured_at, datetime.datetime):
            captured_at = self.captured_at.isoformat()
        else:
            captured_at = self.captured_at

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "text": text,
            }
        )
        if source is not UNSET:
            field_dict["source"] = source
        if source_type is not UNSET:
            field_dict["source_type"] = source_type
        if captured_at is not UNSET:
            field_dict["captured_at"] = captured_at

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        text = d.pop("text")

        _source = d.pop("source", UNSET)
        source: FreeformIngestRequestSource | Unset
        if isinstance(_source, Unset):
            source = UNSET
        else:
            source = FreeformIngestRequestSource(_source)

        _source_type = d.pop("source_type", UNSET)
        source_type: FreeformIngestRequestSourceType | Unset
        if isinstance(_source_type, Unset):
            source_type = UNSET
        else:
            source_type = FreeformIngestRequestSourceType(_source_type)

        def _parse_captured_at(data: object) -> datetime.datetime | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                captured_at_type_0 = isoparse(data)

                return captured_at_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(datetime.datetime | None | Unset, data)

        captured_at = _parse_captured_at(d.pop("captured_at", UNSET))

        freeform_ingest_request = cls(
            text=text,
            source=source,
            source_type=source_type,
            captured_at=captured_at,
        )

        return freeform_ingest_request
