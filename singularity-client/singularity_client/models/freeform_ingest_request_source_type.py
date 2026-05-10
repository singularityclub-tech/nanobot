from enum import Enum


class FreeformIngestRequestSourceType(str, Enum):
    NOTE = "note"
    SELF_REPORT = "self_report"

    def __str__(self) -> str:
        return str(self.value)
