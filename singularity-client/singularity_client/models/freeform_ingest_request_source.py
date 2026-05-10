from enum import Enum


class FreeformIngestRequestSource(str, Enum):
    CHAT = "chat"
    FORM = "form"
    MANUAL_NOTE = "manual_note"

    def __str__(self) -> str:
        return str(self.value)
