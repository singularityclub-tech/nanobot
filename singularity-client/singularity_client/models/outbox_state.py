from enum import Enum


class OutboxState(str, Enum):
    ANSWERED = "answered"
    CANCELLED = "cancelled"
    CLAIMED = "claimed"
    EXPIRED = "expired"
    FAILED = "failed"
    READY = "ready"
    SCHEDULED = "scheduled"
    SENT = "sent"

    def __str__(self) -> str:
        return str(self.value)
