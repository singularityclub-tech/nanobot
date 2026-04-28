from enum import Enum


class ProjectionWindowKind(str, Enum):
    LOCAL_DAY = "local_day"
    ROLLING_28D = "rolling_28d"
    ROLLING_7D = "rolling_7d"

    def __str__(self) -> str:
        return str(self.value)
