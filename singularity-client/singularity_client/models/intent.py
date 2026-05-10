from enum import Enum


class Intent(str, Enum):
    CHECKIN_EXPLANATION = "checkin_explanation"
    DATA_QUALITY = "data_quality"
    HYPOTHESIS_CHECK = "hypothesis_check"
    LONGITUDINAL_SCAN = "longitudinal_scan"
    WEEKLY_REVIEW = "weekly_review"

    def __str__(self) -> str:
        return str(self.value)
