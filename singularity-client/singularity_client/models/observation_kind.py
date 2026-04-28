from enum import Enum


class ObservationKind(str, Enum):
    COURSE = "course"
    ENVIRONMENT_CONTEXT = "environment_context"
    EVENT = "event"
    GENOMIC_TENDENCY = "genomic_tendency"
    INTERVAL = "interval"
    MEASUREMENT = "measurement"
    QUESTIONNAIRE_ANSWER = "questionnaire_answer"
    SUBJECTIVE_REPORT = "subjective_report"

    def __str__(self) -> str:
        return str(self.value)
