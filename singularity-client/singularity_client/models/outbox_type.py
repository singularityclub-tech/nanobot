from enum import Enum


class OutboxType(str, Enum):
    APPROVAL_REQUEST = "approval_request"
    EXPERIMENT_PROPOSAL = "experiment_proposal"
    FILE_SEND = "file_send"
    FOLLOWUP_QUESTION = "followup_question"
    MANAGER_ESCALATION = "manager_escalation"
    NOTIFICATION = "notification"

    def __str__(self) -> str:
        return str(self.value)
