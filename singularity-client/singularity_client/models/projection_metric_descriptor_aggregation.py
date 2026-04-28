from enum import Enum


class ProjectionMetricDescriptorAggregation(str, Enum):
    LATEST = "latest"
    MEAN = "mean"
    SUM = "sum"

    def __str__(self) -> str:
        return str(self.value)
