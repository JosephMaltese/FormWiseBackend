"""Declarative form-fault rules.

Each exercise lists the mistakes it cares about as data instead of hand-rolling
another branch inside the frame loop. A rule reads one summary statistic of one
metric over one rep and compares it against a threshold.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Literal

from app.core.metrics import MetricStats

Comparator = Literal["gt", "lt"]


class Severity(str, Enum):
    MINOR = "minor"
    MAJOR = "major"

    @property
    def weight(self) -> float:
        """How much of a rep's "clean" credit this fault consumes."""
        return 1.0 if self is Severity.MAJOR else 0.5


@dataclass(frozen=True)
class Fault:
    code: str
    label: str
    severity: Severity
    cue: str
    measured: float
    threshold: float
    unit: str = "°"

    @property
    def detail(self) -> str:
        return (
            f"{self.label}: measured {self.measured:.0f}{self.unit} "
            f"(limit {self.threshold:.0f}{self.unit})"
        )


@dataclass(frozen=True)
class FaultRule:
    code: str
    label: str
    metric: str
    stat: str
    comparator: Comparator
    threshold: float
    severity: Severity
    cue: str
    unit: str = "°"

    def evaluate(self, stats: dict[str, MetricStats]) -> Fault | None:
        metric_stats = stats.get(self.metric)
        if metric_stats is None:
            return None
        measured = metric_stats.get(self.stat)
        if math.isnan(measured):
            return None  # metric was never visible for this rep
        tripped = measured > self.threshold if self.comparator == "gt" else measured < self.threshold
        if not tripped:
            return None
        return Fault(
            code=self.code,
            label=self.label,
            severity=self.severity,
            cue=self.cue,
            measured=measured,
            threshold=self.threshold,
            unit=self.unit,
        )


def evaluate_rules(
    rules: tuple[FaultRule, ...], stats: dict[str, MetricStats]
) -> list[Fault]:
    return [fault for rule in rules if (fault := rule.evaluate(stats)) is not None]
