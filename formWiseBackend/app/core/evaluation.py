"""Rep-level evaluation: timings, range-of-motion verdict and faults."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from app.core.faults import Fault, Severity
from app.core.metrics import MetricStats
from app.core.reps import Rep


class ConcentricDirection(str, Enum):
    """Which way the primary angle moves while the muscle shortens.

    A curl flexes the elbow (angle falls) on the way up; a squat and a bench
    press both open their primary joint (angle rises). Declaring this once per
    exercise is all the phase logic any of them need.
    """

    DESCENDING = "descending"
    ASCENDING = "ascending"


class RepQuality(str, Enum):
    FULL = "full"
    PARTIAL = "partial"


@dataclass(frozen=True)
class RepAnalysis:
    rep: Rep
    stats: dict[str, MetricStats]
    faults: list[Fault] = field(default_factory=list)
    quality: RepQuality = RepQuality.PARTIAL
    eccentric_seconds: float = 0.0
    concentric_seconds: float = 0.0

    @property
    def number(self) -> int:
        return self.rep.number

    @property
    def is_clean(self) -> bool:
        return not self.faults

    @property
    def fault_load(self) -> float:
        """0 for a clean rep, 1 when a major fault (or two minor ones) fired."""
        return min(1.0, sum(fault.severity.weight for fault in self.faults))

    def has_fault(self, code: str) -> bool:
        return any(fault.code == code for fault in self.faults)

    def stat(self, metric: str, key: str) -> float:
        metric_stats = self.stats.get(metric)
        return float("nan") if metric_stats is None else metric_stats.get(key)


def phase_durations(rep: Rep, direction: ConcentricDirection) -> tuple[float, float]:
    """Return ``(eccentric_seconds, concentric_seconds)`` for a rep."""
    if direction is ConcentricDirection.DESCENDING:
        return rep.ascent_seconds, rep.descent_seconds
    return rep.descent_seconds, rep.ascent_seconds


def classify_quality(rep: Rep, full_bottom: float, full_top: float) -> RepQuality:
    """A rep is "full" only when it reached depth *and* finished at lockout."""
    if rep.bottom_angle <= full_bottom and rep.top_angle >= full_top:
        return RepQuality.FULL
    return RepQuality.PARTIAL


def severity_rank(faults: list[Fault]) -> Severity | None:
    if any(fault.severity is Severity.MAJOR for fault in faults):
        return Severity.MAJOR
    return Severity.MINOR if faults else None
