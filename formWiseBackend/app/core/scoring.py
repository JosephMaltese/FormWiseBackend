"""Weighted, explainable 0-100 form score.

Every exercise declares a rubric: a list of criteria whose weights sum to 100.
Each criterion returns a ratio in ``[0, 1]`` plus a sentence explaining the
ratio, so the API can show *why* a set scored what it did rather than handing
the user an unsourced number.
"""
from __future__ import annotations

import math
import statistics
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from app.core.evaluation import ConcentricDirection, RepAnalysis
from app.core.metrics import MetricTable

Direction = Literal["lower_is_better", "higher_is_better"]
Phase = Literal["eccentric", "concentric"]


def clamp01(value: float) -> float:
    if math.isnan(value):
        return 0.0
    return max(0.0, min(1.0, value))


def ramp(value: float, full_credit: float, no_credit: float) -> float:
    """Linear partial credit between two thresholds, in either direction."""
    if math.isnan(value) or full_credit == no_credit:
        return 0.0
    return clamp01((no_credit - value) / (no_credit - full_credit))


@dataclass(frozen=True)
class ScoringContext:
    reps: Sequence[RepAnalysis]
    table: MetricTable
    direction: ConcentricDirection


@dataclass(frozen=True)
class CriterionResult:
    key: str
    label: str
    weight: float
    ratio: float
    detail: str
    #: Fault codes this criterion scores, so the cue builder does not repeat
    #: itself ("your heels lifted" followed by "heel contact cost you points").
    covers: tuple[str, ...] = ()

    @property
    def earned(self) -> float:
        return round(self.weight * self.ratio, 1)


class Criterion(ABC):
    def __init__(self, key: str, label: str, weight: float) -> None:
        self.key = key
        self.label = label
        self.weight = weight

    @abstractmethod
    def evaluate(self, context: ScoringContext) -> CriterionResult: ...

    #: Fault codes scored by this criterion; overridden by fault-based criteria.
    covers: tuple[str, ...] = ()

    def _result(self, ratio: float, detail: str) -> CriterionResult:
        return CriterionResult(
            self.key, self.label, self.weight, clamp01(ratio), detail, self.covers
        )


class MetricTargetCriterion(Criterion):
    """Average per-rep partial credit for a metric hitting a target.

    Partial credit matters: a squat stopping two degrees above parallel is not
    the same mistake as a quarter squat, and binary thresholds cannot say so.
    """

    def __init__(
        self,
        key: str,
        label: str,
        weight: float,
        *,
        metric: str,
        stat: str,
        full_credit: float,
        no_credit: float,
        unit: str = "°",
    ) -> None:
        super().__init__(key, label, weight)
        self.metric = metric
        self.stat = stat
        self.full_credit = full_credit
        self.no_credit = no_credit
        self.unit = unit

    def evaluate(self, context: ScoringContext) -> CriterionResult:
        values = [rep.stat(self.metric, self.stat) for rep in context.reps]
        usable = [value for value in values if not math.isnan(value)]
        if not usable:
            return self._result(0.0, "Not measurable from this camera angle.")
        ratio = statistics.fmean(ramp(value, self.full_credit, self.no_credit) for value in usable)
        average = statistics.fmean(usable)
        return self._result(
            ratio,
            f"Averaged {average:.0f}{self.unit} across {len(usable)} rep(s); "
            f"full credit at {self.full_credit:.0f}{self.unit}.",
        )


class RangeOfMotionCriterion(Criterion):
    """Credit for depth and lockout, measured on the primary joint."""

    def __init__(
        self,
        key: str,
        label: str,
        weight: float,
        *,
        full_bottom: float,
        full_top: float,
        bottom_tolerance: float = 30.0,
        top_tolerance: float = 30.0,
    ) -> None:
        super().__init__(key, label, weight)
        self.full_bottom = full_bottom
        self.full_top = full_top
        self.bottom_tolerance = bottom_tolerance
        self.top_tolerance = top_tolerance

    def evaluate(self, context: ScoringContext) -> CriterionResult:
        if not context.reps:
            return self._result(0.0, "No reps detected.")
        scores = []
        for analysis in context.reps:
            depth = ramp(
                analysis.rep.bottom_angle,
                self.full_bottom,
                self.full_bottom + self.bottom_tolerance,
            )
            lockout = ramp(
                -analysis.rep.top_angle,
                -self.full_top,
                -(self.full_top - self.top_tolerance),
            )
            scores.append((depth + lockout) / 2.0)
        full = sum(1 for rep in context.reps if rep.quality.value == "full")
        return self._result(
            statistics.fmean(scores),
            f"{full}/{len(context.reps)} rep(s) reached full range "
            f"(bottom ≤ {self.full_bottom:.0f}°, top ≥ {self.full_top:.0f}°).",
        )


class FaultFreeCriterion(Criterion):
    """Share of reps free of the listed faults, weighted by severity."""

    def __init__(
        self,
        key: str,
        label: str,
        weight: float,
        *,
        codes: tuple[str, ...],
    ) -> None:
        super().__init__(key, label, weight)
        self.codes = codes
        self.covers = codes

    def evaluate(self, context: ScoringContext) -> CriterionResult:
        if not context.reps:
            return self._result(0.0, "No reps detected.")
        penalties = []
        affected = 0
        for analysis in context.reps:
            relevant = [fault for fault in analysis.faults if fault.code in self.codes]
            if relevant:
                affected += 1
            penalties.append(min(1.0, sum(fault.severity.weight for fault in relevant)))
        ratio = 1.0 - statistics.fmean(penalties)
        if affected == 0:
            detail = f"Clean on all {len(context.reps)} rep(s)."
        else:
            detail = f"Flagged on {affected} of {len(context.reps)} rep(s)."
        return self._result(ratio, detail)


class TempoCriterion(Criterion):
    """Rewards a controlled phase duration inside an ideal window."""

    def __init__(
        self,
        key: str,
        label: str,
        weight: float,
        *,
        phase: Phase = "eccentric",
        ideal: tuple[float, float] = (1.5, 3.5),
        bounds: tuple[float, float] = (0.4, 7.0),
    ) -> None:
        super().__init__(key, label, weight)
        self.phase = phase
        self.ideal = ideal
        self.bounds = bounds

    def _score(self, seconds: float) -> float:
        low_ideal, high_ideal = self.ideal
        low_bound, high_bound = self.bounds
        if math.isnan(seconds) or seconds <= 0:
            return 0.0
        if low_ideal <= seconds <= high_ideal:
            return 1.0
        if seconds < low_ideal:
            return ramp(seconds, low_ideal, low_bound)
        return ramp(seconds, high_ideal, high_bound)

    def evaluate(self, context: ScoringContext) -> CriterionResult:
        durations = [
            analysis.eccentric_seconds if self.phase == "eccentric" else analysis.concentric_seconds
            for analysis in context.reps
        ]
        usable = [value for value in durations if value > 0]
        if not usable:
            return self._result(0.0, "No reps detected.")
        ratio = statistics.fmean(self._score(value) for value in usable)
        return self._result(
            ratio,
            f"{self.phase.capitalize()} phase averaged {statistics.fmean(usable):.1f}s; "
            f"target {self.ideal[0]:.1f}-{self.ideal[1]:.1f}s.",
        )


class ConsistencyCriterion(Criterion):
    """Penalises reps that drift -- depth fading across a set is a form issue."""

    def __init__(
        self,
        key: str,
        label: str,
        weight: float,
        *,
        tolerance: float = 12.0,
        no_credit: float = 40.0,
    ) -> None:
        super().__init__(key, label, weight)
        self.tolerance = tolerance
        self.no_credit = no_credit

    def evaluate(self, context: ScoringContext) -> CriterionResult:
        bottoms = [analysis.rep.bottom_angle for analysis in context.reps]
        if not bottoms:
            return self._result(0.0, "No reps detected.")
        if len(bottoms) == 1:
            return self._result(1.0, "Single rep -- nothing to compare against.")
        spread = statistics.pstdev(bottoms)
        return self._result(
            ramp(spread, self.tolerance, self.no_credit),
            f"Bottom position varied by ±{spread:.0f}° across the set.",
        )


class SetupCriterion(Criterion):
    """Whole-set check on a static setup detail (bench leg drive, for example)."""

    def __init__(
        self,
        key: str,
        label: str,
        weight: float,
        *,
        metric: str,
        low: float,
        high: float,
        tolerance: float = 20.0,
    ) -> None:
        super().__init__(key, label, weight)
        self.metric = metric
        self.low = low
        self.high = high
        self.tolerance = tolerance

    def evaluate(self, context: ScoringContext) -> CriterionResult:
        stats = context.table.whole_set_stats().get(self.metric)
        value = float("nan") if stats is None else stats.mean
        if math.isnan(value):
            return self._result(0.0, "Not measurable from this camera angle.")
        if self.low <= value <= self.high:
            ratio = 1.0
        elif value < self.low:
            ratio = ramp(value, self.low, self.low - self.tolerance)
        else:
            ratio = ramp(-value, -self.high, -(self.high + self.tolerance))
        return self._result(
            ratio,
            f"Held at {value:.0f}° on average; target {self.low:.0f}-{self.high:.0f}°.",
        )


@dataclass(frozen=True)
class ScoreBreakdown:
    total: float
    grade: str
    criteria: list[CriterionResult]


GRADE_BANDS: tuple[tuple[float, str], ...] = (
    (90.0, "Excellent"),
    (75.0, "Solid"),
    (60.0, "Developing"),
    (40.0, "Needs work"),
    (0.0, "Poor"),
)


def grade_for(total: float) -> str:
    for floor, label in GRADE_BANDS:
        if total >= floor:
            return label
    return GRADE_BANDS[-1][1]


class Rubric:
    def __init__(self, *criteria: Criterion) -> None:
        if not criteria:
            raise ValueError("a rubric needs at least one criterion")
        total_weight = sum(criterion.weight for criterion in criteria)
        if abs(total_weight - 100.0) > 1e-6:
            raise ValueError(f"rubric weights must sum to 100, got {total_weight}")
        self.criteria = criteria

    def score(self, context: ScoringContext) -> ScoreBreakdown:
        if not context.reps:
            results = [
                CriterionResult(c.key, c.label, c.weight, 0.0, "No reps detected.", c.covers)
                for c in self.criteria
            ]
            return ScoreBreakdown(0.0, grade_for(0.0), results)
        results = [criterion.evaluate(context) for criterion in self.criteria]
        total = round(min(100.0, max(0.0, sum(result.earned for result in results))), 1)
        return ScoreBreakdown(total, grade_for(total), results)
