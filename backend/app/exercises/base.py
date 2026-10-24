"""The contract every exercise implements, plus the registry that exposes them.

An exercise is mostly declarative: which joints the camera must see, which
angles to measure, what counts as a full rep, which mistakes to flag and how the
100 points are split. The frame loop, rep detection, scoring and rendering are
shared.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from app.core.evaluation import ConcentricDirection
from app.core.faults import Fault, FaultRule, evaluate_rules
from app.core.landmarks import PoseSample, Side, choose_side
from app.core.metrics import MetricStats
from app.core.scoring import Rubric


@dataclass(frozen=True)
class AngleOverlay:
    """An angle to draw on the annotated video.

    Three joints render as an arc at the middle joint; two render as a labelled
    segment (used for angles measured against vertical or horizontal).
    """

    metric: str
    joints: tuple[str, ...]
    short_label: str

    @property
    def vertex(self) -> str:
        return self.joints[1]


@dataclass(frozen=True)
class ExerciseSpec:
    key: str
    name: str
    summary: str
    camera_guidance: str
    primary_metric: str
    concentric: ConcentricDirection
    #: A rep's bottom must reach this angle (or lower) to count as full range.
    full_rom_bottom: float
    #: ...and its weaker end position must reach this angle (or higher).
    full_rom_top: float
    #: Minimum peak-to-trough excursion before a wobble is treated as a rep.
    min_range_of_motion: float
    #: Reversal size needed to confirm a turning point in the primary angle.
    extremum_prominence: float
    #: Joints that must be visible; also decides which side of the body to track.
    tracking_joints: tuple[str, ...]
    overlays: tuple[AngleOverlay, ...] = ()
    metric_labels: dict[str, str] = field(default_factory=dict)
    aliases: tuple[str, ...] = ()

    def label_for(self, metric: str) -> str:
        return self.metric_labels.get(metric, metric.replace("_", " ").title())


class ExerciseAnalyzer(ABC):
    """Per-frame measurement + per-rep interpretation for one exercise."""

    spec: ExerciseSpec
    fault_rules: tuple[FaultRule, ...] = ()

    def __init__(self, visibility_threshold: float = 0.45) -> None:
        self.visibility_threshold = visibility_threshold

    # -- frame pass --------------------------------------------------------
    def select_side(self, points: np.ndarray) -> Side | None:
        return choose_side(points, self.spec.tracking_joints, self.visibility_threshold)

    def sample(self, points: np.ndarray, aspect: float = 1.0) -> PoseSample | None:
        side = self.select_side(points)
        return None if side is None else PoseSample(points, side, aspect)

    def measure_frame(
        self, points: np.ndarray, aspect: float = 1.0
    ) -> tuple[dict[str, float], Side | None]:
        """Measure every declared metric for one frame of landmarks."""
        sample = self.sample(points, aspect)
        if sample is None:
            return ({name: math.nan for name in self.metric_names}, None)
        measured = self.measure(sample)
        return ({name: measured.get(name, math.nan) for name in self.metric_names}, sample.side)

    @property
    def metric_names(self) -> tuple[str, ...]:
        return tuple(self.metric_labels)

    @property
    def metric_labels(self) -> dict[str, str]:
        return self.spec.metric_labels

    @abstractmethod
    def measure(self, sample: PoseSample) -> dict[str, float]:
        """Return this frame's metric values; omit or NaN whatever is unclear."""

    # -- rep pass ----------------------------------------------------------
    def faults_for_rep(self, stats: dict[str, MetricStats]) -> list[Fault]:
        return evaluate_rules(self.fault_rules, stats) + self.extra_faults(stats)

    def extra_faults(self, stats: dict[str, MetricStats]) -> list[Fault]:
        """Hook for checks a declarative threshold rule cannot express."""
        return []

    @abstractmethod
    def build_rubric(self) -> Rubric:
        """The 0-100 scoring rubric; weights must sum to 100."""


_REGISTRY: dict[str, type[ExerciseAnalyzer]] = {}
_ALIASES: dict[str, str] = {}


def register(analyzer_cls: type[ExerciseAnalyzer]) -> type[ExerciseAnalyzer]:
    spec = analyzer_cls.spec
    _REGISTRY[spec.key] = analyzer_cls
    for alias in (spec.key, spec.name, *spec.aliases):
        _ALIASES[_normalise(alias)] = spec.key
    return analyzer_cls


def _normalise(value: str) -> str:
    return "_".join(value.strip().lower().replace("-", " ").replace("_", " ").split())


def resolve_key(exercise: str) -> str | None:
    return _ALIASES.get(_normalise(exercise))


def get_analyzer(exercise: str, visibility_threshold: float = 0.45) -> ExerciseAnalyzer:
    key = resolve_key(exercise)
    if key is None:
        raise KeyError(exercise)
    return _REGISTRY[key](visibility_threshold=visibility_threshold)


def available_specs() -> list[ExerciseSpec]:
    return [analyzer_cls.spec for analyzer_cls in _REGISTRY.values()]
