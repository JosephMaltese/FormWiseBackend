"""The analysis pipeline: video in, score + annotated video out.

Two passes over the clip:

1. **Measure.** Sample frames down to ``target_sample_fps``, run pose
   estimation, and record every angle the exercise declared. Landmarks are
   cached so the model only ever runs once per sampled frame.
2. **Annotate.** Re-read the video at full frame rate and draw the overlay,
   this time knowing the final score, where each rep started and ended and
   which of them had faults -- information a single streaming pass cannot have
   while it is still in the middle of the set.

Measurement never mutates rep state, which is what keeps each exercise down to
a declaration instead of a bespoke state machine.
"""
from __future__ import annotations

import logging
import math
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.config import Settings
from app.core.cues import build_cues
from app.core.evaluation import (
    ConcentricDirection,
    RepAnalysis,
    classify_quality,
    phase_durations,
)
from app.core.landmarks import Side
from app.core.metrics import MetricTable
from app.core.reps import segment_reps
from app.core.scoring import ScoringContext
from app.exercises.base import ExerciseAnalyzer
from app.exercises.recorder import MetricRecorder
from app.schemas.analysis import (
    AnalysisReport,
    CriterionOut,
    CueOut,
    FaultOut,
    MetricSummary,
    RepOut,
    RepSummary,
    ScoreOut,
    TempoSummary,
    VideoSummary,
)
from app.video import io as video_io
from app.video.overlay import FrameState, OverlayRenderer, TimelineEntry
from app.video.pose import PoseEstimator

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, float], None]


class AnalysisError(RuntimeError):
    """A video that cannot be analysed (unreadable, too long, ...)."""


def _noop(stage: str, fraction: float) -> None:  # pragma: no cover - default
    return None


class LandmarkTrack:
    """Landmarks for sampled frames, interpolated onto every frame.

    Pose runs at ~15 fps while the annotated video is written at the source
    frame rate, so intermediate frames are filled in by interpolating between
    their neighbours. Holding the previous sample instead would make the
    skeleton visibly stutter.
    """

    def __init__(self, indices: list[int], points: list[np.ndarray | None], max_gap: int) -> None:
        self._indices = np.asarray(indices, dtype=int)
        self._points = points
        self._max_gap = max_gap

    def at(self, frame_index: int) -> np.ndarray | None:
        if self._indices.size == 0:
            return None
        position = int(np.searchsorted(self._indices, frame_index))
        if position < self._indices.size and self._indices[position] == frame_index:
            return self._points[position]

        before = position - 1
        after = position
        left = self._points[before] if 0 <= before < len(self._points) else None
        right = self._points[after] if 0 <= after < len(self._points) else None

        if left is None:
            return right if right is not None and self._within(after, frame_index) else None
        if right is None:
            return left if self._within(before, frame_index) else None

        span = int(self._indices[after] - self._indices[before])
        if span <= 0 or span > self._max_gap:
            return left
        weight = (frame_index - int(self._indices[before])) / span
        return (1.0 - weight) * left + weight * right

    def _within(self, position: int, frame_index: int) -> bool:
        if not 0 <= position < self._indices.size:
            return False
        return abs(int(self._indices[position]) - frame_index) <= self._max_gap


@dataclass
class _Measurement:
    table: MetricTable
    track: LandmarkTrack
    sides: dict[int, Side]
    dominant_side: Side | None
    metadata: video_io.VideoMetadata
    stride: int


def _measure(
    source: Path,
    analyzer: ExerciseAnalyzer,
    settings: Settings,
    progress: ProgressCallback,
) -> _Measurement:
    metadata = video_io.probe(source)
    if metadata.duration > settings.max_video_seconds:
        raise AnalysisError(
            f"Video is {metadata.duration:.0f}s long; the limit is "
            f"{settings.max_video_seconds:.0f}s."
        )

    stride = max(1, round(metadata.fps / settings.target_sample_fps))
    expected = max(1, (metadata.frame_count or 1) // stride)
    recorder = MetricRecorder(analyzer, metadata.aspect)

    with video_io.frames(source) as stream, PoseEstimator(settings) as pose:
        for index, frame in stream:
            if index % stride:
                continue
            recorder.add(index, index / metadata.fps, pose.estimate(frame))
            if len(recorder) % 25 == 0:
                progress("measuring", min(0.65, 0.65 * len(recorder) / expected))

    if not len(recorder):
        raise AnalysisError("No frames could be decoded from the video.")

    table = recorder.table(
        median_window=settings.smoothing_median_window, alpha=settings.smoothing_alpha
    )

    return _Measurement(
        table=table,
        track=LandmarkTrack(recorder.frame_indices, recorder.points, max_gap=stride * 2),
        sides=recorder.sides,
        dominant_side=recorder.dominant_side,
        metadata=metadata,
        stride=stride,
    )


def evaluate_reps(analyzer: ExerciseAnalyzer, table: MetricTable) -> list[RepAnalysis]:
    spec = analyzer.spec
    reps = segment_reps(
        table.series(spec.primary_metric),
        table.times,
        min_range_of_motion=spec.min_range_of_motion,
        prominence=spec.extremum_prominence,
    )
    analyses: list[RepAnalysis] = []
    for rep in reps:
        stats = table.stats_bundle(rep.start_index, rep.end_index, rep.bottom_index)
        eccentric, concentric = phase_durations(rep, spec.concentric)
        analyses.append(
            RepAnalysis(
                rep=rep,
                stats=stats,
                faults=analyzer.faults_for_rep(stats),
                quality=classify_quality(rep, spec.full_rom_bottom, spec.full_rom_top),
                eccentric_seconds=eccentric,
                concentric_seconds=concentric,
            )
        )
    return analyses


def _timeline(analyses: list[RepAnalysis]) -> tuple[TimelineEntry, ...]:
    entries = []
    for analysis in analyses:
        if any(fault.severity.value == "major" for fault in analysis.faults):
            color = "bad"
        elif analysis.faults:
            color = "warn"
        else:
            color = "good"
        entries.append(
            TimelineEntry(analysis.number, analysis.rep.start_time, analysis.rep.end_time, color)
        )
    return tuple(entries)


def _frame_state(
    timestamp: float,
    analyses: list[RepAnalysis],
    direction: ConcentricDirection,
    angles: dict[str, float],
) -> FrameState:
    descending_is_concentric = direction is ConcentricDirection.DESCENDING
    for analysis in analyses:
        rep = analysis.rep
        if not rep.start_time <= timestamp <= rep.end_time:
            continue
        descending = timestamp < rep.bottom_time
        if descending:
            phase = "Concentric" if descending_is_concentric else "Eccentric"
        else:
            phase = "Eccentric" if descending_is_concentric else "Concentric"
        return FrameState(
            timestamp=timestamp,
            rep_number=rep.number,
            phase=phase,
            quality=analysis.quality.value,
            faults=tuple(fault.label for fault in analysis.faults),
            angles=angles,
        )
    return FrameState(timestamp=timestamp, phase="Setup", angles=angles)


def _annotate(
    source: Path,
    destination: Path,
    analyzer: ExerciseAnalyzer,
    measurement: _Measurement,
    analyses: list[RepAnalysis],
    score: float,
    grade: str,
    settings: Settings,
    progress: ProgressCallback,
) -> None:
    metadata = measurement.metadata
    renderer = OverlayRenderer(
        spec=analyzer.spec,
        width=metadata.width,
        height=metadata.height,
        score=score,
        grade=grade,
        total_reps=len(analyses),
        timeline=_timeline(analyses),
        duration=metadata.duration or (len(measurement.table) * measurement.stride / metadata.fps),
    )
    table = measurement.table
    total = max(1, metadata.frame_count)
    side = measurement.dominant_side

    with video_io.frames(source) as stream, video_io.writer(
        destination, metadata.fps, (metadata.width, metadata.height)
    ) as handle:
        for index, frame in stream:
            timestamp = index / metadata.fps
            sample_position = int(np.searchsorted(table.frame_indices, index))
            sample_position = min(sample_position, max(0, len(table) - 1))
            angles = {
                name: table.value_at(name, sample_position) for name in table.metric_names
            }
            state = _frame_state(timestamp, analyses, analyzer.spec.concentric, angles)
            points = measurement.track.at(index)
            handle.write(renderer.render(frame, points, side, state))
            if index % 30 == 0:
                progress("annotating", 0.65 + min(0.33, 0.33 * index / total))


def _build_report(
    analyzer: ExerciseAnalyzer,
    measurement: _Measurement,
    analyses: list[RepAnalysis],
    breakdown,
) -> AnalysisReport:
    spec = analyzer.spec
    table = measurement.table
    whole_set = table.whole_set_stats()

    cues, strengths = build_cues(
        analyses, breakdown, table.detection_rate, spec.camera_guidance
    )

    rep_details = [
        RepOut(
            number=analysis.number,
            start_seconds=round(analysis.rep.start_time, 2),
            bottom_seconds=round(analysis.rep.bottom_time, 2),
            end_seconds=round(analysis.rep.end_time, 2),
            eccentric_seconds=round(analysis.eccentric_seconds, 2),
            concentric_seconds=round(analysis.concentric_seconds, 2),
            range_of_motion_degrees=round(analysis.rep.range_of_motion, 1),
            bottom_angle=round(analysis.rep.bottom_angle, 1),
            top_angle=round(analysis.rep.top_angle, 1),
            quality=analysis.quality.value,
            faults=[
                FaultOut(
                    code=fault.code,
                    label=fault.label,
                    severity=fault.severity.value,
                    cue=fault.cue,
                    measured=round(fault.measured, 2),
                    threshold=fault.threshold,
                    unit=fault.unit,
                    detail=fault.detail,
                )
                for fault in analysis.faults
            ],
        )
        for analysis in analyses
    ]

    def mean_or_none(values: list[float]) -> float | None:
        usable = [value for value in values if value > 0]
        return round(statistics.fmean(usable), 2) if usable else None

    def summarise(value: float) -> float | None:
        return None if math.isnan(value) else round(value, 1)

    full = sum(1 for analysis in analyses if analysis.quality.value == "full")
    clean = sum(1 for analysis in analyses if analysis.is_clean)

    return AnalysisReport(
        exercise=spec.key,
        exercise_name=spec.name,
        primary_metric=spec.primary_metric,
        primary_metric_label=spec.label_for(spec.primary_metric),
        score=ScoreOut(
            total=breakdown.total,
            grade=breakdown.grade,
            criteria=[
                CriterionOut(
                    key=result.key,
                    label=result.label,
                    weight=result.weight,
                    earned=result.earned,
                    ratio=round(result.ratio, 3),
                    detail=result.detail,
                )
                for result in breakdown.criteria
            ],
        ),
        reps=RepSummary(
            attempted=len(analyses),
            full_range=full,
            partial_range=len(analyses) - full,
            clean=clean,
            with_faults=len(analyses) - clean,
        ),
        rep_details=rep_details,
        tempo=TempoSummary(
            average_eccentric_seconds=mean_or_none([a.eccentric_seconds for a in analyses]),
            average_concentric_seconds=mean_or_none([a.concentric_seconds for a in analyses]),
            average_rep_seconds=mean_or_none([a.rep.duration for a in analyses]),
        ),
        metrics=[
            MetricSummary(
                key=name,
                label=spec.label_for(name),
                minimum=summarise(stats.minimum),
                maximum=summarise(stats.maximum),
                mean=summarise(stats.mean),
            )
            for name, stats in whole_set.items()
        ],
        cues=[
            CueOut(
                code=cue.code,
                message=cue.message,
                severity=cue.severity,
                affected_reps=list(cue.affected_reps),
            )
            for cue in cues
        ],
        strengths=strengths,
        video=VideoSummary(
            duration_seconds=round(measurement.metadata.duration, 2),
            source_fps=round(measurement.metadata.fps, 2),
            width=measurement.metadata.width,
            height=measurement.metadata.height,
            frames_total=measurement.metadata.frame_count,
            frames_analyzed=len(table),
            sample_fps=round(measurement.metadata.fps / measurement.stride, 2),
            detection_rate=round(table.detection_rate, 3),
            tracked_side=measurement.dominant_side.value if measurement.dominant_side else None,
        ),
        camera_guidance=spec.camera_guidance,
    )


def analyze_video(
    source: Path,
    destination: Path,
    exercise: str,
    settings: Settings,
    progress: ProgressCallback | None = None,
) -> AnalysisReport:
    """Score ``source`` and write the annotated clip to ``destination``."""
    report_progress = progress or _noop
    from app.exercises import get_analyzer  # local import keeps the registry populated

    analyzer = get_analyzer(exercise, settings.landmark_visibility_threshold)
    report_progress("measuring", 0.0)
    measurement = _measure(source, analyzer, settings, report_progress)

    report_progress("scoring", 0.66)
    analyses = evaluate_reps(analyzer, measurement.table)
    breakdown = analyzer.build_rubric().score(
        ScoringContext(reps=analyses, table=measurement.table, direction=analyzer.spec.concentric)
    )

    if settings.draw_overlay:
        _annotate(
            source, destination, analyzer, measurement, analyses,
            breakdown.total, breakdown.grade, settings, report_progress,
        )
        report_progress("encoding", 0.98)
        video_io.transcode_for_web(destination, settings.resolve_ffmpeg())

    report_progress("done", 1.0)
    return _build_report(analyzer, measurement, analyses, breakdown)
