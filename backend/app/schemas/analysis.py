"""Wire format for analysis results.

These are the project's public contract: the frontend reads nothing from the
pipeline that is not described here.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class AnalysisStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        return self in (AnalysisStatus.COMPLETED, AnalysisStatus.FAILED)


class FaultOut(BaseModel):
    code: str
    label: str
    severity: str
    cue: str
    measured: float
    threshold: float
    unit: str
    detail: str


class RepOut(BaseModel):
    number: int
    start_seconds: float
    bottom_seconds: float
    end_seconds: float
    eccentric_seconds: float
    concentric_seconds: float
    range_of_motion_degrees: float
    bottom_angle: float
    top_angle: float
    quality: str
    faults: list[FaultOut] = Field(default_factory=list)


class CriterionOut(BaseModel):
    key: str
    label: str
    weight: float
    earned: float
    ratio: float
    detail: str


class ScoreOut(BaseModel):
    total: float = Field(ge=0, le=100)
    grade: str
    criteria: list[CriterionOut]


class RepSummary(BaseModel):
    attempted: int
    full_range: int
    partial_range: int
    clean: int
    with_faults: int


class TempoSummary(BaseModel):
    average_eccentric_seconds: float | None = None
    average_concentric_seconds: float | None = None
    average_rep_seconds: float | None = None


class MetricSummary(BaseModel):
    key: str
    label: str
    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None


class CueOut(BaseModel):
    code: str
    message: str
    severity: str
    affected_reps: list[int] = Field(default_factory=list)


class VideoSummary(BaseModel):
    duration_seconds: float
    source_fps: float
    width: int
    height: int
    frames_total: int
    frames_analyzed: int
    sample_fps: float
    detection_rate: float = Field(ge=0, le=1)
    tracked_side: str | None = None


class AnalysisReport(BaseModel):
    exercise: str
    exercise_name: str
    primary_metric: str
    primary_metric_label: str
    score: ScoreOut
    reps: RepSummary
    rep_details: list[RepOut]
    tempo: TempoSummary
    metrics: list[MetricSummary]
    cues: list[CueOut]
    strengths: list[str]
    video: VideoSummary
    camera_guidance: str


class AnalysisEnvelope(BaseModel):
    id: str
    exercise: str
    status: AnalysisStatus
    progress: float = Field(ge=0, le=1)
    stage: str
    created_at: datetime
    updated_at: datetime
    expires_at: datetime
    source_filename: str | None = None
    annotated_video_url: str | None = None
    report: AnalysisReport | None = None
    error: str | None = None


class AnalysisCreated(BaseModel):
    id: str
    exercise: str
    status: AnalysisStatus
    status_url: str
    annotated_video_url: str


class ExerciseOut(BaseModel):
    key: str
    name: str
    summary: str
    camera_guidance: str
    primary_metric: str
    primary_metric_label: str
    scoring: list[dict[str, object]]


class HealthOut(BaseModel):
    status: str
    version: str
    exercises: list[str]
    active_analyses: int
