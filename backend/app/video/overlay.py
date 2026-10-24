"""Draws the coaching overlay onto each frame of the annotated video.

What the viewer gets: a clean body-only skeleton, live arcs for the joint angles
the exercise actually cares about, a heads-up panel with the score and rep
counter, fault chips for the rep on screen, and a timeline strip showing where
the good and bad reps are so the clip can be scrubbed straight to a mistake.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2 as cv
import numpy as np

from app.core.landmarks import SKELETON_EDGES, Landmark, Side, resolve
from app.exercises.base import ExerciseSpec

FONT = cv.FONT_HERSHEY_SIMPLEX

# BGR palette
WHITE = (242, 242, 242)
MUTED = (168, 168, 168)
DIM = (96, 96, 96)
PANEL = (30, 28, 26)
ACCENT = (242, 184, 92)
GOOD = (120, 206, 126)
WARN = (74, 186, 243)
BAD = (82, 82, 236)

SEVERITY_COLORS = {"major": BAD, "minor": WARN, "info": MUTED}
QUALITY_COLORS = {"full": GOOD, "partial": WARN}


@dataclass(frozen=True)
class TimelineEntry:
    number: int
    start_seconds: float
    end_seconds: float
    color_key: str  # "good" | "warn" | "bad"


@dataclass(frozen=True)
class FrameState:
    """Everything the renderer needs to know about one moment in the clip."""

    timestamp: float
    rep_number: int | None = None
    phase: str = "Setup"
    quality: str | None = None
    faults: tuple[str, ...] = ()
    angles: dict[str, float] = field(default_factory=dict)


TIMELINE_COLORS = {"good": GOOD, "warn": WARN, "bad": BAD}


def _text(
    frame: np.ndarray,
    message: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    """Text with a dark halo so it stays legible over any background."""
    cv.putText(frame, message, origin, FONT, scale, (0, 0, 0), thickness + 2, cv.LINE_AA)
    cv.putText(frame, message, origin, FONT, scale, color, thickness, cv.LINE_AA)


def _degree_glyph(
    frame: np.ndarray,
    after: tuple[int, int],
    text: str,
    scale: float,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    """Hershey fonts are ASCII only, so the degree sign is drawn as a ring."""
    (text_width, text_height), _ = cv.getTextSize(text, FONT, scale, thickness)
    radius = max(2, round(text_height * 0.16))
    centre = (after[0] + text_width + radius + 2, after[1] - text_height + radius)
    cv.circle(frame, centre, radius + 1, (0, 0, 0), -1, cv.LINE_AA)
    cv.circle(frame, centre, radius, color, 1, cv.LINE_AA)


def _panel(frame: np.ndarray, top_left: tuple[int, int], bottom_right: tuple[int, int],
           alpha: float = 0.62, color: tuple[int, int, int] = PANEL) -> None:
    x0 = max(0, top_left[0])
    y0 = max(0, top_left[1])
    x1 = min(frame.shape[1], bottom_right[0])
    y1 = min(frame.shape[0], bottom_right[1])
    if x1 <= x0 or y1 <= y0:
        return
    region = frame[y0:y1, x0:x1]
    overlay = np.full_like(region, color, dtype=np.uint8)
    cv.addWeighted(overlay, alpha, region, 1.0 - alpha, 0.0, dst=region)


class OverlayRenderer:
    def __init__(
        self,
        spec: ExerciseSpec,
        width: int,
        height: int,
        score: float,
        grade: str,
        total_reps: int,
        timeline: tuple[TimelineEntry, ...],
        duration: float,
    ) -> None:
        self.spec = spec
        self.width = width
        self.height = height
        self.score = score
        self.grade = grade
        self.total_reps = total_reps
        self.timeline = timeline
        self.duration = max(duration, 1e-6)
        self.scale = max(0.55, height / 900.0)
        self.line = max(2, round(height / 320))
        self.joint_radius = max(3, round(height / 260))

    # -- geometry helpers --------------------------------------------------
    def _pixel(self, points: np.ndarray, landmark: Landmark) -> tuple[int, int]:
        row = points[int(landmark)]
        return (int(row[0] * self.width), int(row[1] * self.height))

    def _visible(self, points: np.ndarray, landmark: Landmark, threshold: float) -> bool:
        return bool(points[int(landmark)][3] >= threshold)

    # -- layers ------------------------------------------------------------
    def draw_skeleton(self, frame: np.ndarray, points: np.ndarray, side: Side | None,
                      threshold: float = 0.3) -> None:
        tracked = set()
        if side is not None:
            tracked = {int(resolve(joint, side)) for joint in self.spec.tracking_joints}

        for start, end in SKELETON_EDGES:
            if not (self._visible(points, start, threshold) and self._visible(points, end, threshold)):
                continue
            on_tracked_side = int(start) in tracked and int(end) in tracked
            color = ACCENT if on_tracked_side else DIM
            thickness = self.line if on_tracked_side else max(1, self.line - 1)
            cv.line(frame, self._pixel(points, start), self._pixel(points, end), color,
                    thickness, cv.LINE_AA)

        for landmark in Landmark:
            if landmark < Landmark.LEFT_SHOULDER:
                continue  # skip the face mesh
            if not self._visible(points, landmark, threshold):
                continue
            color = WHITE if int(landmark) in tracked else MUTED
            cv.circle(frame, self._pixel(points, landmark), self.joint_radius, color, -1, cv.LINE_AA)

    def draw_angles(self, frame: np.ndarray, points: np.ndarray, side: Side | None,
                    angles: dict[str, float]) -> None:
        if side is None:
            return
        for overlay in self.spec.overlays:
            value = angles.get(overlay.metric, math.nan)
            if math.isnan(value):
                continue
            landmarks = [resolve(joint, side) for joint in overlay.joints]
            if not all(self._visible(points, landmark, 0.3) for landmark in landmarks):
                continue
            vertex = self._pixel(points, landmarks[1])
            if len(landmarks) == 3:
                self._draw_arc(
                    frame,
                    self._pixel(points, landmarks[0]),
                    vertex,
                    self._pixel(points, landmarks[2]),
                )
            else:
                self._draw_plumb_line(frame, self._pixel(points, landmarks[0]), vertex)
            label = f"{overlay.short_label} {value:.0f}"
            origin = (vertex[0] + self.joint_radius * 3, vertex[1] - self.joint_radius)
            label_scale = 0.52 * self.scale
            label_weight = max(1, round(self.scale))
            _text(frame, label, origin, label_scale, WHITE, label_weight)
            if not overlay.metric.endswith("travel"):
                _degree_glyph(frame, origin, label, label_scale, WHITE, label_weight)

    def _draw_arc(self, frame: np.ndarray, a: tuple[int, int], b: tuple[int, int],
                  c: tuple[int, int]) -> None:
        radius = max(20, round(self.height / 18))
        start = math.degrees(math.atan2(a[1] - b[1], a[0] - b[0]))
        end = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]))
        sweep = (end - start) % 360
        if sweep > 180:
            start, sweep = end, 360 - sweep
        cv.ellipse(frame, b, (radius, radius), 0.0, start, start + sweep, WARN,
                   max(2, self.line), cv.LINE_AA)

    def _draw_plumb_line(self, frame: np.ndarray, anchor: tuple[int, int],
                         vertex: tuple[int, int]) -> None:
        """Dashed vertical reference, for angles measured against gravity."""
        length = max(24, round(self.height / 10))
        step = max(6, length // 7)
        for offset in range(0, length, step * 2):
            y0 = anchor[1] - offset
            y1 = max(anchor[1] - length, y0 - step)
            cv.line(frame, (anchor[0], y0), (anchor[0], y1), MUTED, max(1, self.line - 1), cv.LINE_AA)
        cv.line(frame, anchor, vertex, WARN, max(1, self.line - 1), cv.LINE_AA)

    def draw_hud(self, frame: np.ndarray, state: FrameState) -> None:
        pad = round(14 * self.scale)
        width = round(300 * self.scale)
        height = round(132 * self.scale)
        _panel(frame, (pad, pad), (pad + width, pad + height))

        x = pad + round(14 * self.scale)
        row = pad + round(28 * self.scale)
        _text(frame, self.spec.name.upper(), (x, row), 0.58 * self.scale, ACCENT,
              max(1, round(1.4 * self.scale)))

        row += round(44 * self.scale)
        score_text = f"{self.score:.0f}"
        score_scale = 1.3 * self.scale
        _text(frame, score_text, (x, row), score_scale, self._score_color(),
              max(2, round(2.4 * self.scale)))
        (score_width, _), _ = cv.getTextSize(score_text, FONT, score_scale, 2)
        tail = x + score_width + round(8 * self.scale)
        _text(frame, "/100", (tail, row), 0.52 * self.scale, MUTED, max(1, round(self.scale)))
        _text(frame, self.grade, (tail, row - round(22 * self.scale)), 0.5 * self.scale,
              self._score_color(), max(1, round(self.scale)))

        row += round(30 * self.scale)
        reps_label = "-" if state.rep_number is None else str(state.rep_number)
        _text(frame, f"REP {reps_label}/{self.total_reps}", (x, row), 0.52 * self.scale,
              WHITE, max(1, round(self.scale)))
        phase_color = QUALITY_COLORS.get(state.quality or "", MUTED)
        _text(frame, state.phase.upper(), (x + round(96 * self.scale), row),
              0.52 * self.scale, phase_color, max(1, round(self.scale)))

        row += round(24 * self.scale)
        _text(frame, f"t = {state.timestamp:.1f}s", (x, row), 0.45 * self.scale, MUTED, 1)

    def _score_color(self) -> tuple[int, int, int]:
        if self.score >= 75:
            return GOOD
        return WARN if self.score >= 50 else BAD

    def draw_faults(self, frame: np.ndarray, faults: tuple[str, ...]) -> None:
        if not faults:
            return
        pad = round(14 * self.scale)
        chip_height = round(30 * self.scale)
        chip_width = round(300 * self.scale)
        for row, label in enumerate(faults[:4]):
            top = pad + row * (chip_height + round(6 * self.scale))
            left = self.width - pad - chip_width
            _panel(frame, (left, top), (left + chip_width, top + chip_height), alpha=0.7)
            cv.rectangle(frame, (left, top), (left + round(5 * self.scale), top + chip_height),
                         BAD, -1)
            _text(frame, label, (left + round(14 * self.scale), top + round(20 * self.scale)),
                  0.48 * self.scale, WHITE, max(1, round(self.scale)))

    def draw_timeline(self, frame: np.ndarray, timestamp: float) -> None:
        if not self.timeline:
            return
        pad = round(14 * self.scale)
        bar_height = round(16 * self.scale)
        top = self.height - pad - bar_height
        left, right = pad, self.width - pad
        span = right - left
        _panel(frame, (left, top), (right, top + bar_height), alpha=0.55)

        for entry in self.timeline:
            x0 = left + int(span * min(1.0, entry.start_seconds / self.duration))
            x1 = left + int(span * min(1.0, entry.end_seconds / self.duration))
            cv.rectangle(frame, (x0, top), (max(x1, x0 + 2), top + bar_height),
                         TIMELINE_COLORS[entry.color_key], -1)

        playhead = left + int(span * min(1.0, timestamp / self.duration))
        cv.line(frame, (playhead, top - round(4 * self.scale)),
                (playhead, top + bar_height + round(4 * self.scale)), WHITE,
                max(1, round(2 * self.scale)), cv.LINE_AA)

    # -- entry point -------------------------------------------------------
    def render(self, frame: np.ndarray, points: np.ndarray | None, side: Side | None,
               state: FrameState) -> np.ndarray:
        if points is not None:
            self.draw_skeleton(frame, points, side)
            self.draw_angles(frame, points, side, state.angles)
        else:
            _text(frame, "NO POSE DETECTED", (round(20 * self.scale), self.height // 2),
                  0.7 * self.scale, BAD, 2)
        self.draw_hud(frame, state)
        self.draw_faults(frame, state.faults)
        self.draw_timeline(frame, state.timestamp)
        return frame
