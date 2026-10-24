"""Turns faults and rubric results into coaching feedback.

Deterministic and testable -- a score with no explanation is not usable
feedback, and a lifter should get the same advice for the same video every
time. (An LLM summary, if one is ever added, belongs on top of these cues
rather than in place of them.)
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from app.core.evaluation import RepAnalysis
from app.core.faults import Severity
from app.core.scoring import ScoreBreakdown

LOW_CRITERION_RATIO = 0.65
STRENGTH_RATIO = 0.9
LOW_DETECTION_RATE = 0.6


@dataclass(frozen=True)
class Cue:
    code: str
    message: str
    severity: str
    affected_reps: tuple[int, ...] = ()


def _plural(reps: Sequence[int]) -> str:
    if len(reps) == 1:
        return f"rep {reps[0]}"
    if len(reps) == 2:
        return f"reps {reps[0]} and {reps[1]}"
    return "reps " + ", ".join(str(rep) for rep in reps[:-1]) + f" and {reps[-1]}"


def build_cues(
    reps: Sequence[RepAnalysis],
    breakdown: ScoreBreakdown,
    detection_rate: float,
    camera_guidance: str,
) -> tuple[list[Cue], list[str]]:
    """Return ``(cues, strengths)`` ordered by how much they cost the lifter."""
    cues: list[Cue] = []

    if detection_rate < LOW_DETECTION_RATE:
        cues.append(
            Cue(
                code="low_visibility",
                message=(
                    f"Your body was only fully visible in {detection_rate:.0%} of frames, "
                    f"so these numbers are rough. {camera_guidance}"
                ),
                severity="major" if detection_rate < 0.3 else "minor",
            )
        )

    if not reps:
        cues.append(
            Cue(
                code="no_reps",
                message=(
                    "No complete reps were detected. Record the full set from start to "
                    f"finish with your whole body in frame. {camera_guidance}"
                ),
                severity="major",
            )
        )
        return cues, []

    grouped: dict[str, list[RepAnalysis]] = defaultdict(list)
    text: dict[str, tuple[str, Severity]] = {}
    for analysis in reps:
        for fault in analysis.faults:
            grouped[fault.code].append(analysis)
            text[fault.code] = (fault.cue, fault.severity)

    for code, affected in sorted(grouped.items(), key=lambda item: -len(item[1])):
        cue_text, severity = text[code]
        numbers = tuple(analysis.number for analysis in affected)
        share = len(affected) / len(reps)
        scope = "Every rep" if share == 1.0 else _plural(numbers).capitalize()
        cues.append(
            Cue(
                code=code,
                message=f"{scope}: {cue_text}",
                severity=severity.value,
                affected_reps=numbers,
            )
        )

    covered = set(grouped)
    for result in breakdown.criteria:
        already_said = result.key in covered or covered.intersection(result.covers)
        if result.ratio >= LOW_CRITERION_RATIO or already_said:
            continue
        cues.append(
            Cue(
                code=result.key,
                message=f"{result.label} cost you {result.weight - result.earned:.0f} "
                f"points. {result.detail}",
                severity="minor",
            )
        )

    strengths = [
        f"{result.label}: {result.detail}"
        for result in breakdown.criteria
        if result.ratio >= STRENGTH_RATIO
    ]
    return cues, strengths
