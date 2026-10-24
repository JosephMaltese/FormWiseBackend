"""Discovery endpoint so the client can render exercises and rubrics itself."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.exercises import available_specs, get_analyzer, resolve_key
from app.schemas.analysis import ExerciseOut

router = APIRouter(prefix="/exercises", tags=["exercises"])


def _to_out(key: str) -> ExerciseOut:
    analyzer = get_analyzer(key)
    spec = analyzer.spec
    return ExerciseOut(
        key=spec.key,
        name=spec.name,
        summary=spec.summary,
        camera_guidance=spec.camera_guidance,
        primary_metric=spec.primary_metric,
        primary_metric_label=spec.label_for(spec.primary_metric),
        scoring=[
            {"key": criterion.key, "label": criterion.label, "weight": criterion.weight}
            for criterion in analyzer.build_rubric().criteria
        ],
    )


@router.get("", response_model=list[ExerciseOut], summary="List supported exercises")
async def list_exercises() -> list[ExerciseOut]:
    return [_to_out(spec.key) for spec in available_specs()]


@router.get("/{exercise}", response_model=ExerciseOut, summary="Describe one exercise")
async def get_exercise(exercise: str) -> ExerciseOut:
    key = resolve_key(exercise)
    if key is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"Unknown exercise '{exercise}'.")
    return _to_out(key)
