"""Liveness / readiness probe."""
from __future__ import annotations

from fastapi import APIRouter

from app.api.deps import StoreDep
from app.exercises import available_specs
from app.schemas.analysis import HealthOut

router = APIRouter(tags=["meta"])


@router.get("/health", response_model=HealthOut, summary="Service health")
async def health(store: StoreDep) -> HealthOut:
    from app import __version__

    return HealthOut(
        status="ok",
        version=__version__,
        exercises=[spec.key for spec in available_specs()],
        active_analyses=store.count_active(),
    )
