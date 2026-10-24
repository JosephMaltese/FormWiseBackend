"""FastAPI application factory.

Run with::

    uvicorn app.main:app --reload
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.routes import analyses, exercises, health
from app.config import Settings, get_settings
from app.logging_config import configure_logging
from app.services.analysis_service import AnalysisService
from app.services.store import InMemoryAnalysisStore

logger = logging.getLogger(__name__)

DESCRIPTION = """
Upload a video of one set and get back a 0-100 form score, a per-rep
breakdown, coaching cues and an annotated copy of the video with the tracked
joints and angles drawn on it.

Analysis is asynchronous:

1. `POST /api/v1/analyses` with the video and an exercise key -> `202` + job id
2. `GET /api/v1/analyses/{id}` until `status` is `completed`
3. `GET /api/v1/analyses/{id}/video` for the annotated clip
"""


async def _cleanup_loop(app: FastAPI, settings: Settings) -> None:
    """Drop expired jobs and their videos so disk usage stays bounded."""
    while True:
        try:
            await asyncio.sleep(settings.cleanup_interval_seconds)
            await asyncio.to_thread(app.state.store.purge_expired)
        except asyncio.CancelledError:
            raise
        except Exception:  # pragma: no cover - the sweeper must never die
            logger.exception("Cleanup sweep failed")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    settings.ensure_directories()
    app.state.store = InMemoryAnalysisStore()
    app.state.service = AnalysisService(app.state.store, settings)
    cleanup = asyncio.create_task(_cleanup_loop(app, settings))
    logger.info("FormCoach %s ready (media root: %s)", __version__, settings.media_root.resolve())
    try:
        yield
    finally:
        cleanup.cancel()
        await asyncio.gather(cleanup, return_exceptions=True)
        await app.state.service.shutdown()


def create_app(settings: Settings | None = None) -> FastAPI:
    configure_logging()
    resolved = settings or get_settings()
    app = FastAPI(
        title="FormCoach API",
        description=DESCRIPTION,
        version=__version__,
        lifespan=lifespan,
    )
    app.state.settings = resolved
    app.add_middleware(
        CORSMiddleware,
        allow_origins=resolved.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health.router, prefix=resolved.api_prefix)
    app.include_router(exercises.router, prefix=resolved.api_prefix)
    app.include_router(analyses.router, prefix=resolved.api_prefix)
    return app


app = create_app()
