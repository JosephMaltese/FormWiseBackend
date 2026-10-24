"""Orchestrates an analysis job from upload to finished report.

Analyses are CPU-bound (pose estimation dominates), so they run in worker
threads with a hard concurrency cap rather than on the event loop -- the
original implementation did this work inside the request coroutine, which
stalled every other request for the duration of the upload.

Moving to Celery later means replacing ``_dispatch`` with a task enqueue and
``InMemoryAnalysisStore`` with a Redis-backed one; ``_execute`` is already a
plain synchronous function that takes nothing but an id.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from app.config import Settings
from app.exercises import resolve_key
from app.schemas.analysis import AnalysisStatus
from app.services.store import AnalysisRecord, AnalysisStore, new_record
from app.video.io import VideoError
from app.video.pipeline import AnalysisError, analyze_video

logger = logging.getLogger(__name__)


class UnknownExerciseError(KeyError):
    """The requested exercise has no analyzer registered."""


class AnalysisService:
    def __init__(self, store: AnalysisStore, settings: Settings) -> None:
        self._store = store
        self._settings = settings
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_analyses)
        self._tasks: set[asyncio.Task[None]] = set()

    @property
    def store(self) -> AnalysisStore:
        return self._store

    def create(self, exercise: str, source_path: Path, source_filename: str | None) -> AnalysisRecord:
        key = resolve_key(exercise)
        if key is None:
            raise UnknownExerciseError(exercise)
        record = new_record(
            exercise=key,
            source_path=source_path,
            output_path=self._settings.output_dir / f"{source_path.stem}.mp4",
            ttl_seconds=self._settings.analysis_ttl_seconds,
            source_filename=source_filename,
        )
        return self._store.save(record)

    def enqueue(self, analysis_id: str) -> None:
        task = asyncio.create_task(self._dispatch(analysis_id))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _dispatch(self, analysis_id: str) -> None:
        async with self._semaphore:
            await asyncio.to_thread(self._execute, analysis_id)

    async def shutdown(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

    # -- worker ------------------------------------------------------------
    def _execute(self, analysis_id: str) -> None:
        record = self._store.get(analysis_id)
        if record is None:
            logger.warning("Analysis %s vanished before it ran", analysis_id)
            return

        self._store.update(analysis_id, status=AnalysisStatus.RUNNING, stage="measuring")

        def progress(stage: str, fraction: float) -> None:
            self._store.update(analysis_id, stage=stage, progress=round(min(1.0, fraction), 3))

        try:
            report = analyze_video(
                source=record.source_path,
                destination=record.output_path,
                exercise=record.exercise,
                settings=self._settings,
                progress=progress,
            )
        except (AnalysisError, VideoError) as exc:
            logger.info("Analysis %s rejected: %s", analysis_id, exc)
            self._store.update(
                analysis_id, status=AnalysisStatus.FAILED, stage="failed", error=str(exc)
            )
        except Exception:  # pragma: no cover - unexpected failure
            logger.exception("Analysis %s crashed", analysis_id)
            self._store.update(
                analysis_id,
                status=AnalysisStatus.FAILED,
                stage="failed",
                error="Analysis failed unexpectedly. Please try another recording.",
            )
        else:
            self._store.update(
                analysis_id,
                status=AnalysisStatus.COMPLETED,
                stage="done",
                progress=1.0,
                report=report,
            )
            logger.info(
                "Analysis %s scored %.1f on %d rep(s)",
                analysis_id, report.score.total, report.reps.attempted,
            )
        finally:
            # The annotated copy is what gets served; the raw upload is not
            # needed once the pipeline has read it.
            record.source_path.unlink(missing_ok=True)
