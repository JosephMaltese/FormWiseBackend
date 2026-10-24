"""Where analysis jobs live.

``AnalysisStore`` is the only thing the API knows about persistence. The
in-memory implementation is fine for a single worker; pointing the same
interface at Redis (so several workers, or Celery tasks, can share state) is a
drop-in replacement.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from app.schemas.analysis import AnalysisReport, AnalysisStatus

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class AnalysisRecord:
    id: str
    exercise: str
    status: AnalysisStatus
    created_at: datetime
    updated_at: datetime
    expires_at: datetime
    source_path: Path
    output_path: Path
    source_filename: str | None = None
    progress: float = 0.0
    stage: str = "queued"
    report: AnalysisReport | None = None
    error: str | None = None
    artifacts: tuple[Path, ...] = field(default_factory=tuple)

    @property
    def has_video(self) -> bool:
        return self.status is AnalysisStatus.COMPLETED and self.output_path.exists()


def new_record(
    exercise: str,
    source_path: Path,
    output_path: Path,
    ttl_seconds: int,
    source_filename: str | None = None,
) -> AnalysisRecord:
    created = _now()
    return AnalysisRecord(
        id=uuid4().hex,
        exercise=exercise,
        status=AnalysisStatus.QUEUED,
        created_at=created,
        updated_at=created,
        expires_at=created + timedelta(seconds=ttl_seconds),
        source_path=source_path,
        output_path=output_path,
        source_filename=source_filename,
        artifacts=(source_path, output_path),
    )


class AnalysisStore(Protocol):
    def save(self, record: AnalysisRecord) -> AnalysisRecord: ...

    def get(self, analysis_id: str) -> AnalysisRecord | None: ...

    def update(self, analysis_id: str, **changes: object) -> AnalysisRecord | None: ...

    def delete(self, analysis_id: str) -> bool: ...

    def purge_expired(self) -> int: ...

    def count_active(self) -> int: ...


class InMemoryAnalysisStore:
    """Process-local job store guarded by a lock (workers run in threads)."""

    def __init__(self) -> None:
        self._records: dict[str, AnalysisRecord] = {}
        self._lock = threading.Lock()

    def save(self, record: AnalysisRecord) -> AnalysisRecord:
        with self._lock:
            self._records[record.id] = record
        return record

    def get(self, analysis_id: str) -> AnalysisRecord | None:
        with self._lock:
            return self._records.get(analysis_id)

    def update(self, analysis_id: str, **changes: object) -> AnalysisRecord | None:
        with self._lock:
            current = self._records.get(analysis_id)
            if current is None:
                return None
            updated = replace(current, updated_at=_now(), **changes)
            self._records[analysis_id] = updated
            return updated

    def delete(self, analysis_id: str) -> bool:
        with self._lock:
            record = self._records.pop(analysis_id, None)
        if record is None:
            return False
        _remove_artifacts(record)
        return True

    def purge_expired(self) -> int:
        cutoff = _now()
        with self._lock:
            expired = [
                record for record in self._records.values()
                if record.expires_at <= cutoff and record.status.is_terminal
            ]
            for record in expired:
                self._records.pop(record.id, None)
        for record in expired:
            _remove_artifacts(record)
        if expired:
            logger.info("Purged %d expired analyses", len(expired))
        return len(expired)

    def count_active(self) -> int:
        with self._lock:
            return sum(
                1 for record in self._records.values() if not record.status.is_terminal
            )


def _remove_artifacts(record: AnalysisRecord) -> None:
    for path in record.artifacts:
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:  # pragma: no cover - filesystem edge case
            logger.warning("Could not remove %s: %s", path, exc)
