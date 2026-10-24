"""Submit a video, poll for the report, fetch the annotated clip.

Analysis takes seconds to minutes, so the upload returns ``202 Accepted`` with
a job id instead of holding the connection open. The client polls the status
URL and renders the annotated video once the job completes.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse, Response

from app.api.deps import ServiceDep, SettingsDep, StoreDep
from app.exercises import resolve_key
from app.schemas.analysis import AnalysisCreated, AnalysisEnvelope, AnalysisStatus
from app.services.store import AnalysisRecord
from app.services.uploads import ALLOWED_SUFFIXES, UploadError, save_upload
from app.video.io import VideoError, probe

router = APIRouter(prefix="/analyses", tags=["analyses"])


def _status_url(request: Request, analysis_id: str) -> str:
    return str(request.url_for("get_analysis", analysis_id=analysis_id))


def _video_url(request: Request, analysis_id: str) -> str:
    return str(request.url_for("get_analysis_video", analysis_id=analysis_id))


def to_envelope(record: AnalysisRecord, request: Request) -> AnalysisEnvelope:
    return AnalysisEnvelope(
        id=record.id,
        exercise=record.exercise,
        status=record.status,
        progress=record.progress,
        stage=record.stage,
        created_at=record.created_at,
        updated_at=record.updated_at,
        expires_at=record.expires_at,
        source_filename=record.source_filename,
        annotated_video_url=_video_url(request, record.id) if record.has_video else None,
        report=record.report,
        error=record.error,
    )


@router.post(
    "",
    response_model=AnalysisCreated,
    status_code=status.HTTP_202_ACCEPTED,
    name="create_analysis",
    summary="Upload a set and start a form analysis",
)
async def create_analysis(
    request: Request,
    settings: SettingsDep,
    service: ServiceDep,
    file: UploadFile = File(..., description="A video of a single set."),
    exercise: str = Form(..., description="Exercise key, e.g. 'squat'."),
) -> AnalysisCreated:
    if resolve_key(exercise) is None:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported exercise '{exercise}'. See GET {settings.api_prefix}/exercises.",
        )

    suffix = Path(file.filename or "").suffix.lower()
    source = settings.upload_dir / f"{uuid4().hex}{suffix if suffix in ALLOWED_SUFFIXES else '.mp4'}"
    try:
        await save_upload(
            file,
            source,
            max_bytes=settings.max_upload_bytes,
            allowed_content_types=settings.allowed_content_types,
        )
    except UploadError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    # Reject undecodable or over-long videos now rather than failing a job the
    # client has already been told to wait for.
    try:
        metadata = await asyncio.to_thread(probe, source)
    except VideoError as exc:
        source.unlink(missing_ok=True)
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    if metadata.duration > settings.max_video_seconds:
        source.unlink(missing_ok=True)
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Video is {metadata.duration:.0f}s long; the limit is "
                f"{settings.max_video_seconds:.0f}s. Upload a single set."
            ),
        )

    record = service.create(exercise, source, file.filename)
    service.enqueue(record.id)
    return AnalysisCreated(
        id=record.id,
        exercise=record.exercise,
        status=record.status,
        status_url=_status_url(request, record.id),
        annotated_video_url=_video_url(request, record.id),
    )


@router.get(
    "/{analysis_id}",
    response_model=AnalysisEnvelope,
    name="get_analysis",
    summary="Poll an analysis for progress and, once finished, its report",
)
async def get_analysis(analysis_id: str, request: Request, store: StoreDep) -> AnalysisEnvelope:
    record = store.get(analysis_id)
    if record is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Unknown analysis id.")
    return to_envelope(record, request)


@router.get(
    "/{analysis_id}/video",
    name="get_analysis_video",
    summary="Download the annotated video",
    response_class=FileResponse,
    responses={200: {"content": {"video/mp4": {}}}},
)
async def get_analysis_video(analysis_id: str, store: StoreDep) -> FileResponse:
    record = store.get(analysis_id)
    if record is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Unknown analysis id.")
    if record.status is AnalysisStatus.FAILED:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=record.error or "Analysis failed.")
    if record.status is not AnalysisStatus.COMPLETED:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail=f"Analysis is still {record.status.value}; poll the status URL first.",
        )
    if not record.output_path.exists():
        raise HTTPException(status.HTTP_410_GONE, detail="The annotated video has been cleaned up.")
    return FileResponse(
        record.output_path,
        media_type="video/mp4",
        filename=f"{record.exercise}-annotated.mp4",
    )


@router.delete(
    "/{analysis_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_model=None,
    response_class=Response,
    name="delete_analysis",
    summary="Delete an analysis and its stored video",
)
async def delete_analysis(analysis_id: str, store: StoreDep) -> None:
    if not store.delete(analysis_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Unknown analysis id.")
