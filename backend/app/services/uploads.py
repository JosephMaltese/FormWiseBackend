"""Validated, streamed handling of uploaded videos."""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import UploadFile

logger = logging.getLogger(__name__)

CHUNK_SIZE = 1024 * 1024
ALLOWED_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".webm", ".mkv"}


class UploadError(ValueError):
    """The upload is not something we are willing to analyse."""


def _suffix(filename: str | None) -> str:
    return Path(filename or "").suffix.lower()


async def save_upload(
    upload: UploadFile,
    destination: Path,
    *,
    max_bytes: int,
    allowed_content_types: set[str],
) -> int:
    """Stream ``upload`` to ``destination``, enforcing type and size limits.

    Written in chunks so a hostile or clumsy upload cannot pull an arbitrary
    number of bytes into memory before the size check runs.
    """
    suffix = _suffix(upload.filename)
    if suffix and suffix not in ALLOWED_SUFFIXES:
        raise UploadError(
            f"Unsupported file type '{suffix}'. Allowed: {', '.join(sorted(ALLOWED_SUFFIXES))}."
        )
    content_type = (upload.content_type or "").split(";")[0].strip().lower()
    if content_type and content_type not in allowed_content_types:
        raise UploadError(f"Unsupported content type '{content_type}'.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    try:
        with destination.open("wb") as buffer:
            while chunk := await upload.read(CHUNK_SIZE):
                written += len(chunk)
                if written > max_bytes:
                    raise UploadError(
                        f"Video exceeds the {max_bytes // (1024 * 1024)} MB upload limit."
                    )
                buffer.write(chunk)
    except UploadError:
        destination.unlink(missing_ok=True)
        raise
    except OSError as exc:
        destination.unlink(missing_ok=True)
        raise UploadError(f"Could not store the upload: {exc}") from exc
    finally:
        await upload.close()

    if written == 0:
        destination.unlink(missing_ok=True)
        raise UploadError("The uploaded file is empty.")
    return written
