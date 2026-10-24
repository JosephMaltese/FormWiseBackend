"""Video reading, writing and codec negotiation."""
from __future__ import annotations

import logging
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import numpy as np

logger = logging.getLogger(__name__)

#: Tried in order. OpenCV wheels are built without H.264 on some platforms, so
#: falling back to mp4v keeps the endpoint working instead of silently writing
#: an empty file (the original implementation's failure mode).
CODEC_PREFERENCE: tuple[str, ...] = ("avc1", "H264", "mp4v")


class VideoError(RuntimeError):
    """Raised when a video cannot be read or written."""


@dataclass(frozen=True)
class VideoMetadata:
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0

    @property
    def aspect(self) -> float:
        return self.width / self.height if self.height else 1.0


def probe(path: Path) -> VideoMetadata:
    capture = cv.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise VideoError("The file could not be decoded as a video.")
    try:
        fps = float(capture.get(cv.CAP_PROP_FPS))
        frame_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()

    # Phone exports occasionally report nonsense metadata; fall back to
    # something sane rather than dividing by zero downstream.
    if not 1.0 <= fps <= 240.0:
        logger.warning("Implausible fps %.2f in %s; assuming 30", fps, path.name)
        fps = 30.0
    if width <= 0 or height <= 0:
        raise VideoError("The video reports no frame dimensions.")
    return VideoMetadata(fps=fps, frame_count=max(frame_count, 0), width=width, height=height)


@contextmanager
def frames(path: Path) -> Iterator[Iterator[tuple[int, np.ndarray]]]:
    """Yield ``(frame_index, bgr_frame)`` for every decodable frame."""
    capture = cv.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise VideoError("The file could not be decoded as a video.")

    def iterator() -> Iterator[tuple[int, np.ndarray]]:
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                return
            yield index, frame
            index += 1

    try:
        yield iterator()
    finally:
        capture.release()


@contextmanager
def writer(path: Path, fps: float, size: tuple[int, int]) -> Iterator[cv.VideoWriter]:
    """Open a video writer, trying each preferred codec in turn."""
    path.parent.mkdir(parents=True, exist_ok=True)
    for codec in CODEC_PREFERENCE:
        handle = cv.VideoWriter(str(path), cv.VideoWriter_fourcc(*codec), fps, size)
        if handle.isOpened():
            logger.debug("Writing %s with codec %s", path.name, codec)
            try:
                yield handle
            finally:
                handle.release()
            return
        handle.release()
    raise VideoError(f"No usable codec among {CODEC_PREFERENCE} for writing {path.name}.")


def transcode_for_web(path: Path, ffmpeg: str | None) -> bool:
    """Re-encode in place to H.264 + faststart so browsers can stream it.

    A no-op (returning ``False``) when ffmpeg is unavailable -- the mp4v
    fallback still plays in Chrome and Safari, just without range requests.
    """
    if not ffmpeg:
        return False
    target = path.with_name(f"{path.stem}.web.mp4")
    command = [
        ffmpeg, "-y", "-loglevel", "error", "-i", str(path),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(target),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, timeout=600)
    except (subprocess.SubprocessError, OSError) as exc:
        logger.warning("ffmpeg transcode failed for %s: %s", path.name, exc)
        target.unlink(missing_ok=True)
        return False
    target.replace(path)
    return True
