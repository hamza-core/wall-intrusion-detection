"""
Video service — file validation and temporary file management.
"""

import uuid
import logging
from pathlib import Path

from fastapi import UploadFile, HTTPException, status

from app.config import settings

logger = logging.getLogger(__name__)


def validate_video_file(file: UploadFile) -> None:
    """
    Raise HTTP 422 if the uploaded file fails basic validation checks.
    """
    if file.filename is None or file.filename.strip() == "":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Uploaded file has no filename.",
        )

    suffix = Path(file.filename).suffix.lower()
    if suffix not in settings.allowed_extensions_set:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Unsupported file extension '{suffix}'. "
                f"Allowed: {', '.join(sorted(settings.allowed_extensions_set))}"
            ),
        )


async def save_upload(file: UploadFile) -> Path:
    """
    Stream the uploaded file to a temporary path in uploads/.

    Returns the path to the saved file.
    Raises HTTP 413 if the file exceeds the configured size limit.
    """
    settings.upload_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(file.filename).suffix.lower()
    temp_name = f"{uuid.uuid4().hex}{suffix}"
    temp_path = settings.upload_dir / temp_name

    total_bytes = 0
    chunk_size = 1024 * 1024  # 1 MB

    try:
        with open(temp_path, "wb") as out:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > settings.max_upload_bytes:
                    out.close()
                    temp_path.unlink(missing_ok=True)
                    limit_mb = settings.max_upload_bytes // (1024 * 1024)
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File exceeds the {limit_mb} MB upload limit.",
                    )
                out.write(chunk)
    except HTTPException:
        raise
    except Exception as exc:
        temp_path.unlink(missing_ok=True)
        logger.exception("Failed to save upload: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save uploaded file.",
        )

    logger.info("Saved upload: %s (%.2f MB)", temp_path.name, total_bytes / 1_048_576)
    return temp_path


def cleanup(path: Path) -> None:
    """Delete a temporary file, logging any error without raising."""
    try:
        path.unlink(missing_ok=True)
        logger.info("Cleaned up: %s", path.name)
    except Exception as exc:
        logger.warning("Could not delete temp file %s: %s", path, exc)
