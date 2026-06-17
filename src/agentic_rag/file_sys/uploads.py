from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO
import uuid

from agentic_rag.core import UploadError


@dataclass(frozen=True)
class StagedUpload:
    path: Path
    original_name: str


class UploadService:
    def __init__(self, upload_dir: str | Path, max_upload_mb: int) -> None:
        self.upload_dir = Path(upload_dir)
        self.max_bytes = max_upload_mb * 1024 * 1024
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def stage(self, filename: str, stream: BinaryIO) -> StagedUpload:
        original_name = Path(filename).name
        if not original_name:
            raise UploadError("Upload filename is empty")

        target_path = self.upload_dir / f"upload_{uuid.uuid4().hex}_{original_name}"
        written = 0
        with target_path.open("wb") as target:
            while chunk := stream.read(1024 * 1024):
                written += len(chunk)
                if written > self.max_bytes:
                    target.close()
                    target_path.unlink(missing_ok=True)
                    raise UploadError(f"Upload exceeds max size: {self.max_bytes} bytes")
                target.write(chunk)

        return StagedUpload(path=target_path, original_name=original_name)
