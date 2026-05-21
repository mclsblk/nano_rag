from pathlib import Path
from dataclasses import dataclass
import mimetypes
import os
import warnings

from pypdf import PdfReader

from agentic_rag.core import Document, DocumentError
from agentic_rag.models.base import VisionModel


@dataclass(frozen=True)
class DocumentLoadReport:
    documents: list[Document]
    skipped: list[str]


class DocumentLoader:
    supported_extensions = {".md", ".txt", ".pdf"}
    image_extensions = {".png", ".jpg", ".jpeg", ".webp"}
    load_strategies = {"text", "auto", "visual"}

    def __init__(
        self,
        load_strategy: str = "text",
        vision_model: VisionModel | None = None,
        visual_min_text_chars: int = 40,
    ) -> None:
        strategy = _normalize_load_strategy(load_strategy)
        if strategy not in self.load_strategies:
            raise DocumentError(
                f"Unsupported document load strategy: {load_strategy}. Supported strategies: auto, text, visual."
            )
        if visual_min_text_chars < 0:
            raise DocumentError("visual_min_text_chars must be greater than or equal to 0.")

        self.load_strategy = strategy
        self.vision_model = vision_model
        self.visual_min_text_chars = visual_min_text_chars

    def load(self, path: str | Path) -> list[Document]:
        report = self.load_with_report(path)
        for message in report.skipped:
            warnings.warn(message, stacklevel=2)
        return report.documents

    def load_with_report(self, path: str | Path) -> DocumentLoadReport:
        target = Path(path)
        if not target.exists():
            raise DocumentError(f"Document path does not exist: {target}")

        if target.is_file():
            return self._load_file_with_report(target)

        if target.is_dir():
            documents: list[Document] = []
            skipped: list[str] = []
            for file_path in sorted(item for item in target.rglob("*") if item.is_file()):
                report = self._load_file_with_report(file_path)
                documents.extend(report.documents)
                skipped.extend(report.skipped)
            return DocumentLoadReport(documents=documents, skipped=skipped)

        raise DocumentError(f"Document path is not a file or directory: {target}")

    def _load_file_with_report(self, file_path: Path) -> DocumentLoadReport:
        file_type = file_path.suffix.lower()
        source = self._source_path(file_path)

        if file_type not in self.supported_extensions and not self._supports_image_file(file_type):
            return DocumentLoadReport(documents=[], skipped=[f"Unsupported file type skipped: {source}"])

        if file_type == ".pdf":
            return self._load_pdf_with_report(file_path)

        if file_type in self.image_extensions:
            return self._load_image_with_report(file_path)

        content = self._read_text(file_path)
        metadata = self._metadata(file_path)
        self._add_loader_metadata(metadata, visual_parsed=False)

        if not content.strip():
            return DocumentLoadReport(documents=[], skipped=[f"Empty document content skipped: {source}"])

        return DocumentLoadReport(
            documents=[
                Document(
                    id=source,
                    content=content,
                    metadata=metadata,
                )
            ],
            skipped=[],
        )

    def _load_pdf_with_report(self, file_path: Path) -> DocumentLoadReport:
        source = self._source_path(file_path)
        page_texts = self._read_pdf(file_path)
        page_count = len(page_texts)
        documents: list[Document] = []
        skipped: list[str] = []

        for page_number, content in enumerate(page_texts, start=1):
            visual_parsed = False
            if self._should_parse_pdf_page_with_vision(content):
                content = self._extract_pdf_page_text_with_vision(file_path, page_number)
                visual_parsed = True

            if not content.strip():
                skipped.append(f"Empty PDF page content skipped: {source}#page={page_number}")
                continue

            metadata = self._metadata(file_path)
            metadata["page_count"] = page_count
            metadata["page_number"] = page_number
            self._add_loader_metadata(metadata, visual_parsed=visual_parsed)
            documents.append(
                Document(
                    id=f"{source}:page:{page_number:04d}",
                    content=content,
                    metadata=metadata,
                )
            )

        if not documents:
            skipped.append(f"Empty document content skipped: {source}")

        return DocumentLoadReport(documents=documents, skipped=skipped)

    def _load_image_with_report(self, file_path: Path) -> DocumentLoadReport:
        source = self._source_path(file_path)
        content = self._extract_image_text_with_vision(file_path)
        if not content.strip():
            return DocumentLoadReport(documents=[], skipped=[f"Empty document content skipped: {source}"])

        metadata = self._metadata(file_path)
        self._add_loader_metadata(metadata, visual_parsed=True)
        return DocumentLoadReport(
            documents=[
                Document(
                    id=source,
                    content=content,
                    metadata=metadata,
                )
            ],
            skipped=[],
        )

    def _read_text(self, file_path: Path) -> str:
        try:
            return file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return file_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise DocumentError(f"Failed to read text document: {file_path}") from exc

    def _read_pdf(self, file_path: Path) -> list[str]:
        try:
            reader = PdfReader(str(file_path))
            page_texts = [page.extract_text() or "" for page in reader.pages]
        except Exception as exc:
            raise DocumentError(f"Failed to read PDF document: {file_path}") from exc

        return page_texts

    def _should_parse_pdf_page_with_vision(self, content: str) -> bool:
        if self.load_strategy == "visual":
            return True
        if self.load_strategy == "auto":
            return len(content.strip()) < self.visual_min_text_chars
        return False

    def _extract_pdf_page_text_with_vision(self, file_path: Path, page_number: int) -> str:
        image_bytes = self._render_pdf_page_as_png(file_path, page_number)
        return self._extract_text_with_vision(image_bytes, mime_type="image/png")

    def _extract_image_text_with_vision(self, file_path: Path) -> str:
        try:
            image_bytes = file_path.read_bytes()
        except OSError as exc:
            raise DocumentError(f"Failed to read image document: {file_path}") from exc

        mime_type = mimetypes.guess_type(file_path.name)[0] or _mime_type_for_suffix(file_path.suffix.lower())
        return self._extract_text_with_vision(image_bytes, mime_type=mime_type)

    def _extract_text_with_vision(self, image_bytes: bytes, *, mime_type: str) -> str:
        if self.vision_model is None:
            raise DocumentError("Visual document loading requires a configured VisionModel.")
        return self.vision_model.extract_text(image_bytes, mime_type=mime_type)

    def _render_pdf_page_as_png(self, file_path: Path, page_number: int) -> bytes:
        try:
            import fitz
        except ImportError as exc:
            raise DocumentError(
                "PDF visual document loading requires PyMuPDF. Install the visual optional dependency first."
            ) from exc

        try:
            with fitz.open(str(file_path)) as document:
                page = document.load_page(page_number - 1)
                pixmap = page.get_pixmap()
                return pixmap.tobytes("png")
        except Exception as exc:
            raise DocumentError(f"Failed to render PDF page for visual parsing: {file_path}#page={page_number}") from exc

    def _metadata(self, file_path: Path) -> dict[str, object]:
        return {
            "source": self._source_path(file_path),
            "file_name": file_path.name,
            "file_type": file_path.suffix.lower(),
        }

    def _add_loader_metadata(self, metadata: dict[str, object], *, visual_parsed: bool) -> None:
        metadata["loader_strategy"] = self.load_strategy
        metadata["visual_parsed"] = visual_parsed

    def _supports_image_file(self, file_type: str) -> bool:
        return self.load_strategy in {"auto", "visual"} and file_type in self.image_extensions

    def _source_path(self, file_path: Path) -> str:
        return Path(os.path.relpath(file_path.resolve(), Path.cwd().resolve())).as_posix()


def _normalize_load_strategy(load_strategy: str) -> str:
    return load_strategy.strip().lower().replace("-", "_")


def _mime_type_for_suffix(suffix: str) -> str:
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(suffix, "application/octet-stream")
