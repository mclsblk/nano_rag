from pathlib import Path
from dataclasses import dataclass
import os
import warnings

from pypdf import PdfReader

from agentic_rag.core import Document, DocumentError


@dataclass(frozen=True)
class DocumentLoadReport:
    documents: list[Document]
    skipped: list[str]


class DocumentLoader:
    supported_extensions = {".md", ".txt", ".pdf"}

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

        if file_type not in self.supported_extensions:
            return DocumentLoadReport(documents=[], skipped=[f"Unsupported file type skipped: {source}"])

        if file_type == ".pdf":
            return self._load_pdf_with_report(file_path)

        content = self._read_text(file_path)
        metadata = self._metadata(file_path)

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
            if not content.strip():
                skipped.append(f"Empty PDF page content skipped: {source}#page={page_number}")
                continue

            metadata = self._metadata(file_path)
            metadata["page_count"] = page_count
            metadata["page_number"] = page_number
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

    def _metadata(self, file_path: Path) -> dict[str, object]:
        return {
            "source": self._source_path(file_path),
            "file_name": file_path.name,
            "file_type": file_path.suffix.lower(),
        }

    def _source_path(self, file_path: Path) -> str:
        return Path(os.path.relpath(file_path.resolve(), Path.cwd().resolve())).as_posix()
