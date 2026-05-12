from pathlib import Path
import os
import warnings

from pypdf import PdfReader

from agentic_rag.core import Document, DocumentError


class DocumentLoader:
    supported_extensions = {".md", ".txt", ".pdf"}

    def load(self, path: str | Path) -> list[Document]:
        target = Path(path)
        if not target.exists():
            raise DocumentError(f"Document path does not exist: {target}")

        if target.is_file():
            return self._load_file(target)

        if target.is_dir():
            documents: list[Document] = []
            for file_path in sorted(item for item in target.rglob("*") if item.is_file()):
                documents.extend(self._load_file(file_path))
            return documents

        raise DocumentError(f"Document path is not a file or directory: {target}")

    def _load_file(self, file_path: Path) -> list[Document]:
        file_type = file_path.suffix.lower()
        source = self._source_path(file_path)

        if file_type not in self.supported_extensions:
            warnings.warn(f"Unsupported file type skipped: {source}", stacklevel=2)
            return []

        if file_type == ".pdf":
            return self._load_pdf(file_path)

        content = self._read_text(file_path)
        metadata = self._metadata(file_path)

        if not content.strip():
            warnings.warn(f"Empty document content skipped: {source}", stacklevel=2)
            return []

        return [
            Document(
                id=source,
                content=content,
                metadata=metadata,
            )
        ]

    def _load_pdf(self, file_path: Path) -> list[Document]:
        source = self._source_path(file_path)
        page_texts = self._read_pdf(file_path)
        page_count = len(page_texts)
        documents: list[Document] = []

        for page_number, content in enumerate(page_texts, start=1):
            if not content.strip():
                warnings.warn(
                    f"Empty PDF page content skipped: {source}#page={page_number}",
                    stacklevel=2,
                )
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
            warnings.warn(f"Empty document content skipped: {source}", stacklevel=2)

        return documents

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
