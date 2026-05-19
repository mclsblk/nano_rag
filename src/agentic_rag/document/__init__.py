from agentic_rag.config import Settings
from agentic_rag.core import ConfigurationError
from agentic_rag.document.chunker import SemanticChunker
from agentic_rag.document.loader import DocumentLoader
from agentic_rag.document.splitter import TextSplitter
from agentic_rag.models import EmbeddingModel, VisionModel


def create_document_loader(
    settings: Settings,
    load_strategy: str | None = None,
    vision_model: VisionModel | None = None,
) -> DocumentLoader:
    loader_settings = settings.loader
    return DocumentLoader(
        load_strategy=(load_strategy or loader_settings.load_strategy).strip().lower().replace("-", "_"),
        vision_model=vision_model,
        visual_min_text_chars=loader_settings.visual_min_text_chars,
    )


def create_document_chunker(settings: Settings, embedding_model: EmbeddingModel | None = None):
    chunking = settings.chunking
    strategy = chunking.strategy.strip().lower().replace("-", "_")

    if strategy == "character":
        return TextSplitter(
            chunk_size=chunking.size_chars,
            chunk_overlap=chunking.overlap_chars,
        )
    if strategy == "semantic":
        return SemanticChunker(
            chunk_size=chunking.size_chars,
            chunk_overlap=chunking.overlap_chars,
            chunk_min_chars=chunking.min_chars,
            breakpoint_threshold=chunking.semantic_breakpoint_threshold,
            page_merge_min_score=chunking.semantic_page_merge_min_score,
            max_units_per_chunk=chunking.semantic_max_units_per_chunk,
            embedding_model=embedding_model,
        )

    raise ConfigurationError(
        f"Unsupported chunk strategy: {chunking.strategy}. Supported strategies: character, semantic."
    )
