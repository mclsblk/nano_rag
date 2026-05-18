class AgenticRAGError(Exception):
    """Base exception for all project-specific errors."""


class ConfigurationError(AgenticRAGError):
    """Raised when configuration is missing or invalid."""


class DocumentError(AgenticRAGError):
    """Raised when document loading or splitting fails."""


class ModelError(AgenticRAGError):
    """Raised when a model provider call fails."""


class VectorStoreError(AgenticRAGError):
    """Raised when vector store operations fail."""


class SourceConflictError(AgenticRAGError):
    """Raised when ingest would overwrite an existing source."""


class RAGPipelineError(AgenticRAGError):
    """Raised when the RAG pipeline fails."""


class OutputFormatError(AgenticRAGError):
    """Raised when output formatting fails."""
