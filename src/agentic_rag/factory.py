from agentic_rag.config import Settings, load_settings
from agentic_rag.document import DocumentLoader, TextSplitter
from agentic_rag.models import OllamaChatModel, OllamaEmbeddingModel
from agentic_rag.output import OutputFormatter
from agentic_rag.rag import Generator, Indexer, RAGPipeline, Retriever
from agentic_rag.vectorstore import ChromaVectorStore


def create_settings() -> Settings:
    return load_settings()


def create_vectorstore(settings: Settings | None = None) -> ChromaVectorStore:
    resolved_settings = settings or create_settings()
    embedding_model = OllamaEmbeddingModel(resolved_settings)
    return ChromaVectorStore(embedding_model=embedding_model, settings=resolved_settings)


def create_indexer(settings: Settings | None = None) -> Indexer:
    resolved_settings = settings or create_settings()
    return Indexer(
        loader=DocumentLoader(),
        splitter=TextSplitter(),
        vectorstore=create_vectorstore(resolved_settings),
    )


def create_pipeline(settings: Settings | None = None) -> RAGPipeline:
    resolved_settings = settings or create_settings()
    vectorstore = create_vectorstore(resolved_settings)
    retriever = Retriever(vectorstore)
    generator = Generator(OllamaChatModel(resolved_settings))
    return RAGPipeline(retriever=retriever, generator=generator)


def create_formatter(content_preview_chars: int | None = None) -> OutputFormatter:
    return OutputFormatter(content_preview_chars=content_preview_chars)
