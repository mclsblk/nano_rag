import typer

from agentic_rag.core import AgenticRAGError, InspectResponse
from agentic_rag.factory import (
    create_agentic_service,
    create_formatter,
    create_indexer,
    create_keyword_store,
    create_pipeline,
    create_settings,
    create_vectorstore,
)


app = typer.Typer(help="CLI-first local Agentic RAG system.")


@app.command()
def ingest(
    path: str,
    json_output: bool = typer.Option(False, "--json", help="Output stable JSON."),
) -> None:
    """Ingest documents into the local knowledge base."""
    def command() -> None:
        response = create_indexer().ingest_with_report(path)
        typer.echo(create_formatter().format_response(response, as_json=json_output))

    _run(command)


@app.command()
def de_ingest(
    source: str,
    json_output: bool = typer.Option(False, "--json", help="Output stable JSON."),
) -> None:
    """Delete chunks for a source from the local knowledge base."""
    def command() -> None:
        response = create_indexer().de_ingest(source)
        typer.echo(create_formatter().format_response(response, as_json=json_output))

    _run(command)


@app.command()
def search(
    query: str,
    top_k: int = typer.Option(5, "--top-k", help="Number of search results to return.", min=1),
    json_output: bool = typer.Option(False, "--json", help="Output stable JSON."),
) -> None:
    """Search the local knowledge base."""
    def command() -> None:
        response = create_pipeline().search(query, top_k=top_k)
        typer.echo(create_formatter().format_response(response, as_json=json_output))

    _run(command)


@app.command()
def ask(
    query: str,
    top_k: int = typer.Option(5, "--top-k", help="Number of search results to use.", min=1),
    agentic: bool = typer.Option(False, "--agentic", help="Use agentic query rewrite and multi-query retrieval."),
    debug: bool = typer.Option(False, "--debug", help="Show agentic debug information when --agentic is enabled."),
    json_output: bool = typer.Option(False, "--json", help="Output stable JSON."),
) -> None:
    """Ask a question using retrieval-augmented generation."""
    def command() -> None:
        if agentic:
            service = create_agentic_service()
            response = service.ask_with_debug(query, top_k=top_k) if debug else service.ask(query, top_k=top_k)
        else:
            response = create_pipeline(require_gen=True).ask(query, top_k=top_k)
        typer.echo(create_formatter().format_response(response, as_json=json_output))

    _run(command)


@app.command()
def inspect(
    json_output: bool = typer.Option(False, "--json", help="Output stable JSON."),
) -> None:
    """Show current configuration and vector store status."""
    def command() -> None:
        settings = create_settings()
        vectorstore = create_vectorstore(settings)
        keyword_store = create_keyword_store(settings)
        response = InspectResponse(
            model_provider=settings.model_provider,
            chat_model_provider=settings.chat_model_provider,
            embedding_model_provider=settings.embedding_model_provider,
            ollama_base_url=settings.ollama_base_url,
            ollama_chat_model=settings.ollama_chat_model,
            ollama_embedding_model=settings.ollama_embedding_model,
            ollama_timeout_seconds=settings.ollama_timeout_seconds,
            ollama_think=settings.ollama_think,
            openai_compatible_base_url=settings.openai_compatible_base_url,
            openai_compatible_chat_model=settings.openai_compatible_chat_model,
            openai_compatible_embedding_model=settings.openai_compatible_embedding_model,
            openai_compatible_timeout_seconds=settings.openai_compatible_timeout_seconds,
            chroma_persist_dir=str(settings.chroma_persist_dir),
            chroma_collection=settings.chroma_collection,
            chroma_count=vectorstore.count_chunks(),
            search_strategy=settings.search_strategy,
            keyword_index_path=str(settings.keyword_index_path),
            keyword_source_count=keyword_store.count_sources(),
            keyword_chunk_count=keyword_store.count_chunks(),
        )
        typer.echo(create_formatter().format_response(response, as_json=json_output))

    _run(command)


def _run(command) -> None:
    try:
        command()
    except AgenticRAGError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


if __name__ == "__main__":
    app()
