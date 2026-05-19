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
    loader: str | None = typer.Option(None, "--loader", help="Document load strategy: text, auto, or visual."),
    json_output: bool = typer.Option(False, "--json", help="Output stable JSON."),
) -> None:
    """Ingest documents into the local knowledge base."""
    def command() -> None:
        response = create_indexer(load_strategy=loader).ingest_with_report(path)
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
    engine: str | None = typer.Option(None, "--engine", help="Agentic engine: service or langgraph."),
    debug: bool = typer.Option(False, "--debug", help="Show agentic debug information when --agentic is enabled."),
    json_output: bool = typer.Option(False, "--json", help="Output stable JSON."),
) -> None:
    """Ask a question using retrieval-augmented generation."""
    def command() -> None:
        if agentic:
            service = create_agentic_service(engine=engine)
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
        models = settings.models
        loader = settings.loader
        vectorstore_settings = settings.vectorstore
        keyword = settings.keyword
        search = settings.search
        agentic = settings.agentic
        response = InspectResponse(
            model_provider=models.provider,
            chat_model_provider=models.chat_provider,
            embedding_model_provider=models.embedding_provider,
            ollama_base_url=models.ollama_base_url,
            ollama_chat_model=models.ollama_chat_model,
            ollama_embedding_model=models.ollama_embedding_model,
            ollama_timeout_seconds=models.ollama_timeout_seconds,
            ollama_think=models.ollama_think,
            openai_compatible_base_url=models.openai_compatible_base_url,
            openai_compatible_chat_model=models.openai_compatible_chat_model,
            openai_compatible_embedding_model=models.openai_compatible_embedding_model,
            openai_compatible_visual_model=models.openai_compatible_visual_model,
            openai_compatible_timeout_seconds=models.openai_compatible_timeout_seconds,
            document_load_strategy=loader.load_strategy,
            visual_model_provider=models.visual_provider,
            visual_min_text_chars=loader.visual_min_text_chars,
            chroma_persist_dir=str(vectorstore_settings.chroma_persist_dir),
            chroma_collection=vectorstore_settings.chroma_collection,
            chroma_count=vectorstore.count_chunks(),
            search_strategy=search.strategy,
            keyword_index_path=str(keyword.index_path),
            keyword_source_count=keyword_store.count_sources(),
            keyword_chunk_count=keyword_store.count_chunks(),
            agentic_engine=agentic.engine,
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
