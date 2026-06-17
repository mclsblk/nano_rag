from collections.abc import Iterator
from contextlib import contextmanager

from rich.console import Console
import typer

from agentic_rag.core import AgenticRAGError, InspectResponse
from agentic_rag.factory import (
    create_agentic_service,
    create_collection_service,
    create_file_service,
    create_formatter,
    create_keyword_store,
    create_pipeline,
    create_registry_service,
    create_settings,
    create_vectorstore,
)


app = typer.Typer(help="CLI-first local Agentic RAG system.")
file_app = typer.Typer(help="Manage file assets.")
collection_app = typer.Typer(help="Manage collections.")
registry_app = typer.Typer(help="Inspect registry records.")
app.add_typer(file_app, name="file")
app.add_typer(collection_app, name="collection")
app.add_typer(registry_app, name="registry")
_console = Console(stderr=True)


@file_app.command("import")
def import_file(
    path: str,
    json_output: bool = typer.Option(False, "--json", help="Output stable JSON."),
) -> None:
    """Import a file into managed storage and return a stable file_id."""
    def command() -> None:
        with _status("Importing file...", enabled=not json_output):
            response = create_file_service().import_file(path)
        typer.echo(create_formatter().format_response(response, as_json=json_output))

    _run(command)


@file_app.command("list")
def list_files(
    json_output: bool = typer.Option(False, "--json", help="Output stable JSON."),
) -> None:
    """List managed files."""
    def command() -> None:
        response = create_file_service().list_files()
        typer.echo(create_formatter().format_response(response, as_json=json_output))

    _run(command)


@file_app.command("delete")
def delete_file(
    file_id: str,
    json_output: bool = typer.Option(False, "--json", help="Output stable JSON."),
) -> None:
    """Delete a managed file after it has no active registry records."""
    def command() -> None:
        with _status("Deleting file...", enabled=not json_output):
            response = create_registry_service().delete_file(file_id)
        typer.echo(create_formatter().format_response(response, as_json=json_output))

    _run(command)


@collection_app.command("create")
def create_collection(
    name: str,
    description: str = typer.Option("", "--description", help="Collection description."),
    json_output: bool = typer.Option(False, "--json", help="Output stable JSON."),
) -> None:
    """Create a collection."""
    def command() -> None:
        response = create_collection_service().create_collection(name, description=description)
        typer.echo(create_formatter().format_response(response, as_json=json_output))

    _run(command)


@collection_app.command("list")
def list_collections(
    json_output: bool = typer.Option(False, "--json", help="Output stable JSON."),
) -> None:
    """List collections."""
    def command() -> None:
        response = create_collection_service().list_collections()
        typer.echo(create_formatter().format_response(response, as_json=json_output))

    _run(command)


@collection_app.command("delete")
def delete_collection(
    collection_id: str,
    json_output: bool = typer.Option(False, "--json", help="Output stable JSON."),
) -> None:
    """Delete a collection after it has no active registry records."""
    def command() -> None:
        with _status("Deleting collection...", enabled=not json_output):
            response = create_registry_service().delete_collection(collection_id)
        typer.echo(create_formatter().format_response(response, as_json=json_output))

    _run(command)


@registry_app.command("list")
def list_registry_records(
    json_output: bool = typer.Option(False, "--json", help="Output stable JSON."),
) -> None:
    """List file-to-collection registry records."""
    def command() -> None:
        response = create_registry_service().list_records()
        typer.echo(create_formatter().format_response(response, as_json=json_output))

    _run(command)


@app.command()
def ingest(
    file_id: str,
    collection_id: str = typer.Option(..., "--collection", help="Target collection_id."),
    loader: str | None = typer.Option(None, "--loader", help="Document load strategy: text, auto, or visual."),
    json_output: bool = typer.Option(False, "--json", help="Output stable JSON."),
) -> None:
    """Ingest a managed file into a collection."""
    def command() -> None:
        with _status("Indexing file...", enabled=not json_output):
            response = create_registry_service().ingest(file_id, collection_id, load_strategy=loader)
        typer.echo(create_formatter().format_response(response, as_json=json_output))

    _run(command)


@app.command()
def de_ingest(
    file_id: str,
    collection_id: str = typer.Option(..., "--collection", help="Target collection_id."),
    json_output: bool = typer.Option(False, "--json", help="Output stable JSON."),
) -> None:
    """De-ingest a managed file from one collection."""
    def command() -> None:
        with _status("De-ingesting file...", enabled=not json_output):
            response = create_registry_service().de_ingest(file_id, collection_id)
        typer.echo(create_formatter().format_response(response, as_json=json_output))

    _run(command)


@app.command()
def search(
    query: str,
    collection_id: str = typer.Option(..., "--collection", help="Target collection_id."),
    top_k: int = typer.Option(5, "--top-k", help="Number of search results to return.", min=1),
    debug: bool = typer.Option(False, "--debug", help="Show full result metadata when --json is enabled."),
    json_output: bool = typer.Option(False, "--json", help="Output stable JSON."),
) -> None:
    """Search the local knowledge base."""
    def command() -> None:
        with _status("Searching...", enabled=not json_output):
            response = create_pipeline(collection_id=collection_id).search(query, top_k=top_k)
        typer.echo(create_formatter().format_response(response, as_json=json_output, debug=debug))

    _run(command)


@app.command()
def ask(
    query: str,
    collection_id: str = typer.Option(..., "--collection", help="Target collection_id."),
    top_k: int = typer.Option(5, "--top-k", help="Number of search results to use.", min=1),
    agentic: bool = typer.Option(False, "--agentic", help="Use agentic query rewrite and multi-query retrieval."),
    engine: str | None = typer.Option(None, "--engine", help="Agentic engine: service or langgraph."),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Show agentic debug information and full result metadata when --json is enabled.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Output stable JSON."),
) -> None:
    """Ask a question using retrieval-augmented generation."""
    def command() -> None:
        with _status("Retrieving and generating answer...", enabled=not json_output):
            if agentic:
                service = create_agentic_service(engine=engine, collection_id=collection_id)
                response = service.ask_with_debug(query, top_k=top_k) if debug else service.ask(query, top_k=top_k)
            else:
                response = create_pipeline(require_gen=True, collection_id=collection_id).ask(query, top_k=top_k)
        typer.echo(create_formatter().format_response(response, as_json=json_output, debug=debug))

    _run(command)


@app.command()
def inspect(
    json_output: bool = typer.Option(False, "--json", help="Output stable JSON."),
) -> None:
    """Show current configuration and vector store status."""
    def command() -> None:
        with _status("Inspecting knowledge base...", enabled=not json_output):
            settings = create_settings()
            vectorstore = create_vectorstore(settings)
            keyword_store = create_keyword_store(settings)
            file_service = create_file_service(settings)
            collection_service = create_collection_service(settings)
            registry = create_registry_service(settings)
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
                file_count=file_service.count_active(),
                collection_count=collection_service.count(),
                registry_record_count=registry.count_records(),
                indexed_chunk_count=registry.count_indexed_chunks(),
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


@contextmanager
def _status(message: str, *, enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return

    with _console.status(message, spinner="dots"):
        yield


if __name__ == "__main__":
    app()
