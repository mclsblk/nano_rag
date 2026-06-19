from collections.abc import Iterator
from contextlib import contextmanager

from rich.console import Console
import typer

from agentic_rag.application import ApplicationService
from agentic_rag.core import AgenticRAGError
from agentic_rag.factory import (
    create_collection_service,
    create_file_service,
    create_formatter,
    create_registry_service,
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
        response = ApplicationService().list_files()
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
        response = ApplicationService().list_collections()
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
    debug: bool = typer.Option(False, "--debug", help="Show retrieval diagnostics."),
    json_output: bool = typer.Option(False, "--json", help="Output stable JSON."),
) -> None:
    """Search the local knowledge base."""
    def command() -> None:
        with _status("Searching...", enabled=not json_output):
            if debug:
                response = ApplicationService().search_debug(query, collection_id=collection_id, top_k=top_k)
            else:
                response = ApplicationService().search(query, collection_id=collection_id, top_k=top_k)
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
            if agentic and debug:
                from agentic_rag.factory import create_agentic_service

                response = create_agentic_service(engine=engine, collection_id=collection_id).ask_with_debug(
                    query,
                    top_k=top_k,
                )
            else:
                response = ApplicationService().ask(
                    query,
                    collection_id=collection_id,
                    top_k=top_k,
                    agentic=agentic,
                    engine=engine,
                )
        typer.echo(create_formatter().format_response(response, as_json=json_output, debug=debug))

    _run(command)


@app.command()
def inspect(
    json_output: bool = typer.Option(False, "--json", help="Output stable JSON."),
) -> None:
    """Show current configuration and vector store status."""
    def command() -> None:
        with _status("Inspecting knowledge base...", enabled=not json_output):
            response = ApplicationService().inspect_state()
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
