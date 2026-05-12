import typer

from agentic_rag.core import AgenticRAGError
from agentic_rag.factory import create_formatter, create_indexer, create_pipeline, create_settings, create_vectorstore


app = typer.Typer(help="CLI-first local Agentic RAG system.")


@app.command()
def ingest(path: str) -> None:
    """Ingest documents into the local knowledge base."""
    _run(lambda: typer.echo(f"Ingested chunks: {create_indexer().ingest(path)}"))


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
    json_output: bool = typer.Option(False, "--json", help="Output stable JSON."),
) -> None:
    """Ask a question using retrieval-augmented generation."""
    def command() -> None:
        response = create_pipeline().ask(query, top_k=top_k)
        typer.echo(create_formatter().format_response(response, as_json=json_output))

    _run(command)


@app.command()
def inspect() -> None:
    """Show current configuration and vector store status."""
    def command() -> None:
        settings = create_settings()
        vectorstore = create_vectorstore(settings)
        typer.echo(f"ollama_base_url={settings.ollama_base_url}")
        typer.echo(f"ollama_chat_model={settings.ollama_chat_model}")
        typer.echo(f"ollama_embedding_model={settings.ollama_embedding_model}")
        typer.echo(f"ollama_timeout_seconds={settings.ollama_timeout_seconds}")
        typer.echo(f"ollama_think={settings.ollama_think}")
        typer.echo(f"chroma_persist_dir={settings.chroma_persist_dir}")
        typer.echo(f"chroma_collection={settings.chroma_collection}")
        typer.echo(f"chroma_count={vectorstore.collection.count()}")

    _run(command)


def _run(command) -> None:
    try:
        command()
    except AgenticRAGError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


if __name__ == "__main__":
    app()
