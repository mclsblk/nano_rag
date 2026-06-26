import typer
import uvicorn


app = typer.Typer(help="Run the Agentic RAG API server.", invoke_without_command=True)


@app.callback()
def run(
    host: str = typer.Option("127.0.0.1", "--host", help="Host to bind."),
    port: int = typer.Option(8000, "--port", help="Port to bind."),
    reload: bool = typer.Option(False, "--reload", help="Enable uvicorn reload."),
) -> None:
    uvicorn.run("agentic_rag.server.app:app", host=host, port=port, reload=reload)
