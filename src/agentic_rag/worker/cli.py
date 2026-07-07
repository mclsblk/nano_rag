import typer
from redis import Redis
from rq import Queue, Worker

from agentic_rag.factory import create_settings


app = typer.Typer(help="Run Agentic RAG background workers.", invoke_without_command=True)


@app.callback()
def run(
    queue_name: str | None = typer.Option(None, "--queue", help="RQ queue name."),
) -> None:
    settings = create_settings()
    resolved_queue_name = queue_name or settings.queue.rq_queue_name
    redis = Redis.from_url(settings.queue.redis_url)
    worker = Worker([Queue(resolved_queue_name, connection=redis)], connection=redis)
    worker.work()
