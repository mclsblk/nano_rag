from agentic_rag.application import ApplicationService


def run_ingest_job(job_id: str) -> None:
    ApplicationService().run_ingest_job(job_id)
