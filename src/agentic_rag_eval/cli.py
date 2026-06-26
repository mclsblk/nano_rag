import json
import os
from pathlib import Path

import typer

from agentic_rag_eval.cases import load_cases
from agentic_rag_eval.client import ApiClient
from agentic_rag_eval.metrics import score_case
from agentic_rag_eval.report import build_report


app = typer.Typer(help="External retrieval evaluator for Agentic RAG APIs.")


@app.callback()
def main() -> None:
    """Run external retrieval evaluation commands."""


@app.command("run")
def run(
    cases_path: Path,
    base_url: str = typer.Option("http://127.0.0.1:8000", "--base-url", help="Agentic RAG API base URL."),
    top_k: int = typer.Option(5, "--top-k", help="Default retrieval top_k.", min=1),
    api_key: str = typer.Option("", "--api-key", help="API key. Defaults to RAG_API_KEY."),
    timeout: float = typer.Option(30.0, "--timeout", help="HTTP request timeout seconds.", min=0.1),
    json_out: Path = typer.Option(
        Path("storage/eval/retrieval_report.json"),
        "--json-out",
        help="Path to write JSON report.",
    ),
) -> None:
    cases = load_cases(cases_path)
    client = ApiClient(base_url=base_url, api_key=api_key or os.getenv("RAG_API_KEY", ""), timeout=timeout)
    case_results = []

    for case in cases:
        try:
            search_response, search_latency_ms = client.search(case, top_k)
            debug_response, debug_latency_ms = client.search_debug(case, top_k)
            case_results.append(
                score_case(
                    case,
                    search_response,
                    debug_response,
                    search_latency_ms + debug_latency_ms,
                )
            )
        except Exception as exc:
            case_results.append(score_case(case, error=str(exc)))

    report = build_report(case_results)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = report["summary"]
    typer.echo(
        f"Wrote {json_out} with {summary['case_count']} cases and {summary['error_count']} errors."
    )
