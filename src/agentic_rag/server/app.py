from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from agentic_rag.application import ApplicationService
from agentic_rag.core import AgenticRAGError
from agentic_rag.server.schemas import AskRequest, HealthResponse, SearchRequest, error_response


def create_app() -> FastAPI:
    api = FastAPI(title="Agentic RAG API", version="0.2.6")

    @api.exception_handler(AgenticRAGError)
    async def handle_agentic_rag_error(_request: Request, exc: AgenticRAGError) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content=error_response(type(exc).__name__, str(exc)),
        )

    @api.exception_handler(Exception)
    async def handle_unexpected_error(_request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content=error_response(type(exc).__name__, str(exc)),
        )

    @api.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse()

    @api.get("/v1/inspect")
    def inspect():
        return ApplicationService().inspect_state()

    @api.post("/v1/search")
    def search(request: SearchRequest):
        return ApplicationService().search(
            request.query,
            collection_id=request.collection_id,
            top_k=request.top_k,
        )

    @api.post("/v1/ask")
    def ask(request: AskRequest):
        return ApplicationService().ask(
            request.query,
            collection_id=request.collection_id,
            top_k=request.top_k,
            agentic=request.agentic,
            engine=request.engine,
        )

    @api.get("/v1/files")
    def list_files():
        return ApplicationService().list_files()

    @api.get("/v1/files/{file_id}")
    def get_file(file_id: str):
        return ApplicationService().get_file(file_id)

    @api.get("/v1/collections")
    def list_collections():
        return ApplicationService().list_collections()

    @api.get("/v1/collections/{collection_id}")
    def get_collection(collection_id: str):
        return ApplicationService().get_collection(collection_id)

    return api


app = create_app()
