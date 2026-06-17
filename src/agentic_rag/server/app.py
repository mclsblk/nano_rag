from fastapi import BackgroundTasks, FastAPI, File, Request, UploadFile
from fastapi.responses import JSONResponse

from agentic_rag.application import ApplicationService
from agentic_rag.core import AgenticRAGError
from agentic_rag.server.schemas import (
    AskRequest,
    CollectionCreateRequest,
    FilePathImportRequest,
    HealthResponse,
    IngestJobCreateRequest,
    IngestRequest,
    ReadyResponse,
    SearchRequest,
    error_response,
)


def create_app() -> FastAPI:
    api = FastAPI(title="Agentic RAG API", version="0.2.8")

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

    @api.get("/ready", response_model=ReadyResponse)
    def ready():
        return ApplicationService().ready()

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

    @api.post("/v1/files/path")
    def import_file_path(request: FilePathImportRequest):
        return ApplicationService().import_file_path(request.path)

    @api.post("/v1/files/upload")
    def upload_file(file: UploadFile = File(...)):
        return ApplicationService().upload_file(file.filename or "", file.file)

    @api.get("/v1/files/{file_id}")
    def get_file(file_id: str):
        return ApplicationService().get_file(file_id)

    @api.delete("/v1/files/{file_id}")
    def delete_file(file_id: str):
        return ApplicationService().delete_file(file_id)

    @api.get("/v1/collections")
    def list_collections():
        return ApplicationService().list_collections()

    @api.post("/v1/collections")
    def create_collection(request: CollectionCreateRequest):
        return ApplicationService().create_collection(request.name, description=request.description)

    @api.get("/v1/collections/{collection_id}")
    def get_collection(collection_id: str):
        return ApplicationService().get_collection(collection_id)

    @api.delete("/v1/collections/{collection_id}")
    def delete_collection(collection_id: str):
        return ApplicationService().delete_collection(collection_id)

    @api.post("/v1/collections/{collection_id}/files/{file_id}/ingest")
    def ingest_file(collection_id: str, file_id: str, request: IngestRequest | None = None):
        loader = request.loader if request is not None else None
        return ApplicationService().ingest_file(file_id, collection_id, loader=loader)

    @api.delete("/v1/collections/{collection_id}/files/{file_id}")
    def de_ingest_file(collection_id: str, file_id: str):
        return ApplicationService().de_ingest_file(file_id, collection_id)

    @api.post("/v1/ingest/jobs")
    def create_ingest_job(request: IngestJobCreateRequest, background_tasks: BackgroundTasks):
        response = ApplicationService().create_ingest_job(
            request.file_id,
            request.collection_id,
            loader=request.loader,
        )
        background_tasks.add_task(ApplicationService().run_ingest_job, response.job.job_id)
        return response

    @api.get("/v1/jobs")
    def list_jobs():
        return ApplicationService().list_jobs()

    @api.get("/v1/jobs/{job_id}")
    def get_job(job_id: str):
        return ApplicationService().get_job(job_id)

    return api


app = create_app()
