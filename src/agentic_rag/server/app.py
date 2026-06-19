from contextlib import asynccontextmanager
import json
import logging
import time
import uuid

from fastapi import BackgroundTasks, FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from agentic_rag.application import ApplicationService
from agentic_rag.core import (
    AgenticRAGError,
    CollectionDeleteResponse,
    CollectionListResponse,
    CollectionResponse,
    DeIngestResponse,
    FileDeleteResponse,
    FileListResponse,
    FileResponse,
    IngestResponse,
    InspectResponse,
    JobListResponse,
    JobResponse,
    PublicAnswerResponse,
    PublicSearchResponse,
    SearchDebugResponse,
)
from agentic_rag.factory import create_job_service, create_settings
from agentic_rag.server.schemas import (
    AskRequest,
    CollectionCreateRequest,
    FilePathImportRequest,
    HealthResponse,
    IngestJobCreateRequest,
    IngestRequest,
    ReadyResponse,
    SearchDebugRequest,
    SearchRequest,
    error_response,
)


LOGGER = logging.getLogger("agentic_rag.api")
PUBLIC_PATHS = {"/health", "/ready", "/docs", "/openapi.json"}


@asynccontextmanager
async def _lifespan(_api: FastAPI):
    create_job_service().mark_interrupted_jobs()
    yield


def create_app() -> FastAPI:
    api = FastAPI(title="Agentic RAG API", version="0.3.2", lifespan=_lifespan)
    settings = create_settings()
    if settings.server.cors_origins:
        api.add_middleware(
            CORSMiddleware,
            allow_origins=settings.server.cors_origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @api.middleware("http")
    async def add_request_context(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        start = time.perf_counter()
        status_code = 500
        try:
            auth_response = _authenticate(request)
            if auth_response is not None:
                status_code = auth_response.status_code
                response = auth_response
            else:
                response = await call_next(request)
                status_code = response.status_code
        finally:
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            LOGGER.info(
                json.dumps(
                    {
                        "request_id": request_id,
                        "method": request.method,
                        "path": request.url.path,
                        "status_code": status_code,
                        "duration_ms": duration_ms,
                    },
                    ensure_ascii=True,
                )
            )
        response.headers["X-Request-ID"] = request_id
        return response

    @api.exception_handler(AgenticRAGError)
    async def handle_agentic_rag_error(_request: Request, exc: AgenticRAGError) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content=error_response(type(exc).__name__, str(exc)),
        )

    @api.exception_handler(RequestValidationError)
    async def handle_validation_error(_request: Request, _exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=error_response("ValidationError", "Invalid request"),
        )

    @api.exception_handler(Exception)
    async def handle_unexpected_error(_request: Request, _exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content=error_response("InternalServerError", "Internal server error"),
        )

    @api.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse()

    @api.get("/ready", response_model=ReadyResponse)
    def ready():
        return ApplicationService().ready()

    @api.get("/v1/inspect", response_model=InspectResponse)
    def inspect():
        return ApplicationService().inspect_state()

    @api.post("/v1/search", response_model=PublicSearchResponse)
    def search(request: SearchRequest):
        return ApplicationService().search_public(
            request.query,
            collection_id=request.collection_id,
            top_k=request.top_k,
        )

    @api.post("/v1/search/debug", response_model=SearchDebugResponse, response_model_exclude_none=True)
    def search_debug(request: SearchDebugRequest):
        return ApplicationService().search_debug(
            request.query,
            collection_id=request.collection_id,
            top_k=request.top_k,
            include_content=request.include_content,
        )

    @api.post("/v1/ask", response_model=PublicAnswerResponse)
    def ask(request: AskRequest):
        return ApplicationService().ask_public(
            request.query,
            collection_id=request.collection_id,
            top_k=request.top_k,
            agentic=request.agentic,
            engine=request.engine,
        )

    @api.get("/v1/files", response_model=FileListResponse)
    def list_files():
        return ApplicationService().list_files()

    @api.post("/v1/files/path", response_model=FileResponse)
    def import_file_path(request: FilePathImportRequest):
        return ApplicationService().import_file_path(request.path)

    @api.post("/v1/files/upload", response_model=FileResponse)
    def upload_file(file: UploadFile = File(...)):
        return ApplicationService().upload_file(file.filename or "", file.file)

    @api.get("/v1/files/{file_id}", response_model=FileResponse)
    def get_file(file_id: str):
        return ApplicationService().get_file(file_id)

    @api.delete("/v1/files/{file_id}", response_model=FileDeleteResponse)
    def delete_file(file_id: str):
        return ApplicationService().delete_file(file_id)

    @api.get("/v1/collections", response_model=CollectionListResponse)
    def list_collections():
        return ApplicationService().list_collections()

    @api.post("/v1/collections", response_model=CollectionResponse)
    def create_collection(request: CollectionCreateRequest):
        return ApplicationService().create_collection(request.name, description=request.description)

    @api.get("/v1/collections/{collection_id}", response_model=CollectionResponse)
    def get_collection(collection_id: str):
        return ApplicationService().get_collection(collection_id)

    @api.delete("/v1/collections/{collection_id}", response_model=CollectionDeleteResponse)
    def delete_collection(collection_id: str):
        return ApplicationService().delete_collection(collection_id)

    @api.post("/v1/collections/{collection_id}/files/{file_id}/ingest", response_model=IngestResponse)
    def ingest_file(collection_id: str, file_id: str, request: IngestRequest | None = None):
        loader = request.loader if request is not None else None
        return ApplicationService().ingest_file(file_id, collection_id, loader=loader)

    @api.delete("/v1/collections/{collection_id}/files/{file_id}", response_model=DeIngestResponse)
    def de_ingest_file(collection_id: str, file_id: str):
        return ApplicationService().de_ingest_file(file_id, collection_id)

    @api.post("/v1/ingest/jobs", response_model=JobResponse)
    def create_ingest_job(request: IngestJobCreateRequest, background_tasks: BackgroundTasks):
        response = ApplicationService().create_ingest_job(
            request.file_id,
            request.collection_id,
            loader=request.loader,
        )
        background_tasks.add_task(ApplicationService().run_ingest_job, response.job.job_id)
        return response

    @api.get("/v1/jobs", response_model=JobListResponse)
    def list_jobs():
        return ApplicationService().list_jobs()

    @api.get("/v1/jobs/{job_id}", response_model=JobResponse)
    def get_job(job_id: str):
        return ApplicationService().get_job(job_id)

    return api


app = create_app()


def _authenticate(request: Request) -> JSONResponse | None:
    api_key = create_settings().server.api_key
    if not api_key or request.url.path in PUBLIC_PATHS:
        return None

    authorization = request.headers.get("Authorization", "")
    bearer_token = authorization.removeprefix("Bearer ").strip() if authorization.startswith("Bearer ") else ""
    provided_key = request.headers.get("X-API-Key", "") or bearer_token
    if provided_key == api_key:
        return None

    return JSONResponse(
        status_code=401,
        content=error_response("Unauthorized", "Invalid or missing API key"),
    )
