"""HTTP API for chat, calculations, validation, evidence, and exports."""

from __future__ import annotations

import csv
import io
import json
import logging
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from agent.comparison import compare_models
from agent.executor import execute_task
from agent.orchestrator import ConversationOrchestrator, DeterministicProvider
from agent.providers import (
    DeepSeekProvider,
    LLMProvider,
    LLMProviderError,
    LLMProviderOutputError,
    OpenAIProvider,
)
from agent.router import available_parameter_models_for_task, load_model_cards, recommend_models
from database.session import ParameterSetConflictError, Repository, initialize_database
from schemas.domain import (
    CalculationEnvelope,
    CalculationResult,
    ChatRequest,
    ChatResponse,
    ErrorBody,
    ErrorResponse,
    ModelCard,
    ModelComparisonResponse,
    ModelRecommendation,
    ParameterSet,
    RunListResponse,
    RunRecord,
    RunStatus,
    TaskManifest,
    ValidationReport,
)
from thermo_engine.dwsim_export import export_dwsim_flowsheet
from thermo_engine.errors import ThermoEquiError
from thermo_engine.service import validate_equilibrium_result

load_dotenv()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(levelname)s %(message)s")
logger = logging.getLogger("thermoequi.api")
repository = Repository()


def parameter_availability(task: TaskManifest) -> set[str]:
    parameter_sets = repository_parameter_sets()
    return available_parameter_models_for_task(task, parameter_sets)


def repository_parameter_sets() -> list[ParameterSet]:
    return repository.search_parameter_sets(None, [])


def _task_with_repository_parameters(
    task: TaskManifest,
    parameter_sets: list[ParameterSet],
) -> TaskManifest:
    seen = {parameter_set.parameter_set_id for parameter_set in task.parameters}
    return task.model_copy(
        update={
            "parameters": [
                *task.parameters,
                *(parameter_set for parameter_set in parameter_sets if parameter_set.parameter_set_id not in seen),
            ]
        }
    )


def configured_provider() -> LLMProvider:
    provider_name = os.getenv("LLM_PROVIDER", "deterministic").casefold()
    if provider_name == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY", "")
        if api_key:
            return DeepSeekProvider(
                api_key=api_key,
                model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
                base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            )
        logger.warning("LLM_PROVIDER=deepseek but no API key is set; using deterministic provider")
    if provider_name == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "")
        if api_key:
            return OpenAIProvider(api_key=api_key, model=os.getenv("OPENAI_MODEL", "gpt-5-mini"))
        logger.warning("LLM_PROVIDER=openai but no API key is set; using deterministic provider")
    return DeterministicProvider()


orchestrator = ConversationOrchestrator(configured_provider())


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    initialize_database()
    yield


app = FastAPI(
    title="ThermoEqui-Agent API",
    version="0.1.0",
    description="Knowledge-grounded and physically verified phase-equilibrium API.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_context(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    request_id = request.headers.get("X-Request-ID", str(uuid4()))
    request.state.request_id = request_id
    try:
        response = await call_next(request)
    except Exception as exc:
        logger.exception("Unhandled server error", exc_info=exc)
        payload = ErrorResponse(
            error=ErrorBody(
                code="internal_server_error",
                message="Internal server error.",
                request_id=request_id,
            )
        )
        error_response = JSONResponse(status_code=500, content=payload.model_dump(mode="json"))
        error_response.headers["X-Request-ID"] = request_id
        return error_response
    response.headers["X-Request-ID"] = request_id
    return response


@app.exception_handler(ThermoEquiError)
async def thermo_error(request: Request, exc: ThermoEquiError) -> JSONResponse:
    payload = ErrorResponse(
        error=ErrorBody(
            code=exc.detail.failure_type,
            message=exc.detail.message,
            details={**exc.detail.details, "recovery_action": exc.detail.recovery_action},
            request_id=request.state.request_id,
        )
    )
    return JSONResponse(status_code=422, content=payload.model_dump(mode="json"))


@app.exception_handler(LLMProviderError)
async def llm_provider_error(request: Request, exc: LLMProviderError) -> JSONResponse:
    payload = ErrorResponse(
        error=ErrorBody(
            code="external_llm_provider_error",
            message=str(exc),
            details={"provider": exc.provider, "upstream_status": exc.status_code},
            request_id=request.state.request_id,
        )
    )
    return JSONResponse(status_code=502, content=payload.model_dump(mode="json"))


@app.exception_handler(LLMProviderOutputError)
async def llm_provider_output_error(request: Request, exc: LLMProviderOutputError) -> JSONResponse:
    payload = ErrorResponse(
        error=ErrorBody(
            code="external_llm_output_error",
            message=str(exc),
            request_id=request.state.request_id,
        )
    )
    return JSONResponse(status_code=502, content=payload.model_dump(mode="json"))


@app.exception_handler(ValueError)
async def value_error(request: Request, exc: ValueError) -> JSONResponse:
    payload = ErrorResponse(
        error=ErrorBody(
            code="invalid_input",
            message=str(exc),
            request_id=request.state.request_id,
        )
    )
    return JSONResponse(status_code=422, content=payload.model_dump(mode="json"))


@app.exception_handler(ParameterSetConflictError)
async def parameter_set_conflict(request: Request, exc: ParameterSetConflictError) -> JSONResponse:
    payload = ErrorResponse(
        error=ErrorBody(
            code="duplicate_parameter_set",
            message=str(exc),
            request_id=request.state.request_id,
        )
    )
    return JSONResponse(status_code=409, content=payload.model_dump(mode="json"))


@app.exception_handler(RequestValidationError)
async def request_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    payload = ErrorResponse(
        error=ErrorBody(
            code="request_validation_error",
            message="Request payload validation failed.",
            details={"errors": exc.errors()},
            request_id=request.state.request_id,
        )
    )
    return JSONResponse(status_code=422, content=payload.model_dump(mode="json"))


@app.exception_handler(HTTPException)
async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
    payload = ErrorResponse(
        error=ErrorBody(
            code=f"http_{exc.status_code}",
            message=str(exc.detail),
            request_id=request.state.request_id,
        )
    )
    return JSONResponse(status_code=exc.status_code, content=payload.model_dump(mode="json"))


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "version": "0.1.0",
        "llm_provider": type(orchestrator.provider).__name__,
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, request: Request) -> ChatResponse:
    response = await orchestrator.chat(
        body.message,
        body.conversation_id,
        parameter_sets=repository_parameter_sets(),
    )
    response.request_id = request.state.request_id
    payload = response.model_dump(mode="json")
    repository.save_chat_and_run(
        response.conversation_id,
        body.message,
        payload,
        response.calculation,
        request.state.request_id,
    )
    return response


class ParseResponse(BaseModel):
    intent: str
    task: TaskManifest | None


@app.post("/api/tasks/parse", response_model=ParseResponse)
async def parse_task(body: ChatRequest) -> ParseResponse:
    intent, task = await orchestrator.parse(
        body.message,
        body.conversation_id,
        parameter_sets=repository_parameter_sets(),
    )
    return ParseResponse(intent=intent, task=task)


@app.post("/api/models/recommend", response_model=list[ModelRecommendation])
def model_recommendations(task: TaskManifest) -> list[ModelRecommendation]:
    return recommend_models(task, available_parameter_models=parameter_availability(task))


@app.get("/api/models")
def models() -> list[ModelCard]:
    return load_model_cards()


@app.post("/api/parameters", status_code=201)
def create_parameter_set(parameter_set: ParameterSet) -> ParameterSet:
    repository.add_parameter_set(parameter_set)
    return parameter_set


@app.get("/api/parameters/search", response_model=list[ParameterSet])
def search_parameters(model_name: str | None = None, components: list[str] = Query(default=[])) -> list[ParameterSet]:
    return repository.search_parameter_sets(model_name, components)


def execute(task: TaskManifest, request_id: str) -> CalculationEnvelope:
    if task.original_question is None:
        task = task.model_copy(update={"original_question": "Structured API submission"})
    parameter_sets = repository_parameter_sets()
    task = _task_with_repository_parameters(task, parameter_sets)
    envelope = execute_task(
        task,
        available_parameter_models=available_parameter_models_for_task(task, parameter_sets),
    )
    repository.save_run(envelope, request_id)
    return envelope


def _force_type(task: TaskManifest, calculation_type: str) -> TaskManifest:
    equilibrium_type = "FLASH" if calculation_type == "tp_flash" else "VLE"
    return task.model_copy(update={"calculation_type": calculation_type, "equilibrium_type": equilibrium_type})


@app.post("/api/calculations/bubble-point", response_model=CalculationEnvelope)
def bubble_point(task: TaskManifest, request: Request) -> CalculationEnvelope:
    return execute(_force_type(task, "bubble_point"), request.state.request_id)


@app.post("/api/calculations/dew-point", response_model=CalculationEnvelope)
def dew_point(task: TaskManifest, request: Request) -> CalculationEnvelope:
    return execute(_force_type(task, "dew_point"), request.state.request_id)


@app.post("/api/calculations/isobaric-vle", response_model=CalculationEnvelope)
def isobaric_vle(task: TaskManifest, request: Request) -> CalculationEnvelope:
    return execute(_force_type(task, "isobaric_vle"), request.state.request_id)


@app.post("/api/calculations/isothermal-vle", response_model=CalculationEnvelope)
def isothermal_vle(task: TaskManifest, request: Request) -> CalculationEnvelope:
    return execute(_force_type(task, "isothermal_vle"), request.state.request_id)


@app.post("/api/calculations/tp-flash", response_model=CalculationEnvelope)
def tp_flash(task: TaskManifest, request: Request) -> CalculationEnvelope:
    return execute(_force_type(task, "tp_flash"), request.state.request_id)


@app.post("/api/calculations/compare", response_model=ModelComparisonResponse)
def compare_calculations(task: TaskManifest, request: Request) -> ModelComparisonResponse:
    parameter_sets = repository_parameter_sets()
    task = _task_with_repository_parameters(task, parameter_sets)
    return compare_models(
        task,
        available_parameter_models=available_parameter_models_for_task(task, parameter_sets),
    )


@app.post("/api/calculations/azeotrope", response_model=CalculationEnvelope)
def azeotrope(task: TaskManifest, request: Request) -> CalculationEnvelope:
    return execute(_force_type(task, "azeotrope"), request.state.request_id)


@app.post("/api/calculations/lle", response_model=CalculationEnvelope)
def lle(task: TaskManifest, request: Request) -> CalculationEnvelope:
    return execute(
        task.model_copy(update={"calculation_type": "lle", "equilibrium_type": "LLE"}),
        request.state.request_id,
    )


@app.post("/api/calculations/infinite-dilution-activity", response_model=CalculationEnvelope)
def infinite_dilution_activity(task: TaskManifest, request: Request) -> CalculationEnvelope:
    """PGSSI infinite-dilution activity coefficient prediction.

    The first component is the solute and the second the solvent.  Requires the
    PGSSI checkpoint (PGSSI_CHECKPOINT), the PGSSI source tree (PGSSI_SRC), and
    SMILES on every component.
    """
    return execute(
        task.model_copy(
            update={
                "calculation_type": "infinite_dilution_activity",
                "equilibrium_type": "VLE",
                "model_name": "PGSSI",
            }
        ),
        request.state.request_id,
    )

@app.post("/api/calculations/infinite-dilution-activity-ghgeat", response_model=CalculationEnvelope)
def infinite_dilution_activity_ghgeat(task: TaskManifest, request: Request) -> CalculationEnvelope:
    """GHGEAT infinite-dilution activity coefficient prediction.

    The first component is the solute and the second the solvent.  Requires the
    GHGEAT checkpoint (GHGEAT_CHECKPOINT), the GHGEAT source tree (GHGEAT_SRC), and
    SMILES on every component.
    """
    return execute(
        task.model_copy(
            update={
                "calculation_type": "infinite_dilution_activity",
                "equilibrium_type": "VLE",
                "model_name": "GHGEAT",
            }
        ),
        request.state.request_id,
    )


@app.post("/api/validation", response_model=ValidationReport)
def validation(result: CalculationResult) -> ValidationReport:
    return validate_equilibrium_result(result)


@app.get("/api/runs", response_model=RunListResponse)
def list_runs(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    status: RunStatus | None = None,
) -> RunListResponse:
    items, total = repository.list_runs(limit=limit, offset=offset, status=status)
    return RunListResponse(items=items, total=total, limit=limit, offset=offset)


@app.get("/api/runs/{run_id}", response_model=RunRecord)
def get_run(run_id: str) -> RunRecord:
    run = repository.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@app.get("/api/runs/{run_id}/export")
def export_run(run_id: str, format: str = Query(default="json", pattern="^(json|csv|dwsim)$")) -> Response:
    run = repository.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    repository.record_export(run_id, format)
    if format == "dwsim":
        with TemporaryDirectory(prefix="thermoequi-dwsim-") as export_dir:
            path = export_dwsim_flowsheet(run, Path(export_dir) / f"{run_id}.dwxmz")
            dwsim_content = path.read_bytes()
        return Response(
            content=dwsim_content,
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="{run_id}.dwxmz"'},
        )
    if format == "json":
        content = json.dumps(run.model_dump(mode="json"), ensure_ascii=False, indent=2)
        return Response(
            content=content,
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{run_id}.json"'},
        )
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["temperature_K", "pressure_kPa", "x1", "y1", "equilibrium_residual"])
    for point in run.result.get("points", []):
        writer.writerow(
            [
                point["temperature_K"],
                point["pressure_kPa"],
                point["liquid_composition"][0],
                point["vapor_composition"][0],
                point["equilibrium_residual"],
            ]
        )
    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{run_id}.csv"'},
    )
