"""FastAPI application: load a packaged model and serve smoke predictions."""

import logging
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from .errors import ApiError, InferenceError, ModelNotLoaded
from .model_runner import ModelRunner
from .s3 import fetch_frames, make_s3_client
from .schemas import PredictRequest, PredictResponse, to_response
from .settings import settings

logger = logging.getLogger(__name__)


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    model_name: str | None = None
    model_version: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.s3_client = make_s3_client(settings)
    try:
        app.state.runner = ModelRunner.load(Path(settings.model_path), settings.device)
    except Exception as exc:  # noqa: BLE001 — degrade to not-ready, report via /health
        logger.warning("model load failed: %s", exc)
        app.state.runner = None
    yield


app = FastAPI(title="Temporal Model API", version="0.1.0", lifespan=lifespan)


@app.exception_handler(ApiError)
async def _api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code, content={"detail": exc.detail, "code": exc.code}
    )


@app.exception_handler(RequestValidationError)
async def _validation_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    errors = exc.errors()
    detail = errors[0]["msg"] if errors else "invalid request"
    return JSONResponse(
        status_code=400, content={"detail": detail, "code": "invalid_request"}
    )


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    runner = getattr(request.app.state, "runner", None)
    if runner is None:
        return HealthResponse(status="unavailable", model_loaded=False)
    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_name=runner.name,
        model_version=runner.version,
    )


@app.post("/predict", response_model=PredictResponse, response_model_exclude_unset=True)
async def predict(
    body: PredictRequest, request: Request, verbose: bool = False
) -> PredictResponse:
    runner = getattr(request.app.state, "runner", None)
    if runner is None:
        raise ModelNotLoaded("model is not loaded")
    s3_client = request.app.state.s3_client

    with tempfile.TemporaryDirectory() as tmp:
        paths = fetch_frames(s3_client, settings.s3_bucket, body.frames, Path(tmp))
        try:
            out = await runner.predict(paths)
        except ApiError:
            raise
        except Exception as exc:  # noqa: BLE001 — surface as inference_error
            raise InferenceError(str(exc)) from exc
        return to_response(
            out,
            name=runner.name,
            version=runner.version,
            calibrated=runner.calibrated,
            verbose=verbose,
        )
