"""FastAPI application: load a packaged model and serve smoke predictions."""

import json
import logging
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict
from starlette.concurrency import run_in_threadpool

from temporal_model.core.stage_timer import StageTimer, stage_ctx

from .auth import require_token
from .errors import ApiError, InferenceError, InvalidRequest, ModelNotLoaded
from .model_runner import ModelRunner
from .s3 import fetch_frames, make_s3_client
from .schemas import PredictRequest, PredictResponse, to_response
from .settings import settings

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    """Surface ``temporal_model`` INFO logs through uvicorn's console handler.

    Uvicorn configures its own ``uvicorn.*`` loggers but leaves the root logger
    untouched, so application ``INFO`` records (e.g. a calibrator-threshold
    override) are otherwise dropped. Attach uvicorn's handler(s) to the
    ``temporal_model`` package logger.

    Idempotent (safe to call on every lifespan), and propagation is left intact
    so test log-capture (``caplog``) keeps working. No-op outside uvicorn (e.g.
    under ``TestClient``), where ``uvicorn`` has no handlers yet.
    """
    app_logger = logging.getLogger("temporal_model")
    if app_logger.level == logging.NOTSET:
        app_logger.setLevel(logging.INFO)
    for handler in logging.getLogger("uvicorn").handlers:
        if handler not in app_logger.handlers:
            app_logger.addHandler(handler)


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    model_name: str | None = None
    model_version: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    _configure_logging()
    if settings.token:
        logger.info("auth enabled")
    else:
        logger.warning("auth disabled: TEMPORAL_API_TOKEN not set")
    app.state.s3_client = make_s3_client(settings)
    try:
        app.state.runner = ModelRunner.load(
            Path(settings.model_path),
            settings.device,
            settings.calibrator_threshold,
            detection_cache_size=settings.detection_cache_size,
        )
    except Exception as exc:  # noqa: BLE001 — degrade to not-ready, report via /health
        logger.warning("model load failed: %s", exc)
        app.state.runner = None
    yield


app = FastAPI(title="Temporal Model API", version="0.1.0", lifespan=lifespan)


@app.exception_handler(ApiError)
async def _api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "code": exc.code},
        headers=exc.headers,
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


@app.post(
    "/predict",
    response_model=PredictResponse,
    response_model_exclude_unset=True,
    dependencies=[Depends(require_token)],
)
async def predict(
    body: PredictRequest, request: Request, verbose: bool = False
) -> PredictResponse:
    bucket = body.bucket or settings.s3_bucket
    if not bucket:
        raise InvalidRequest(
            "no S3 bucket: set request 'bucket' or TEMPORAL_API_S3_BUCKET"
        )

    runner = getattr(request.app.state, "runner", None)
    if runner is None:
        raise ModelNotLoaded("model is not loaded")
    s3_client = request.app.state.s3_client

    with tempfile.TemporaryDirectory() as tmp:
        try:
            # Timer carries the model device so GPU/MPS stages are synced for
            # honest timing; on the CPU serving target this is a no-op.
            timer = StageTimer(settings.device) if settings.profile else None
            profile: dict | None = {} if settings.profile else None

            with stage_ctx(timer, "s3_fetch"):
                # fetch_frames is blocking boto3 I/O — run it off the event loop.
                paths = await run_in_threadpool(
                    fetch_frames, s3_client, bucket, body.frames, Path(tmp)
                )

            out = await runner.predict(
                paths, roi=body.roi_xyxyn, timer=timer, profile=profile
            )

            profiling = None
            if timer is not None:
                stages = timer.as_dict()
                profiling = {
                    "stages_ms": stages,
                    "total_ms": round(sum(stages.values()), 3),
                    **(profile or {}),
                }
                logger.info("profile %s", json.dumps(profiling))

            return to_response(
                out,
                name=runner.name,
                version=runner.version,
                calibrated=runner.calibrated,
                verbose=verbose,
                threshold_overridden=runner.threshold_overridden,
                packaged_threshold=runner.packaged_threshold,
                profiling=profiling,
            )
        except ApiError:
            raise
        except Exception as exc:  # noqa: BLE001 — surface as inference_error
            raise InferenceError(str(exc)) from exc
