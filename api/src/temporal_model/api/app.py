"""FastAPI application.

Scaffold stub: ``/health`` is live; ``/predict`` returns 501 until the model
loading + inference path is migrated from ``temporal_model.core``.
"""

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI(title="Temporal Model API", version="0.1.0")


class PredictRequest(BaseModel):
    frame_paths: list[str]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictRequest) -> JSONResponse:
    return JSONResponse(
        status_code=501,
        content={"detail": "predict not implemented yet (scaffold stub)"},
    )
