"""Domain errors mapped to HTTP responses by the FastAPI app.

Each error carries a stable machine-readable ``code`` and a human ``detail``.
The app's exception handler renders them as ``{"detail": ..., "code": ...}``.
"""


class ApiError(Exception):
    """Base class for errors that map to a specific HTTP status + code."""

    status_code: int = 500
    code: str = "error"

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(detail)


class InvalidRequest(ApiError):
    status_code = 400
    code = "invalid_request"


class FrameNotFound(ApiError):
    status_code = 404
    code = "frame_not_found"


class S3Unavailable(ApiError):
    status_code = 502
    code = "s3_unavailable"


class ModelNotLoaded(ApiError):
    status_code = 503
    code = "model_not_loaded"


class InferenceError(ApiError):
    status_code = 500
    code = "inference_error"
