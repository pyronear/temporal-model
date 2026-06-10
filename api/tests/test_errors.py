from temporal_model.api.errors import (
    ApiError,
    FrameNotFound,
    InferenceError,
    ModelNotLoaded,
    S3Unavailable,
    Unauthorized,
)


def test_error_codes_and_status():
    assert (FrameNotFound("x").status_code, FrameNotFound("x").code) == (
        404,
        "frame_not_found",
    )
    assert (S3Unavailable("x").status_code, S3Unavailable("x").code) == (
        502,
        "s3_unavailable",
    )
    assert (ModelNotLoaded("x").status_code, ModelNotLoaded("x").code) == (
        503,
        "model_not_loaded",
    )
    assert (InferenceError("x").status_code, InferenceError("x").code) == (
        500,
        "inference_error",
    )


def test_detail_is_carried():
    err = FrameNotFound("missing key cam12/a.jpg")
    assert err.detail == "missing key cam12/a.jpg"
    assert isinstance(err, ApiError)


def test_unauthorized_shape():
    exc = Unauthorized("missing or invalid token")
    assert exc.status_code == 401
    assert exc.code == "unauthorized"
    assert exc.headers == {"WWW-Authenticate": "Bearer"}


def test_apierror_headers_default_none():
    assert ApiError("x").headers is None
