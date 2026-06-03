FROM python:3.11-slim

# OpenCV (pulled in by ultralytics) needs these system libs at runtime;
# the slim base image omits them (libGL.so.1, libgthread-2.0.so.0).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Bring in the core path-dependency and the api package itself.
COPY core/ ./core/
COPY api/ ./api/

WORKDIR /app/api
RUN uv sync --no-dev

# Bake the released model.zip (placed in the build context by `release fetch`)
# to MODEL_PATH.
COPY api/models/model.zip /models/model.zip

EXPOSE 8000
# --no-dev: don't let `uv run` re-sync dev deps into the runtime at startup.
CMD ["uv", "run", "--no-dev", "uvicorn", "temporal_model.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
