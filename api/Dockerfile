FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Bring in the core path-dependency and the api package itself.
COPY core/ ./core/
COPY api/ ./api/

WORKDIR /app/api
RUN uv sync --no-dev

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "temporal_model.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
