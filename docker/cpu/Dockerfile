FROM --platform=$BUILDPLATFORM python:3.10-slim

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    espeak-ng \
    git \
    libsndfile1 \
    curl \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install uv for speed and glory
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Create non-root user
RUN useradd -m -u 1000 appuser

# Create directories and set ownership
RUN mkdir -p /app/api/src/voices && \
    chown -R appuser:appuser /app

USER appuser
WORKDIR /app

# Copy dependency files
COPY --chown=appuser:appuser pyproject.toml ./pyproject.toml

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv && \
    uv sync --extra cpu --no-install-project

# Copy project files including models
COPY --chown=appuser:appuser api ./api
COPY --chown=appuser:appuser web ./web
COPY --chown=appuser:appuser docker/scripts/download_model.* ./

# Install project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --extra cpu

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV PATH="/app/.venv/bin:$PATH"
ENV UV_LINK_MODE=copy

ENV USE_GPU=false
ENV USE_ONNX=true
ENV DOWNLOAD_ONNX=true
ENV DOWNLOAD_PTH=false

# Download models based on environment variables
RUN if [ "$DOWNLOAD_ONNX" = "true" ]; then \
    python download_model.py --type onnx; \
    fi && \
    if [ "$DOWNLOAD_PTH" = "true" ]; then \
    python download_model.py --type pth; \
    fi

# Run FastAPI server
CMD ["uv", "run", "python", "-m", "uvicorn", "api.src.main:app", "--host", "0.0.0.0", "--port", "8880", "--log-level", "debug"]
