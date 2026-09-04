# Stage 1: Build frontend
FROM node:22-slim AS frontend-builder

WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Stage 2: Python backend + frontend dist
FROM python:3.12-slim-bookworm

WORKDIR /app

# Install system dependencies for chromadb and onnxruntime
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files
COPY pyproject.toml uv.lock ./
ARG INSTALL_OBSERVABILITY=0
RUN if [ "$INSTALL_OBSERVABILITY" = "1" ]; then \
      uv sync --frozen --no-group dev --extra observability; \
    else \
      uv sync --frozen --no-group dev; \
    fi

# Copy backend source
COPY novel_agent/ novel_agent/

# Copy pre-built frontend dist from builder stage
COPY --from=frontend-builder /app/frontend/dist frontend/dist/

# Create data directories
RUN mkdir -p /app/novel-data /app/traces

ENV PYTHONUNBUFFERED=1
ENV NOVEL_DATA_DIR=/app/novel-data

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "novel_agent.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
