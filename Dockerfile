FROM python:3.12-slim-bookworm

WORKDIR /app

# Install system dependencies for chromadb and onnxruntime
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files
COPY pyproject.toml .
COPY novel_agent/ novel_agent/

# Install dependencies
RUN uv sync --frozen

# Create data directories
RUN mkdir -p /app/novel-data /app/traces

ENV PYTHONUNBUFFERED=1

# Default command: CLI
ENTRYPOINT ["uv", "run", "novel-agent"]
