"""FastAPI application for novel-agent Web UI.

Usage:
    uv run uvicorn novel_agent.api.app:app --host 0.0.0.0 --port 8000
"""

from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root before any other imports that need env vars
_env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(_env_path)

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from novel_agent.api.routes import router  # noqa: E402

app = FastAPI(title="Novel Agent API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")

# Serve frontend static files in production
frontend_dist = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"
is_prod = frontend_dist.exists()

if is_prod:
    app.mount("/assets", StaticFiles(directory=str(frontend_dist / "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        file_path = frontend_dist / full_path
        if full_path and file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(frontend_dist / "index.html")

    @app.get("/")
    async def serve_root():
        return FileResponse(frontend_dist / "index.html")
