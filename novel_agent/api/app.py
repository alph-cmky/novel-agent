"""FastAPI application for novel-agent Web UI.

Usage:
    uv run uvicorn novel_agent.api.app:app --host 127.0.0.1 --port 8000
"""

import hmac
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root before any other imports that need env vars
_env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(_env_path)

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from novel_agent.api.routes import router  # noqa: E402


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    # 关掉 AsyncSqliteSaver 的 aiosqlite 连接：其 worker thread 是非守护线程，
    # 不 close 会在进程退出时挂死（见 graph/chapter.py:aclose_checkpointers）。
    from novel_agent.graph.chapter import aclose_checkpointers

    await aclose_checkpointers()


app = FastAPI(title="Novel Agent API", version="0.1.0", lifespan=lifespan)

_cors_origins = [
    o.strip()
    for o in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.middleware("http")
async def api_token_auth(request: Request, call_next):
    """Optional API token gate — only active when ``NOVEL_AGENT_API_TOKEN`` is set.

    Without a token configured, every request passes through unchanged (dev-friendly).
    When set, ``/api/*`` requests must carry the token via ``Authorization: Bearer <t>``
    or ``X-API-Key: <t>``. Static frontend assets are never gated. Compare with a
    constant-time check to avoid leaking the token via response timing.
    """
    token = os.getenv("NOVEL_AGENT_API_TOKEN", "").strip()
    if token and request.url.path.startswith("/api/"):
        auth = request.headers.get("Authorization", "").strip()
        provided = auth[7:] if auth.lower().startswith("bearer ") else auth
        provided = provided or request.headers.get("X-API-Key", "").strip()
        if not hmac.compare_digest(provided, token):
            return JSONResponse(status_code=401, content={"error": "Unauthorized"})
    return await call_next(request)


@app.exception_handler(Exception)
async def global_exception_handler(_request: Request, exc: Exception):
    """Catch-all handler returning JSON errors instead of HTML 500 pages.

    Returns a generic message to the client — exception details (which may include
    file paths, API responses, or other internal state) go to server logs only.
    """
    import traceback
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"},
    )


# Serve frontend static files in production
frontend_dist = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"
is_prod = frontend_dist.exists()

if is_prod:
    app.mount("/assets", StaticFiles(directory=str(frontend_dist / "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        base = frontend_dist.resolve()
        file_path = (frontend_dist / full_path).resolve()
        if full_path and file_path.is_relative_to(base) and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(base / "index.html")

    @app.get("/")
    async def serve_root():
        return FileResponse(frontend_dist / "index.html")
