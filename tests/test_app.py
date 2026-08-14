"""Tests for API app — token auth middleware + exception handler.

锁住两个安全修复：
1. ``NOVEL_AGENT_API_TOKEN`` 设置时，``/api/*`` 必须带正确 token，否则 401。
2. 全局异常处理器只返回通用 ``"Internal server error"``，不把 ``str(exc)``
   （可能含文件路径 / API 响应等内部状态）泄露给客户端。

middleware 直接调用函数而非走 TestClient，避免依赖路由表 / 生产静态文件
catch-all（``/{full_path:path}``）吞掉未匹配路径、以及 ``/api/projects``
触发真实 ProjectManager（ChromaDB 落盘）。
"""

import asyncio
import json
import os
from unittest.mock import patch

from starlette.requests import Request

from novel_agent.api.app import api_token_auth, global_exception_handler

_PASS = object()


def _make_request(path: str, headers: list[tuple[bytes, bytes]]) -> Request:
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "server": ("testserver", 80),
        "client": ("testclient", 50000),
        "root_path": "",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": headers,
    }
    return Request(scope)


async def _run_auth(path: str, headers, token: str):
    async def call_next(request):
        return _PASS

    with patch.dict(os.environ, {"NOVEL_AGENT_API_TOKEN": token}):
        return await api_token_auth(_make_request(path, headers), call_next)


class TestTokenAuth:
    def test_no_token_configured_passes_through(self):
        """未设置 token 时放行（dev-friendly）。"""
        resp = asyncio.run(_run_auth("/api/foo", [], ""))
        assert resp is _PASS

    def test_missing_token_rejected(self):
        resp = asyncio.run(_run_auth("/api/foo", [], "secret"))
        assert resp.status_code == 401
        assert json.loads(resp.body) == {"error": "Unauthorized"}

    def test_wrong_token_rejected(self):
        resp = asyncio.run(_run_auth(
            "/api/foo", [(b"authorization", b"Bearer wrong")], "secret"
        ))
        assert resp.status_code == 401

    def test_correct_bearer_token_accepted(self):
        resp = asyncio.run(_run_auth(
            "/api/foo", [(b"authorization", b"Bearer secret")], "secret"
        ))
        assert resp is _PASS

    def test_correct_api_key_accepted(self):
        resp = asyncio.run(_run_auth(
            "/api/foo", [(b"x-api-key", b"secret")], "secret"
        ))
        assert resp is _PASS

    def test_non_api_path_not_gated(self):
        """静态资源 / 非 api 路径即使 token 设置也不拦截。"""
        resp = asyncio.run(_run_auth("/index.html", [], "secret"))
        assert resp is _PASS


class TestExceptionHandler:
    def test_hides_internal_detail(self):
        resp = asyncio.run(
            global_exception_handler(None, Exception("secret path /tmp/internal.db"))
        )
        body = json.loads(resp.body)
        assert body == {"error": "Internal server error"}
        assert "secret" not in resp.body.decode()
        assert "/tmp/internal.db" not in resp.body.decode()
