"""
BoneScintiVision — ミドルウェアモジュール

FastAPI ミドルウェア群を集約:
  - RequestLoggingMiddleware: リクエストログ
  - RateLimitMiddleware: IP ベースレートリミット
  - ApiKeyMiddleware: API キー認証
"""

import collections
import logging
import os
import secrets
import time

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

# ─── Rate Limit defaults ─────────────────────────────────────────────────────
RATE_LIMIT_RPM = 60          # requests per minute per client IP
RATE_LIMIT_WINDOW = 60       # sliding window in seconds

# ─── API Key env var name ─────────────────────────────────────────────────────
API_KEY_ENV = "BONESCINTVISION_API_KEY"


# ─── Request Logging ─────────────────────────────────────────────────────────
class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """全リクエストのメソッド・パス・ステータス・処理時間をログ出力する。"""

    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "%s %s %d %.1fms",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response


# ─── Rate Limiting ───────────────────────────────────────────────────────────
class RateLimitMiddleware(BaseHTTPMiddleware):
    """IPベースのスライディングウィンドウ・レートリミッター。

    ``RATE_LIMIT_RPM`` リクエスト/分 を超えたクライアントに 429 を返す。
    期限切れのIPエントリは自動的にクリーンアップされる。
    """

    # Module-level shared state for easy test reset
    _hits: dict[str, collections.deque] = {}

    # Cleanup runs every _CLEANUP_INTERVAL requests to remove stale IP entries
    _CLEANUP_INTERVAL = 100
    _request_count: int = 0

    def __init__(self, app, rpm: int = RATE_LIMIT_RPM, window: int = RATE_LIMIT_WINDOW):
        super().__init__(app)
        self.rpm = rpm
        self.window = window

    def _client_ip(self, request: Request) -> str:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _cleanup_stale_entries(self, now: float) -> None:
        """Remove IP keys whose deques are empty or fully expired."""
        stale = [
            ip for ip, q in self._hits.items()
            if not q or q[-1] <= now - self.window
        ]
        for ip in stale:
            del self._hits[ip]

    # Paths exempt from rate limiting (lightweight endpoints)
    EXEMPT_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}

    async def dispatch(self, request: Request, call_next):
        if request.url.path in self.EXEMPT_PATHS:
            return await call_next(request)

        ip = self._client_ip(request)
        now = time.monotonic()

        # Periodic cleanup of stale IP entries to prevent memory leak
        RateLimitMiddleware._request_count += 1
        if RateLimitMiddleware._request_count >= self._CLEANUP_INTERVAL:
            RateLimitMiddleware._request_count = 0
            self._cleanup_stale_entries(now)

        q = self._hits.setdefault(ip, collections.deque())

        # expire old timestamps
        while q and q[0] <= now - self.window:
            q.popleft()

        if len(q) >= self.rpm:
            return JSONResponse(
                status_code=429,
                content={"detail": f"レートリミット超過（最大 {self.rpm} req/{self.window}s）"},
                headers={"Retry-After": str(self.window)},
            )

        q.append(now)
        return await call_next(request)


# ─── API Key Authentication ──────────────────────────────────────────────────
class ApiKeyMiddleware(BaseHTTPMiddleware):
    """APIキー認証ミドルウェア。

    環境変数 ``BONESCINTVISION_API_KEY`` が設定されている場合、
    ``X-API-Key`` ヘッダーによる認証を要求する。
    未設定時は全リクエストを許可する（開発モード）。
    """

    EXEMPT_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}

    async def dispatch(self, request: Request, call_next):
        expected_key = os.environ.get(API_KEY_ENV, "")
        if not expected_key:
            # 開発モード: APIキー未設定 → 認証スキップ
            return await call_next(request)

        if request.url.path in self.EXEMPT_PATHS:
            return await call_next(request)

        provided_key = request.headers.get("x-api-key", "")
        if not provided_key or not secrets.compare_digest(provided_key, expected_key):
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key"},
            )

        return await call_next(request)
