"""
BoneScintiVision — FastAPI スコアリングエンドポイント

訓練済みモデルで骨シンチグラフィ画像の骨転移負荷スコアを返す。

起動:
  cd ~/develop/research/BoneScintiVision
  uvicorn api.app:app --reload --port 8765

エンドポイント:
  POST /score          - 画像アップロード → スコア返却
  POST /score/batch    - 複数画像アップロード → バッチスコア返却
  GET  /health         - ヘルスチェック
"""

import os
import sys
import time
import logging
import collections
import secrets
import numpy as np
import cv2
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, Request, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from models.score_burden import (
    compute_bone_burden_score,
    extract_detections,
    classify_clinical_region,
)

logger = logging.getLogger(__name__)

# ─── 入力制限 ───────────────────────────────────────────────────────────────
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB per file
MAX_BATCH_FILES = 20              # max files per batch request

app = FastAPI(
    title="BoneScintiVision API",
    description="骨シンチグラフィ hot spot 検出・負荷スコアリング",
    version="0.1.0",
)

# ─── CORS ────────────────────────────────────────────────────────────────────
CORS_ORIGINS: list[str] = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


app.add_middleware(RequestLoggingMiddleware)


# ─── Rate Limiting ───────────────────────────────────────────────────────────
RATE_LIMIT_RPM = 60          # requests per minute per client IP
RATE_LIMIT_WINDOW = 60       # sliding window in seconds


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


app.add_middleware(RateLimitMiddleware)


# ─── API Key Authentication ──────────────────────────────────────────────────
API_KEY_ENV = "BONESCINTVISION_API_KEY"


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


app.add_middleware(ApiKeyMiddleware)

MODEL_PATH = BASE_DIR / "runs" / "detect" / "bone_scinti_detector_v8" / "weights" / "best.pt"
_model = None


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(f"モデルが見つかりません: {MODEL_PATH}")
        from ultralytics import YOLO
        _model = YOLO(str(MODEL_PATH))
    return _model


class Detection(BaseModel):
    x: float          # center x (pixels)
    y: float          # center y (pixels)
    w: float          # width (pixels)
    h: float          # height (pixels)
    conf: float       # confidence score
    region: str       # clinical region name


class RiskStage(BaseModel):
    stage: str
    label: str
    n_lesions: int


class ScoreResponse(BaseModel):
    n_lesions: int
    total_bone_burden: float        # % of image area
    bsi_equivalent: float           # BSI相当スコア
    mean_conf: float
    region_scores: dict[str, float]
    risk_stage: RiskStage
    detections: List[Detection]     # 個別検出結果（座標・領域付き）


class BatchItemResponse(BaseModel):
    filename: str
    score: Optional[ScoreResponse] = None
    error: Optional[str] = None


class BatchScoreResponse(BaseModel):
    total: int
    succeeded: int
    failed: int
    results: List[BatchItemResponse]


class HealthResponse(BaseModel):
    status: str
    model_ready: bool
    api_version: str
    model_path: str
    model_experiment: str


@app.get("/health", response_model=HealthResponse)
def health():
    model_ready = MODEL_PATH.exists()
    # Extract experiment directory name (e.g. "bone_scinti_detector_v8")
    model_experiment = MODEL_PATH.parent.parent.name if model_ready else "unknown"
    return {
        "status": "ok",
        "model_ready": model_ready,
        "api_version": app.version,
        "model_path": str(MODEL_PATH.relative_to(BASE_DIR)),
        "model_experiment": model_experiment,
    }


def _is_dicom(contents: bytes, filename: str) -> bool:
    """DICOMファイルかどうかを判定する（マジックバイトまたは拡張子）"""
    # DICOM magic: 128-byte preamble + "DICM" at offset 128
    if len(contents) >= 132 and contents[128:132] == b"DICM":
        return True
    # 拡張子による判定（マジックバイトがないDICOMもある）
    if filename and filename.lower().endswith(".dcm"):
        return True
    return False


def _decode_dicom(contents: bytes) -> np.ndarray:
    """DICOMバイト列からモデル入力用RGB画像を返す"""
    import tempfile
    from synth.dicom_reader import BoneScintiDicom

    with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name
    try:
        ds = BoneScintiDicom(tmp_path)
        img = ds.get_single_view_rgb(size=256)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    return img


def _decode_image(contents: bytes, filename: str) -> np.ndarray:
    """画像バイト列をデコードしてRGB numpy配列を返す。

    DICOM/PNG/JPEG を自動判別する。
    """
    if _is_dicom(contents, filename):
        return _decode_dicom(contents)

    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("画像のデコードに失敗しました")
    return img


def _score_single_image(img: np.ndarray, model, conf: float) -> dict:
    """1枚の画像をスコアリングし、ScoreResponse相当のdictを返す。"""
    results = model(img, verbose=False, conf=conf)
    detections = extract_detections(results[0].boxes)

    img_h, img_w = img.shape[:2]
    score = compute_bone_burden_score(detections, image_w=img_w, image_h=img_h)

    detection_results = [
        {
            **d,
            "region": classify_clinical_region(d["y"] / img_h),
        }
        for d in detections
    ]
    score["detections"] = detection_results
    return score


@app.post("/score", response_model=ScoreResponse)
async def score_image(
    file: UploadFile = File(...),
    conf: float = Query(default=0.25, ge=0.0, le=1.0),
):
    """
    骨シンチグラフィ画像をアップロードしてスコアを取得。

    - **file**: PNG/JPEG/DICOM画像（256×256推奨）
    - **conf**: 検出信頼度しきい値（デフォルト0.25）
    """
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"ファイルサイズが上限を超えています（最大 {MAX_FILE_SIZE // (1024 * 1024)} MB）",
        )
    filename = file.filename or ""

    try:
        img = _decode_image(contents, filename)
    except ImportError:
        raise HTTPException(
            status_code=400,
            detail="DICOM読み込みにはpydicomが必要です: pip install pydicom",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Image decode failed")
        raise HTTPException(
            status_code=400,
            detail=f"画像のデコードに失敗しました: {e}",
        )

    try:
        model = get_model()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return _score_single_image(img, model, conf)


@app.post("/score/batch", response_model=BatchScoreResponse)
async def score_batch(
    files: List[UploadFile] = File(...),
    conf: float = Query(default=0.25, ge=0.0, le=1.0),
):
    """
    複数の骨シンチグラフィ画像をバッチスコアリング。

    - **files**: PNG/JPEG/DICOM画像（複数）
    - **conf**: 検出信頼度しきい値（デフォルト0.25）
    """
    if not files:
        raise HTTPException(status_code=400, detail="ファイルが指定されていません")
    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(
            status_code=413,
            detail=f"バッチ上限を超えています（最大 {MAX_BATCH_FILES} ファイル）",
        )

    try:
        model = get_model()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    results = []
    succeeded = 0
    failed = 0

    for f in files:
        filename = f.filename or "unknown"
        try:
            contents = await f.read()
            if len(contents) > MAX_FILE_SIZE:
                raise ValueError(
                    f"ファイルサイズが上限を超えています（最大 {MAX_FILE_SIZE // (1024 * 1024)} MB）"
                )
            img = _decode_image(contents, filename)
            score = _score_single_image(img, model, conf)
            results.append({"filename": filename, "score": score})
            succeeded += 1
        except Exception as e:
            logger.exception("Batch scoring failed for %s", filename)
            results.append({"filename": filename, "error": str(e)})
            failed += 1

    return {
        "total": len(files),
        "succeeded": succeeded,
        "failed": failed,
        "results": results,
    }
