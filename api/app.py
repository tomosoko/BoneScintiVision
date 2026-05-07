"""
BoneScintiVision — FastAPI スコアリングエンドポイント

訓練済みモデルで骨シンチグラフィ画像の骨転移負荷スコアを返す。

起動:
  cd ~/develop/research/BoneScintiVision
  uvicorn api.app:app --reload --port 8765

エンドポイント:
  POST /score          - 画像アップロード → スコア返却
  GET  /health         - ヘルスチェック
"""

import io
import sys
import logging
import numpy as np
import cv2
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from pydantic import BaseModel

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from models.score_burden import (
    compute_bone_burden_score,
    extract_detections,
    classify_clinical_region,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="BoneScintiVision API",
    description="骨シンチグラフィ hot spot 検出・負荷スコアリング",
    version="0.1.0",
)

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


@app.get("/health")
def health():
    model_ready = MODEL_PATH.exists()
    return {"status": "ok", "model_ready": model_ready}


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
    filename = file.filename or ""

    if _is_dicom(contents, filename):
        try:
            img = _decode_dicom(contents)
        except ImportError:
            raise HTTPException(
                status_code=400,
                detail="DICOM読み込みにはpydicomが必要です: pip install pydicom",
            )
        except Exception as e:
            logger.exception("DICOM decode failed")
            raise HTTPException(
                status_code=400,
                detail=f"DICOMファイルのデコードに失敗しました: {e}",
            )
    else:
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="画像のデコードに失敗しました")

    try:
        model = get_model()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

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
