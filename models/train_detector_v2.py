"""
BoneScintiVision — YOLO11m デュアルビュー訓練 (EXP-002)

前後面512×256デュアルビュ���データセットでYOLO11mを訓練する。

使い方:
  python3.12 models/train_detector_v2.py
  python3.12 models/train_detector_v2.py --generate  # データセット生成 + 訓練
"""

import os
import sys
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

DATA_DIR   = BASE_DIR / "data" / "yolo_dataset_v2"
YAML_PATH  = DATA_DIR / "dataset.yaml"
MODEL_FILE = BASE_DIR / "yolo11m.pt"
RUN_NAME   = "bone_scinti_detector_v2"


def train(generate: bool = False, epochs: int = 150, batch: int = 16):
    if generate or not YAML_PATH.exists():
        from synth.generate_dataset_v2 import generate_dataset_v2
        print("  データセット生成中...")
        generate_dataset_v2(n_total=2400, out_dir=DATA_DIR)

    from ultralytics import YOLO

    # yolo11m.pt がなければダウンロード
    model_path = str(MODEL_FILE) if MODEL_FILE.exists() else "yolo11m.pt"
    model = YOLO(model_path)

    results = model.train(
        data=str(YAML_PATH),
        epochs=epochs,
        batch=batch,
        imgsz=512,
        rect=True,       # 矩形訓練: 512×256 → アスペクト比維持（パディング最小化）
        device="mps",
        project=str(BASE_DIR / "runs"),
        name=RUN_NAME,
        exist_ok=True,
        # 骨シンチ特化拡張（デュアルビュー対応）
        fliplr=0.0,      # 左右反転禁止（前後面の意味が変わるため）
        flipud=0.0,
        degrees=3.0,     # わずかな回転のみ
        hsv_v=0.3,
        hsv_h=0.0,
        hsv_s=0.0,
        mosaic=0.0,      # モザイク禁止（前後面の対応関係が崩れるため）
        patience=30,
        workers=4,
        verbose=True,
    )

    best_pt = BASE_DIR / "runs" / RUN_NAME / "weights" / "best.pt"
    print(f"\n訓練完了。ベストモデル: {best_pt}")
    return str(best_pt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()
    train(generate=args.generate, epochs=args.epochs, batch=args.batch)
