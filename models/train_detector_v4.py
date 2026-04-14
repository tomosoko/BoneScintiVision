"""
BoneScintiVision — EXP-005 訓練スクリプト (YOLO11m, v4データ, imgsz=512)

EXP-004(アンサンブル)の課題: Precision=0.633, 腹部Recall=0.757 (目標0.800未達)
根本原因: 生理的集積による偽陽性が多発

EXP-005戦略: データ改善でPrecision-Recall両立を狙う
  - データ: yolo_dataset_v4 (生理的集積なし50%, 腹部オーバーサンプリング45%)
  - モデル: yolo11m.pt (EXP-003bと同じ)
  - 設定: imgsz=512, batch=16, epochs=150

目標:
  - Precision ≥ 0.900 (EXP-003b比維持)
  - 腹部Recall ≥ 0.800 (未達課題クリア)

使い方:
  python3.12 models/train_detector_v4.py
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

DATA_YAML = BASE_DIR / "data" / "yolo_dataset_v4" / "dataset.yaml"
RUN_NAME = "bone_scinti_detector_v4"


def main():
    from ultralytics import YOLO
    import torch

    if not DATA_YAML.exists():
        print(f"データセットが見つかりません: {DATA_YAML}")
        print("先に python3.12 synth/generate_dataset_v4.py を実行してください。")
        return

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"BoneScintiVision EXP-005 訓練 (device={device})")
    print(f"  データ: {DATA_YAML}")
    print(f"  モデル: yolo11m.pt")
    print(f"  戦略: 生理的集積なし50% + 腹部オーバーサンプリング45%")

    model = YOLO("yolo11m.pt")
    model.train(
        data=str(DATA_YAML),
        epochs=150,
        patience=30,
        imgsz=512,
        batch=16,
        device=device,
        project=str(BASE_DIR / "runs" / "detect"),
        name=RUN_NAME,
        rect=True,
        fliplr=0.0,
        mosaic=0.0,
        optimizer="AdamW",
        lr0=5e-4,
        lrf=0.01,
        warmup_epochs=3,
        pretrained=True,
    )

    best = BASE_DIR / "runs" / "detect" / RUN_NAME / "weights" / "best.pt"
    print(f"\n訓練完了! Best: {best}")


if __name__ == "__main__":
    main()
