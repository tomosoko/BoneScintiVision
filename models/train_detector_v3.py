"""
BoneScintiVision — EXP-003 訓練スクリプト (YOLO11m, v3データ)

v2との差分:
  - データ: yolo_dataset_v3 (3540枚, 腹部強化+生理的集積ランダム化)
  - モデル: yolo11m.pt (v2と同じ)
  - imgsz: 512 → 640 (小病変検出向上)
  - epochs: 150 (patience=30)

使い方:
  python3.12 models/train_detector_v3.py
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

DATA_YAML = BASE_DIR / "data" / "yolo_dataset_v3" / "dataset.yaml"
RUN_NAME = "bone_scinti_detector_v3"


def main():
    from ultralytics import YOLO
    import torch

    if not DATA_YAML.exists():
        print(f"データセットが見つかりません: {DATA_YAML}")
        print("先に python3.12 synth/generate_dataset_v3.py を実行してください。")
        return

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"BoneScintiVision EXP-003 訓練 (device={device})")
    print(f"  データ: {DATA_YAML}")
    print(f"  モデル: yolo11m.pt")
    print(f"  imgsz: 640")

    model = YOLO("yolo11m.pt")
    model.train(
        data=str(DATA_YAML),
        epochs=150,
        patience=30,
        imgsz=640,
        batch=12,          # 640px + 12GB VRAM 想定
        device=device,
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
