"""
BoneScintiVision — EXP-003b 訓練スクリプト (YOLO11m, v3データ, imgsz=512)

EXP-003 (imgsz=640) が MPS index out of bounds で繰り返しクラッシュするため、
EXP-002 と同じ imgsz=512 に戻しつつ v3データセット（腹部強化）を使用。

v2との差分:
  - データ: yolo_dataset_v3 (3540枚, 腹部強化+生理的集積ランダム化) ← 改善
  - モデル: yolo11m.pt (v2と同じ)
  - imgsz: 512 (EXP-003の640→512に戻す)
  - batch: 16 (EXP-003の12→16, EXP-002と同じ)
  - epochs: 150 (patience=30)

使い方:
  python3.12 models/train_detector_v3b.py
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

DATA_YAML = BASE_DIR / "data" / "yolo_dataset_v3" / "dataset.yaml"
RUN_NAME = "bone_scinti_detector_v3b"


def main():
    from ultralytics import YOLO
    import torch

    if not DATA_YAML.exists():
        print(f"データセットが見つかりません: {DATA_YAML}")
        print("先に python3.12 synth/generate_dataset_v3.py を実行してください。")
        return

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"BoneScintiVision EXP-003b 訓練 (device={device})")
    print(f"  データ: {DATA_YAML}")
    print(f"  モデル: yolo11m.pt")
    print(f"  imgsz: 512 (640→512 MPS crash回避)")

    model = YOLO("yolo11m.pt")
    model.train(
        data=str(DATA_YAML),
        epochs=150,
        patience=30,
        imgsz=512,
        batch=16,          # EXP-002と同じ設定（MPS安定実績あり）
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
