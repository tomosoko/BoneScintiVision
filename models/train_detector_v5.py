"""
BoneScintiVision — EXP-005 訓練スクリプト (YOLO11m, v4データ, 生理的集積抑制50%)

EXP-004(アンサンブル)の課題:
  - Precision=0.633（過検出: 生理的集積FP多発）
  - 腹部Recall=0.757（目標0.800未達）

EXP-005の戦略:
  - データ: yolo_dataset_v4（生理的集積なし比率 30%→50%）
  - モデル・ハイパーパラメータはEXP-003bと同一（変数を一つだけ変える）
  - 期待: Precisionの改善（FP減少）、Recallは維持 or 微増

v3b との差分:
  - データ: yolo_dataset_v3 → yolo_dataset_v4
  - その他: 同一

使い方:
  python3.12 synth/generate_dataset_v4.py     # データ生成（先に実行）
  python3.12 models/train_detector_v5.py       # 訓練
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

DATA_YAML = BASE_DIR / "data" / "yolo_dataset_v4" / "dataset.yaml"
RUN_NAME = "bone_scinti_detector_v5"


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
    print(f"  変更点: 生理的集積なし比率 30%→50%（FP抑制）")

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
    print("次: python3.12 models/validate_detector_v3b.py --model runs/detect/bone_scinti_detector_v5/weights/best.pt")


if __name__ == "__main__":
    main()
