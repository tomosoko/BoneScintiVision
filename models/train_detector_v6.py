"""
BoneScintiVision — EXP-006 訓練スクリプト (YOLO11m, v5データ, 腹部65%強化)

EXP-005の課題:
  - 腹部Recall=0.757（目標0.800未達）

EXP-006の戦略:
  - データ: yolo_dataset_v5（腹部65%強化+生理的集積なし60%+5000枚）
  - モデル・ハイパーパラメータはEXP-005と同一（変数を一つだけ変える）
  - 期待: 腹部Recall≥0.800達成

v5 との差分:
  - データ: yolo_dataset_v4(3540枚) → yolo_dataset_v5(5000枚)
  - その他: 同一

使い方:
  python3.12 synth/generate_dataset_v5.py     # データ生成（先に実行）
  python3.12 models/train_detector_v6.py       # 訓練
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

DATA_YAML = BASE_DIR / "data" / "yolo_dataset_v5" / "dataset.yaml"
RUN_NAME = "bone_scinti_detector_v6"


def main():
    from ultralytics import YOLO
    import torch

    if not DATA_YAML.exists():
        print(f"データセットが見つかりません: {DATA_YAML}")
        print("先に python3.12 synth/generate_dataset_v5.py を実行してください。")
        return

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"BoneScintiVision EXP-006 訓練 (device={device})")
    print(f"  データ: {DATA_YAML}")
    print(f"  モデル: yolo11m.pt")
    print(f"  変更点: 腹部オーバーサンプリング45%→65%, 生理的集積なし50%→60%, 5000枚")

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
        verbose=False,
    )
    print(f"\nEXP-006 訓練完了: {BASE_DIR / 'runs' / 'detect' / RUN_NAME}")
    print("次のステップ: python3.12 models/validate_detector_v6.py")


if __name__ == "__main__":
    main()
