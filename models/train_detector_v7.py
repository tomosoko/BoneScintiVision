"""
BoneScintiVision — EXP-007 訓練スクリプト (YOLO11m, v7データ, 7000枚+腹部80%+生理70%)

EXP-006の課題:
  - 腹部Recall=0.759（目標0.800未達, EXP-006比+0.2pp）

EXP-007の戦略:
  - データ: yolo_dataset_v7（6000枚, 腹部病変60%, 生理的集積なし50%）
  - モデル・ハイパーパラメータはEXP-005と同一（変数を一つだけ変える）
  - 期待: 腹部OS 80% + no-physio 70% + データ量増加で腹部Recall≥0.800突破

v5 との差分:
  - データ: yolo_dataset_v5(5000枚,腹部65%) → yolo_dataset_v7(6000枚,腹部60%)
  - その他: 同一

使い方:
  python3.12 synth/generate_dataset_v7.py     # データ生成（先に実行）
  python3.12 models/train_detector_v7.py       # 訓練
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

DATA_YAML = BASE_DIR / "data" / "yolo_dataset_v7" / "dataset.yaml"
RUN_NAME = "bone_scinti_detector_v7"


def main():
    from ultralytics import YOLO
    import torch

    if not DATA_YAML.exists():
        print(f"データセットが見つかりません: {DATA_YAML}")
        print("先に python3.12 synth/generate_dataset_v7.py を実行してください。")
        return

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"BoneScintiVision EXP-007 訓練 (device={device})")
    print(f"  データ: {DATA_YAML}")
    print(f"  モデル: yolo11m.pt")
    print(f"  変更点: 6000枚 + 腹部オーバーサンプリング60%（腹部Recall↑）")

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
    print(f"\nEXP-007 訓練完了: {BASE_DIR / 'runs' / 'detect' / RUN_NAME}")
    print("次のステップ: python3.12 models/validate_detector_v6.py")


if __name__ == "__main__":
    main()
