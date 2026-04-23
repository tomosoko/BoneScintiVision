"""
BoneScintiVision — EXP-009 訓練スクリプト (YOLO11m, v8データ, 生理的集積比率調整)

EXP-007/008の課題:
  - EXP-007: 腹部Recall=0.828(目標達成)、P=0.707(FP過多)
  - EXP-008: physio mask後処理でもP=0.718止まり
  - 原因仮説: no_physio=70%が高すぎ→モデルが生理的集積を未学習→腎臓・膀胱周辺でFP増加

EXP-009の戦略:
  - データ: yolo_dataset_v8（7000枚, 腹部80%, 生理的集積なし55%←70%から引き下げ）
  - 期待: 生理的集積パターン学習強化でFP抑制→Precision改善（P=0.707→P≥0.800目標）
  - 腹部Recall≥0.800を維持しつつPrecision改善を目指す

v7 との差分:
  - データ: yolo_dataset_v7(生理なし70%) → yolo_dataset_v8(生理なし55%)
  - その他: 同一

使い方:
  python3.12 synth/generate_dataset_v8.py     # データ生成（先に実行）
  python3.12 models/train_detector_v8.py       # 訓練
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

DATA_YAML = BASE_DIR / "data" / "yolo_dataset_v8" / "dataset.yaml"
RUN_NAME = "bone_scinti_detector_v8"


def main():
    from ultralytics import YOLO
    import torch

    if not DATA_YAML.exists():
        print(f"データセットが見つかりません: {DATA_YAML}")
        print("先に python3.12 synth/generate_dataset_v8.py を実行してください。")
        return

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"BoneScintiVision EXP-009 訓練 (device={device})")
    print(f"  データ: {DATA_YAML}")
    print(f"  モデル: yolo11m.pt")
    print(f"  変更点: 生理的集積なし比率 70%→55%（FP抑制のため生理的集積学習強化）")

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
    print(f"\nEXP-009 訓練完了: {BASE_DIR / 'runs' / 'detect' / RUN_NAME}")
    print("次のステップ: python3.12 models/validate_detector_v9.py")


if __name__ == "__main__":
    main()
