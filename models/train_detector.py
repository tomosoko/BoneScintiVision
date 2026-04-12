"""
BoneScintiVision — YOLO11s hot spot 検出訓練スクリプト

全身骨シンチグラフィの骨転移hot spotをYOLO11sで検出。

使い方:
  cd /Users/kohei/develop/research/BoneScintiVision
  # Step 1: データセット生成
  python3.12 synth/generate_dataset.py
  # Step 2: 訓練
  python3.12 models/train_detector.py
  # または一発実行:
  python3.12 models/train_detector.py --generate
"""

import os
import sys
import argparse
import time
from pathlib import Path

import torch

BASE_DIR = Path(__file__).parent.parent
DATA_YAML = BASE_DIR / "data" / "yolo_dataset" / "dataset.yaml"
RUNS_DIR  = BASE_DIR / "runs"
RUN_NAME  = "bone_scinti_detector_v1"


def main(args):
    # --- Step 1: データセット生成（オプション） ---
    if args.generate or not DATA_YAML.exists():
        print("[Step 1] データセット生成中...")
        sys.path.insert(0, str(BASE_DIR))
        from synth.generate_dataset import generate_dataset
        DATA_YAML_ACTUAL = generate_dataset(
            n_total=args.n_samples,
            val_ratio=0.15,
            n_workers=args.workers,
        )
    else:
        DATA_YAML_ACTUAL = str(DATA_YAML)
        print(f"[Step 1] 既存データセット使用: {DATA_YAML_ACTUAL}")

    # --- Step 2: YOLO訓練 ---
    from ultralytics import YOLO

    print(f"\n[Step 2] YOLO11s 訓練開始")
    print("=" * 60)
    print(f"  モデル  : yolo11s.pt")
    print(f"  データ  : {DATA_YAML_ACTUAL}")
    print(f"  MPS GPU : {torch.backends.mps.is_available()}")
    print(f"  PyTorch : {torch.__version__}")
    print(f"  Epochs  : {args.epochs}")
    print(f"  Batch   : {args.batch}")
    print(f"  ImgSize : {args.imgsz}")
    print("=" * 60)

    model = YOLO("yolo11s.pt")
    t0 = time.time()

    results = model.train(
        data=DATA_YAML_ACTUAL,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device="mps" if torch.backends.mps.is_available() else "cpu",
        workers=4,
        patience=args.patience,
        project=str(RUNS_DIR),
        name=RUN_NAME,

        # 学習率
        lr0=0.01,
        lrf=0.01,
        warmup_epochs=5,

        # Augmentation（骨シンチ向け: 強い回転は不要、小さな強度変化が重要）
        fliplr=0.5,
        flipud=0.0,      # 上下反転は解剖学的に非現実的
        degrees=5.0,     # 軽微な回転のみ
        translate=0.08,
        scale=0.3,
        mosaic=0.5,      # モザイク適度に（小さいhot spotに有効）
        hsv_v=0.3,       # 輝度変化（集積強度の個人差を模擬）
        hsv_h=0.0,       # グレースケールなのでHue不要
        hsv_s=0.0,

        # その他
        save=True,
        save_period=20,
        pretrained=True,
        verbose=True,
        exist_ok=True,
    )

    elapsed = time.time() - t0
    map50   = results.results_dict.get("metrics/mAP50(B)", "N/A")
    map5095 = results.results_dict.get("metrics/mAP50-95(B)", "N/A")

    best_pt = RUNS_DIR / RUN_NAME / "weights" / "best.pt"
    print()
    print("=" * 60)
    print(f"  訓練完了: {elapsed:.0f}秒 ({elapsed/60:.1f}分)")
    print(f"  mAP50       : {map50}")
    print(f"  mAP50-95    : {map5095}")
    print(f"  ベストモデル : {best_pt}")
    print("=" * 60)
    print()
    print("次のステップ:")
    print(f"  python3.12 models/validate_detector.py --model {best_pt}")

    return str(best_pt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BoneScintiVision YOLO11s訓練")
    parser.add_argument("--generate", action="store_true",
                        help="訓練前にデータセットを再生成する")
    parser.add_argument("--n-samples", type=int, default=1200,
                        help="総サンプル数（--generate 時のみ有効）")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    main(args)
