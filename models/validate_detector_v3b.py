"""
BoneScintiVision — EXP-003b 検証スクリプト

EXP-003b (yolo11m, 512px, yolo_dataset_v3, dual-view) 用。
validate_detector_v2.py と同じロジックを使用し、モデルパスを変更。

使い方:
  cd ~/develop/research/BoneScintiVision
  python3.12 models/validate_detector_v3b.py
  python3.12 models/validate_detector_v3b.py --n 200
"""

import sys
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

V3B_DEFAULT_MODEL = BASE_DIR / "runs" / "detect" / "bone_scinti_detector_v3b" / "weights" / "best.pt"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BoneScintiVision EXP-003b 検証")
    parser.add_argument("--model", default=str(V3B_DEFAULT_MODEL),
                        help="検証するモデルのパス (default: EXP-003b best.pt)")
    parser.add_argument("--n", type=int, default=200,
                        help="テストサンプル数 (default: 200)")
    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"ERROR: モデルが見つかりません: {args.model}")
        print(f"  EXP-003bの訓練が完了しているか確認してください。")
        print(f"  期待パス: {V3B_DEFAULT_MODEL}")
        sys.exit(1)

    # validate_detector_v2 の同じロジックを使用
    from models.validate_detector_v2 import run_validation_v2
    results = run_validation_v2(args.model, n_test=args.n)

    print("\n=== EXP-003b vs EXP-002 比較目標 ===")
    print(f"abdomen_pelvis Recall: EXP-002=0.711 → 目標 0.800+")
    r = results.get("region_recall", {})
    if "abdomen_pelvis" in r:
        achieved = r["abdomen_pelvis"]
        symbol = "✅" if achieved >= 0.800 else "❌"
        print(f"  EXP-003b実績: {achieved:.3f} {symbol}")
    print(f"\n全体 Recall: {results.get('recall', 0):.3f}")
    print(f"全体 Precision: {results.get('precision', 0):.3f}")
