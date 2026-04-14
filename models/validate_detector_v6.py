"""
BoneScintiVision — EXP-006 検証スクリプト

EXP-006 (yolo11m, 512px, yolo_dataset_v5, 腹部65%強化+5000枚) 用。
validate_detector_v2.py と同じロジックを使用し、モデルパスを変更。

使い方:
  cd /Users/kohei/develop/research/BoneScintiVision
  python3.12 models/validate_detector_v6.py
  python3.12 models/validate_detector_v6.py --n 200
"""

import sys
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

V6_DEFAULT_MODEL = BASE_DIR / "runs" / "detect" / "bone_scinti_detector_v6" / "weights" / "best.pt"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BoneScintiVision EXP-006 検証")
    parser.add_argument("--model", default=str(V6_DEFAULT_MODEL),
                        help="検証するモデルのパス (default: EXP-006 best.pt)")
    parser.add_argument("--n", type=int, default=200,
                        help="テストサンプル数 (default: 200)")
    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"ERROR: モデルが見つかりません: {args.model}")
        print(f"  訓練スクリプト: models/train_detector_v6.py")
        sys.exit(1)

    from models.validate_detector_v2 import run_validation_v2
    results = run_validation_v2(args.model, n_test=args.n)

    print("\n=== EXP-006 vs EXP-005 比較目標 ===")
    print(f"目標: 腹部Recall ≥ 0.800, Precision ≥ 0.900")
    print(f"EXP-005実績: P=0.980, R=0.818, 腹部=0.757")
    r = results.get("region_recall", {})
    overall_p = results.get("precision", 0)
    overall_r = results.get("recall", 0)

    if "abdomen_pelvis" in r:
        abd = r["abdomen_pelvis"]
        abd_sym = "✅" if abd >= 0.800 else "❌"
        print(f"  腹部Recall: {abd:.3f} {abd_sym} (目標0.800, EXP-005=0.757)")
    p_sym = "✅" if overall_p >= 0.900 else "❌"
    r_sym = "✅" if overall_r >= 0.800 else "❌"
    print(f"  Precision: {overall_p:.3f} {p_sym} (目標≥0.900)")
    print(f"  全体Recall: {overall_r:.3f} {r_sym} (目標≥0.800)")
