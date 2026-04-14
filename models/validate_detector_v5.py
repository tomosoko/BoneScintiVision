"""
BoneScintiVision — EXP-005 検証スクリプト

EXP-005 (yolo11m, 512px, yolo_dataset_v4, 生理的集積なし50%) 用。
validate_detector_v2.py と同じロジックを使用し、モデルパスを変更。

EXP-005 目標:
  - Precision ≥ 0.900 (EXP-004アンサンブル Precision=0.633 から改善)
  - 腹部 Recall ≥ 0.800 (EXP-003b abdomen=0.724 から改善)
  - 全体 Recall ≥ 0.800 維持

使い方:
  cd ~/develop/research/BoneScintiVision
  python3.12 models/validate_detector_v5.py
  python3.12 models/validate_detector_v5.py --n 200
  python3.12 models/validate_detector_v5.py --model runs/detect/bone_scinti_detector_v5/weights/best.pt
"""

import sys
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

V5_DEFAULT_MODEL = BASE_DIR / "runs" / "detect" / "bone_scinti_detector_v5" / "weights" / "best.pt"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BoneScintiVision EXP-005 検証")
    parser.add_argument("--model", default=str(V5_DEFAULT_MODEL),
                        help="検証するモデルのパス (default: EXP-005 best.pt)")
    parser.add_argument("--n", type=int, default=200,
                        help="テストサンプル数 (default: 200)")
    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"ERROR: モデルが見つかりません: {args.model}")
        print(f"  EXP-005の訓練が完了しているか確認してください。")
        print(f"  期待パス: {V5_DEFAULT_MODEL}")
        sys.exit(1)

    # validate_detector_v2 の同じロジックを使用
    from models.validate_detector_v2 import run_validation_v2
    results = run_validation_v2(args.model, n_test=args.n)

    print("\n=== EXP-005 vs EXP-003b 比較目標 ===")
    print(f"EXP-003b基準: Precision=0.965, 腹部Recall=0.724, 全体Recall=0.808")
    print(f"EXP-005目標:  Precision≥0.900 維持, 腹部Recall≥0.800, 全体Recall≥0.800")

    r = results.get("region_recall", {})
    overall_p = results.get("precision", 0)
    overall_r = results.get("recall", 0)

    p_ok = "✅" if overall_p >= 0.900 else "❌"
    r_ok = "✅" if overall_r >= 0.800 else "❌"
    print(f"\n  全体 Precision: {overall_p:.3f} {p_ok} (目標 ≥ 0.900)")
    print(f"  全体 Recall:    {overall_r:.3f} {r_ok} (目標 ≥ 0.800)")

    if "abdomen_pelvis" in r:
        achieved = r["abdomen_pelvis"]
        symbol = "✅" if achieved >= 0.800 else "❌"
        print(f"  腹部 Recall:    {achieved:.3f} {symbol} (目標 ≥ 0.800, EXP-003b=0.724)")
