"""
BoneScintiVision — EXP-009 検証スクリプト

EXP-009 (yolo11m, 512px, yolo_dataset_v8, 生理的集積なし55%←70%) 用。
validate_detector_v2.py と同じロジックを使用し、モデルパスを変更。

EXP-009の狙い:
  - EXP-007 P=0.707 (FP過多) の原因: no_physio=70%で生理的集積を学習不足
  - no_physio 70%→55% で生理的集積パターンを学習させFPを削減
  - 腹部Recall≥0.800 を維持しつつ Precision改善を目指す

使い方:
  cd ~/develop/research/BoneScintiVision
  python3.12 models/validate_detector_v9.py
  python3.12 models/validate_detector_v9.py --n 200
"""

import sys
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

V8_DEFAULT_MODEL = BASE_DIR / "runs" / "detect" / "bone_scinti_detector_v8" / "weights" / "best.pt"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BoneScintiVision EXP-009 検証")
    parser.add_argument("--model", default=str(V8_DEFAULT_MODEL),
                        help="検証するモデルのパス (default: EXP-009 best.pt)")
    parser.add_argument("--n", type=int, default=200,
                        help="テストサンプル数 (default: 200)")
    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"ERROR: モデルが見つかりません: {args.model}")
        print(f"  先に以下を実行してください:")
        print(f"    python3.12 synth/generate_dataset_v8.py")
        print(f"    python3.12 models/train_detector_v8.py")
        sys.exit(1)

    from models.validate_detector_v2 import run_validation_v2
    results = run_validation_v2(args.model, n_test=args.n)

    print("\n=== EXP-009 vs EXP-007 比較 ===")
    print(f"目標: 腹部Recall ≥ 0.800, Precision ≥ 0.900")
    print(f"EXP-007実績: P=0.707, R=0.899, 腹部=0.828")
    print(f"EXP-008実績: P=0.718(+0.011), 腹部=0.810 (physio mask後処理)")
    r = results.get("region_recall", {})
    overall_p = results.get("precision", 0)
    overall_r = results.get("recall", 0)

    if "abdomen_pelvis" in r:
        abd = r["abdomen_pelvis"]
        abd_sym = "✅" if abd >= 0.800 else "❌"
        print(f"  腹部Recall: {abd:.3f} {abd_sym} (目標0.800, EXP-007=0.828)")
    p_sym = "✅" if overall_p >= 0.900 else "❌"
    r_sym = "✅" if overall_r >= 0.800 else "❌"
    print(f"  Precision: {overall_p:.3f} {p_sym} (目標≥0.900, EXP-007=0.707)")
    print(f"  全体Recall: {overall_r:.3f} {r_sym} (目標≥0.800, EXP-007=0.899)")
