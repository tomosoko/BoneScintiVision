"""
BoneScintiVision — EXP-003 検証スクリプト

v3データセット (yolo11m, 640px, デュアルビュー) 用。
validate_detector_v2.py と同じロジックを使用し、モデルパスだけ変更。

使い方:
  cd /Users/kohei/develop/research/BoneScintiVision
  python3.12 models/validate_detector_v3.py
  python3.12 models/validate_detector_v3.py --n 200
"""

import sys
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

# EXP-003 (imgsz=640) は MPS クラッシュで失敗 → EXP-003b を使うこと
# 将来 imgsz=640 再試行時のために正しいパスを設定しておく
V3_DEFAULT_MODEL = BASE_DIR / "runs" / "detect" / "bone_scinti_detector_v3" / "weights" / "best.pt"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BoneScintiVision EXP-003 検証")
    parser.add_argument("--model", default=str(V3_DEFAULT_MODEL),
                        help="検証するモデルのパス (default: EXP-003 best.pt)")
    parser.add_argument("--n", type=int, default=200,
                        help="テストサンプル数 (default: 200)")
    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"ERROR: モデルが見つかりません: {args.model}")
        print(f"  EXP-003の訓練が完了しているか確認してください。")
        sys.exit(1)

    # validate_detector_v2 の同じロジックを使用
    from models.validate_detector_v2 import run_validation_v2
    results = run_validation_v2(args.model, n_test=args.n)

    print("\n=== EXP-003 vs EXP-002 比較目標 ===")
    print(f"abdomen_pelvis Recall: EXP-002=0.711 → 目標 0.800+")
    r = results.get("region_recall", {})
    if "abdomen_pelvis" in r:
        achieved = r["abdomen_pelvis"]
        symbol = "✅" if achieved >= 0.800 else "❌"
        print(f"  EXP-003実績: {achieved:.3f} {symbol}")
