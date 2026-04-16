"""
BoneScintiVision — 訓練後最終評価スクリプト

訓練完了後に以下を一括実行する:
  1. validate_detector.py → Precision/Recall/F1 + 部位別Recall
  2. infer_demo.py → 推論デモグリッド画像
  3. EXPERIMENTS.md を自動更新

使い方:
  python3.12 models/eval_final.py
  python3.12 models/eval_final.py --model runs/bone_scinti_detector_v1/weights/best.pt
"""

import sys
import json
import time
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

RUNS_DIR = BASE_DIR / "runs"
DEFAULT_MODEL = RUNS_DIR / "detect" / "bone_scinti_detector_v62" / "weights" / "best.pt"


def update_experiments_md(results: dict, model_path: str, elapsed: float):
    """EXPERIMENTS.mdの最終結果セクションを更新する"""
    exp_path = BASE_DIR / "EXPERIMENTS.md"
    content = exp_path.read_text(encoding="utf-8")

    # 最終結果テーブルを更新
    from datetime import date
    today = date.today().isoformat()
    final_results = f"""### 最終結果（{today}, 評価完了）

| 指標 | 値 |
|---|---|
| mAP50 (YOLO val) | — |
| Precision | {results['precision']:.3f} |
| Recall    | {results['recall']:.3f} |
| F1        | {results['f1']:.3f} |
| 病変数MAE | {results['mae_count']:.2f} |
| 評価画像数 | 100枚 |
| 評価時間   | {elapsed:.0f}秒 |

#### 部位別 Recall
| 部位 | Recall |
|---|---|"""

    for region, recall in results['region_recall'].items():
        final_results += f"\n| {region} | {recall:.3f} |"

    # 既存の最終結果セクションを置換
    old_section = "### 最終結果\n*(訓練完了後に記録)*\n\n| 指標 | 値 |\n|---|---|\n| mAP50 | — |\n| Precision | — |\n| Recall | — |\n| 訓練時間 | — |"
    if old_section in content:
        content = content.replace(old_section, final_results)
        exp_path.write_text(content, encoding="utf-8")
        print(f"  EXPERIMENTS.md 更新完了")
    else:
        print(f"  EXPERIMENTS.md の自動更新スキップ（既に更新済みか形式が異なります）")


def run_eval(model_path: str, n_test: int = 100, save_demo: bool = True):
    if not Path(model_path).exists():
        print(f"ERROR: モデルが見つかりません: {model_path}")
        print("  先に python3.12 models/train_detector.py を実行してください")
        sys.exit(1)

    print("=" * 60)
    print("  BoneScintiVision — 最終評価")
    print(f"  モデル: {model_path}")
    print("=" * 60)

    # 1. Precision/Recall/F1 評価 (dual-view v2 ロジック使用)
    print("\n[1/3] 検出精度評価...")
    from models.validate_detector_v2 import run_validation_v2
    t0 = time.time()
    results = run_validation_v2(model_path, n_test=n_test)
    elapsed = time.time() - t0

    # 2. 推論デモグリッド
    demo_path = "/tmp/bonescinti_final_demo.png"
    if save_demo:
        print(f"\n[2/3] 推論デモ生成...")
        from models.infer_demo import make_infer_grid
        make_infer_grid(model_path, n=12, out_path=demo_path)

    # 3. EXPERIMENTS.md 更新
    print(f"\n[3/3] EXPERIMENTS.md 更新...")
    update_experiments_md(results, model_path, elapsed)

    # サマリー
    print(f"\n{'='*60}")
    print(f"  最終評価完了")
    print(f"  F1={results['f1']:.3f}  P={results['precision']:.3f}  R={results['recall']:.3f}")
    if save_demo:
        print(f"  デモ画像: {demo_path}")
    print(f"{'='*60}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--no-demo", action="store_true")
    args = parser.parse_args()

    run_eval(args.model, n_test=args.n, save_demo=not args.no_demo)
