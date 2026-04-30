"""
BoneScintiVision — 骨転移負荷スコアリング (Phase 2)

訓練済みYOLO11sで検出したhot spotから骨転移負荷スコアを算出する。

スコア定義:
  - Total Bone Burden (TBB): 全検出hot spotの正規化面積の合計
  - Bone Scan Index (BSI) 相当: TBB × 重み係数
  - 部位別スコア: 7大臨床領域への分類
  - リスク分類: Stage 0-4 (病変数ベース)

使い方:
  python3.12 models/score_burden.py --image /path/to/scan.png
  python3.12 models/score_burden.py --image /path/to/scan.png --model runs/bone_scinti_detector_v1/weights/best.pt
"""

import sys
import argparse
import json
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Optional

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

RUNS_DIR = BASE_DIR / "runs"

# ─── 臨床領域マッピング ───────────────────────────────────────────────────────
# 画像 Y座標（0〜1）による大まかな部位分類
# 256×256にパディング済みの座標前提（元は256×512の上半身〜下半身）

CLINICAL_REGIONS = {
    "head_neck":       (0.00, 0.18),   # 頭蓋〜頸椎
    "thorax_upper":    (0.18, 0.38),   # 上胸部（胸椎上部・肋骨上・鎖骨・胸骨）
    "thorax_lower":    (0.38, 0.55),   # 下胸部（胸椎下部・肋骨下・腰椎上部）
    "lumbar_pelvis":   (0.55, 0.72),   # 腰椎〜骨盤・仙骨
    "proximal_femur":  (0.72, 0.88),   # 大腿骨近位部
    "distal_extremity":(0.88, 1.00),   # 遠位肢
}

# 各領域の臨床的重み（転移時の予後影響度・痛み頻度に基づく）
REGION_WEIGHTS = {
    "head_neck":        0.8,
    "thorax_upper":     1.5,   # 脊髄圧迫リスク高
    "thorax_lower":     1.5,
    "lumbar_pelvis":    1.8,   # 最多転移部位
    "proximal_femur":   1.2,   # 病的骨折リスク
    "distal_extremity": 0.5,
}

# リスク分類（病変数ベース）
RISK_STAGES = [
    (0,  0,  "Stage 0", "正常範囲"),
    (1,  2,  "Stage 1", "軽度"),
    (3,  5,  "Stage 2", "中等度"),
    (6, 10,  "Stage 3", "高度"),
    (11, 999, "Stage 4", "広汎性"),
]


def classify_clinical_region(y_center_norm: float) -> str:
    """Y座標（0〜1正規化）から臨床領域を返す"""
    for region, (y0, y1) in CLINICAL_REGIONS.items():
        if y0 <= y_center_norm < y1:
            return region
    return "distal_extremity"


def classify_risk_stage(n_lesions: int) -> Dict:
    for lo, hi, stage, label in RISK_STAGES:
        if lo <= n_lesions <= hi:
            return {"stage": stage, "label": label, "n_lesions": n_lesions}
    return {"stage": "Stage 4", "label": "広汎性", "n_lesions": n_lesions}


def compute_bone_burden_score(
    detections: List[Dict],
    image_size: int = 256,
    image_w: Optional[int] = None,
    image_h: Optional[int] = None,
) -> Dict:
    """
    骨転移負荷スコアを計算。

    Args:
        detections: [{x, y, w, h, conf, class}]  x,y は中心座標（ピクセル）
        image_size: 画像サイズ（正方形の場合のみ有効、後方互換用）
        image_w:    画像幅（非正方形画像では必須）
        image_h:    画像高さ（非正方形画像では必須）

    Returns:
        Dict with:
          total_bone_burden: 正規化面積合計 (0.0〜100.0 %)
          bsi_equivalent:    BSI相当スコア（加重補正）
          n_lesions:         検出病変数
          mean_conf:         平均信頼度
          region_scores:     部位別スコア
          risk_stage:        リスク分類
    """
    img_w = image_w if image_w is not None else image_size
    img_h = image_h if image_h is not None else image_size

    if not detections:
        return {
            "n_lesions": 0,
            "total_bone_burden": 0.0,
            "bsi_equivalent": 0.0,
            "mean_conf": 0.0,
            "region_scores": {r: 0.0 for r in CLINICAL_REGIONS},
            "risk_stage": classify_risk_stage(0),
        }

    n = len(detections)
    region_scores = {r: 0.0 for r in CLINICAL_REGIONS}
    total_area = 0.0
    weighted_area = 0.0

    for d in detections:
        yc_norm = d["y"] / img_h
        area_norm = (d["w"] / img_w) * (d["h"] / img_h) * 100.0  # %
        region = classify_clinical_region(yc_norm)
        weight = REGION_WEIGHTS[region]

        region_scores[region] += area_norm
        total_area += area_norm
        weighted_area += area_norm * weight

    mean_conf = float(np.mean([d["conf"] for d in detections]))

    # BSI相当: 加重面積をスケーリング（実測BSIは0〜100%換算）
    bsi_equivalent = min(100.0, weighted_area * 0.5)

    return {
        "n_lesions": n,
        "total_bone_burden": round(total_area, 3),
        "bsi_equivalent": round(bsi_equivalent, 3),
        "mean_conf": round(mean_conf, 3),
        "region_scores": {k: round(v, 4) for k, v in region_scores.items()},
        "risk_stage": classify_risk_stage(n),
    }


def run_inference_and_score(
    image_path: str,
    model_path: str,
    conf: float = 0.25,
    save_vis: bool = True,
) -> Dict:
    """
    1枚の画像に対してYOLO推論 + スコア算出。

    Args:
        image_path: 入力画像パス（256×256 PNG）
        model_path: YOLOモデルパス
        conf:       信頼度しきい値
        save_vis:   アノテーション画像を保存するか

    Returns:
        score dict
    """
    from ultralytics import YOLO

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"画像が見つかりません: {image_path}")

    # グレースケール→RGB変換（必要なら）
    if len(img.shape) == 2 or img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    model = YOLO(model_path)
    results = model(img, verbose=False, conf=conf)
    boxes = results[0].boxes

    detections = []
    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        for b, c in zip(xyxy, confs):
            x1, y1, x2, y2 = b[:4]
            detections.append({
                "x": float((x1 + x2) / 2),
                "y": float((y1 + y2) / 2),
                "w": float(x2 - x1),
                "h": float(y2 - y1),
                "conf": float(c),
            })

    score = compute_bone_burden_score(detections, image_w=img.shape[1], image_h=img.shape[0])

    # 可視化
    if save_vis:
        vis = img.copy()
        for d in detections:
            x1 = int(d["x"] - d["w"] / 2)
            y1 = int(d["y"] - d["h"] / 2)
            x2 = int(d["x"] + d["w"] / 2)
            y2 = int(d["y"] + d["h"] / 2)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 80, 255), 2)
            cv2.putText(vis, f"{d['conf']:.2f}", (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 80, 255), 1)

        # スコア表示
        stage = score["risk_stage"]["stage"]
        label = score["risk_stage"]["label"]
        tbb = score["total_bone_burden"]
        cv2.putText(vis, f"{stage} ({label})", (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 255, 80), 1)
        cv2.putText(vis, f"TBB={tbb:.2f}%  N={score['n_lesions']}", (4, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 255, 80), 1)

        out_path = str(Path(image_path).parent / (Path(image_path).stem + "_scored.png"))
        cv2.imwrite(out_path, vis)
        score["vis_path"] = out_path

    return score


def batch_score(
    image_dir: str,
    model_path: str,
    out_json: Optional[str] = None,
    conf: float = 0.25,
) -> List[Dict]:
    """
    ディレクトリ内全画像をバッチ処理してスコア一覧を返す。
    """
    from ultralytics import YOLO

    model = YOLO(model_path)
    image_paths = list(Path(image_dir).glob("*.png")) + list(Path(image_dir).glob("*.jpg"))
    results_all = []

    print(f"  バッチスコアリング: {len(image_paths)}枚")
    for img_path in sorted(image_paths):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        res = model(img, verbose=False, conf=conf)
        boxes = res[0].boxes
        detections = []
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            for b, c in zip(xyxy, confs):
                x1, y1, x2, y2 = b[:4]
                detections.append({
                    "x": float((x1 + x2) / 2),
                    "y": float((y1 + y2) / 2),
                    "w": float(x2 - x1),
                    "h": float(y2 - y1),
                    "conf": float(c),
                })

        score = compute_bone_burden_score(
            detections, image_w=img.shape[1], image_h=img.shape[0]
        )
        score["image"] = str(img_path)
        results_all.append(score)

    if out_json:
        with open(out_json, "w") as f:
            json.dump(results_all, f, indent=2, ensure_ascii=False)
        print(f"  結果保存: {out_json}")

    return results_all


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BoneScintiVision スコアリング")
    parser.add_argument("--image", type=str, help="単一画像パス")
    parser.add_argument("--dir",   type=str, help="ディレクトリ（バッチ）")
    parser.add_argument("--model", default=str(RUNS_DIR / "detect" / "bone_scinti_detector_v8" / "weights" / "best.pt"))
    parser.add_argument("--conf",  type=float, default=0.25)
    parser.add_argument("--out",   type=str, default=None, help="JSON出力パス（バッチ時）")
    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"ERROR: モデルが見つかりません: {args.model}")
        print("  先に python3.12 models/train_detector.py を実行してください")
        sys.exit(1)

    if args.image:
        score = run_inference_and_score(args.image, args.model, conf=args.conf)
        print(json.dumps(score, indent=2, ensure_ascii=False))
    elif args.dir:
        results = batch_score(args.dir, args.model, out_json=args.out, conf=args.conf)
        # サマリー表示
        n_cases = len(results)
        avg_tbb = np.mean([r["total_bone_burden"] for r in results]) if results else 0
        avg_les = np.mean([r["n_lesions"] for r in results]) if results else 0
        print(f"\n{'='*50}")
        print(f"  バッチスコアリング完了: {n_cases}枚")
        print(f"  平均 TBB: {avg_tbb:.3f}%")
        print(f"  平均病変数: {avg_les:.1f}")
        print(f"{'='*50}")
    else:
        parser.print_help()
