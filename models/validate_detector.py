"""
BoneScintiVision — 検出モデル検証スクリプト

訓練済みYOLO11sモデルで合成骨シンチ画像を検証し、
hot spot 検出精度と定量スコアリングを評価する。

使い方:
  python3.12 models/validate_detector.py
  python3.12 models/validate_detector.py --model runs/bone_scinti_detector_v1/weights/best.pt
"""

import os
import sys
import argparse
import json
import time
import numpy as np
import cv2
from pathlib import Path
from typing import Dict

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))
RUNS_DIR = BASE_DIR / "runs"
OUT_DIR  = BASE_DIR / "data" / "validation_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

from synth.bone_phantom import BonePhantom, REGION_LABELS_JP, BONE_REGIONS
from synth.scintigraphy_sim import ScintSim


def calc_bone_burden_score(detections, image_size: int = 256) -> Dict:
    """
    骨転移負荷スコアを計算（Total Bone Burden 相当）。

    Args:
        detections: YOLO結果 [{x, y, w, h, conf, class}]
        image_size: 画像サイズ

    Returns:
        {
            "n_lesions": 検出数,
            "total_burden": 総集積面積(正規化),
            "mean_conf": 平均信頼度,
            "burden_by_region": 部位別スコア（推定）,
        }
    """
    if not detections:
        return {"n_lesions": 0, "total_burden": 0.0, "mean_conf": 0.0}

    n = len(detections)
    total_area = sum(d["w"] * d["h"] for d in detections)
    mean_conf = np.mean([d["conf"] for d in detections])

    # 部位別分類（Y座標から大まかに推定）
    region_counts = {"head_neck": 0, "thorax": 0, "abdomen_pelvis": 0, "extremities": 0}
    for d in detections:
        yc = d["y"] / image_size
        if yc < 0.25:
            region_counts["head_neck"] += 1
        elif yc < 0.55:
            region_counts["thorax"] += 1
        elif yc < 0.75:
            region_counts["abdomen_pelvis"] += 1
        else:
            region_counts["extremities"] += 1

    return {
        "n_lesions": n,
        "total_burden": round(float(total_area * 100), 2),  # % of image
        "mean_conf": round(float(mean_conf), 3),
        "burden_by_region": region_counts,
    }


def run_validation(model_path: str, n_test: int = 100) -> Dict:
    from ultralytics import YOLO
    from synth.bone_phantom import BonePhantom
    from synth.scintigraphy_sim import ScintSim

    model = YOLO(model_path)
    rng = np.random.default_rng(99)
    results_all = []

    tp_total = fp_total = fn_total = 0
    iou_threshold = 0.3

    print(f"  検証中 ({n_test}枚)...")
    for i in range(n_test):
        seed = int(rng.integers(1e9))
        phantom = BonePhantom(body_size=rng.uniform(0.85, 1.15), seed=seed)
        sim = ScintSim(counts=int(rng.integers(400_000, 1_200_000)), seed=seed + 1)

        n_les = int(rng.choice([0, 1, 2, 3, 4, 5, 6, 8]))
        lesions = phantom.sample_lesion_sites(n_les)
        base_img, _ = phantom.get_anterior_view()
        img = sim.acquire(base_img, lesions, view="anterior")

        # リサイズ: 256×512 → pad to 256×256
        h, w = img.shape[:2]
        scale = min(256 / w, 256 / h)
        nw, nh = int(w * scale), int(h * scale)
        img_r = cv2.resize(img, (nw, nh))
        img_padded = np.zeros((256, 256), dtype=np.uint8)
        px, py = (256 - nw) // 2, (256 - nh) // 2
        img_padded[py:py+nh, px:px+nw] = img_r
        img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_GRAY2RGB)

        res = model(img_rgb, verbose=False, conf=0.25)
        boxes = res[0].boxes

        # GT ボックス
        gt_boxes = []
        for les in lesions:
            x = les["x"] * scale + px
            y = les["y"] * scale + py
            s = les["size"] * scale * 1.1
            gt_boxes.append([x - s, y - s, x + s, y + s])

        # 検出ボックス
        pred_boxes = []
        if boxes is not None and len(boxes) > 0:
            for b in boxes.xyxy.cpu().numpy():
                pred_boxes.append(b[:4].tolist())

        # IoU マッチング（簡易）
        matched_gt = set()
        for pb in pred_boxes:
            best_iou = 0
            best_gi = -1
            for gi, gb in enumerate(gt_boxes):
                if gi in matched_gt:
                    continue
                iou = compute_iou(pb, gb)
                if iou > best_iou:
                    best_iou = iou
                    best_gi = gi
            if best_iou >= iou_threshold and best_gi >= 0:
                tp_total += 1
                matched_gt.add(best_gi)
            else:
                fp_total += 1
        fn_total += len(gt_boxes) - len(matched_gt)

    # メトリクス
    precision = tp_total / (tp_total + fp_total + 1e-9)
    recall    = tp_total / (tp_total + fn_total + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    print(f"\n{'='*60}")
    print(f"  検出精度評価 (n={n_test}, IoU>{iou_threshold})")
    print(f"  TP={tp_total}  FP={fp_total}  FN={fn_total}")
    print(f"  Precision : {precision:.3f}")
    print(f"  Recall    : {recall:.3f}")
    print(f"  F1        : {f1:.3f}")
    print(f"{'='*60}")

    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp_total, "fp": fp_total, "fn": fn_total}


def compute_iou(box1, box2) -> float:
    """XYXY形式のIoU計算"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter + 1e-9
    return inter / union


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(RUNS_DIR / "bone_scinti_detector_v1" / "weights" / "best.pt"))
    parser.add_argument("--n", type=int, default=100)
    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"ERROR: モデルが見つかりません: {args.model}")
        print("  先に python3.12 models/train_detector.py を実行してください")
        sys.exit(1)

    run_validation(args.model, n_test=args.n)
