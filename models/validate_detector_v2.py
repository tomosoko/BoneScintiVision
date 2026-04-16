"""
BoneScintiVision — v2 デュアルビュー検出モデル検証スクリプト

v2モデル（前面+後面512×256）用の検証スクリプト。
validate_detector.py (v1 単面256×256) とは別に管理する。

使い方:
  python3.12 models/validate_detector_v2.py
  python3.12 models/validate_detector_v2.py --model runs/bone_scinti_detector_v2/weights/best.pt
"""

import sys
import argparse
import numpy as np
import cv2
from pathlib import Path
from typing import Dict

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

RUNS_DIR = BASE_DIR / "runs"
ANT_W = POST_W = IMG_H = 256
FULL_W = 512


def compute_iou(box1, box2) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter + 1e-9)


def _resize_and_pad(img, target_w, target_h):
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    nw, nh = int(w * scale), int(h * scale)
    img_r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target_h, target_w), dtype=np.uint8)
    px = (target_w - nw) // 2
    py = (target_h - nh) // 2
    canvas[py:py + nh, px:px + nw] = img_r
    return canvas, scale, px, py


def run_validation_v2(model_path: str, n_test: int = 100) -> Dict:
    from ultralytics import YOLO
    from synth.bone_phantom import BonePhantom
    from synth.scintigraphy_sim import ScintSim

    model = YOLO(model_path)
    rng = np.random.default_rng(999)

    tp_total = fp_total = fn_total = 0
    iou_threshold = 0.3
    region_keys = ["head_neck", "thorax", "abdomen_pelvis", "extremities"]
    region_tp = {k: 0 for k in region_keys}
    region_fn = {k: 0 for k in region_keys}
    count_errors = []

    print(f"  デュアルビュー検証 ({n_test}枚, 512×256)...")
    for i in range(n_test):
        seed = int(rng.integers(1e9))
        phantom = BonePhantom(body_size=rng.uniform(0.85, 1.15), seed=seed)
        sim = ScintSim(counts=int(rng.integers(400_000, 1_200_000)), seed=seed + 1)

        n_les = int(rng.choice([0, 1, 2, 3, 4, 5, 6, 8]))
        lesions = phantom.sample_lesion_sites(n_les)

        # 後面病変: 左右反転
        lesions_post = []
        for les in lesions:
            lp = les.copy()
            lp["x"] = phantom.IMG_W - les["x"]
            lesions_post.append(lp)

        base_ant, _ = phantom.get_anterior_view(add_variation=True)
        base_post, _ = phantom.get_posterior_view(add_variation=True)
        # 訓練データと同じ分布: 50%確率で生理的集積なし
        add_phys = rng.random() > 0.5
        img_ant = sim.acquire(base_ant, lesions, view="anterior", add_physiological=add_phys)
        img_post = sim.acquire(base_post, lesions_post, view="posterior", add_physiological=add_phys)

        img_ant_pad, scale_ant, px_ant, py_ant = _resize_and_pad(img_ant, ANT_W, IMG_H)
        img_post_pad, scale_post, px_post, py_post = _resize_and_pad(img_post, POST_W, IMG_H)
        dual_img = np.hstack([img_ant_pad, img_post_pad])
        img_rgb = cv2.cvtColor(dual_img, cv2.COLOR_GRAY2RGB)

        res = model(img_rgb, verbose=False, conf=0.25)
        boxes = res[0].boxes

        pred_boxes = []
        if boxes is not None and len(boxes) > 0:
            for b in boxes.xyxy.cpu().numpy():
                pred_boxes.append(b[:4].tolist())

        # GT: 前面のみ評価（前半 x∈[0,256]）
        gt_boxes = []
        gt_regions = []
        for les in lesions:
            x = les["x"] * scale_ant + px_ant  # 前面座標
            y = les["y"] * scale_ant + py_ant
            s = les["size"] * scale_ant * 1.1
            gt_boxes.append([x - s, y - s, x + s, y + s])
            yc = y / IMG_H
            if yc < 0.25:
                gt_regions.append("head_neck")
            elif yc < 0.55:
                gt_regions.append("thorax")
            elif yc < 0.75:
                gt_regions.append("abdomen_pelvis")
            else:
                gt_regions.append("extremities")

        # 前面側の予測のみ: ボックス中心が x < 256 のもの
        pred_ant = [pb for pb in pred_boxes if (pb[0] + pb[2]) / 2 < ANT_W]
        count_errors.append(len(pred_ant) - len(gt_boxes))

        matched_gt = set()
        for pb in pred_ant:
            best_iou, best_gi = 0, -1
            for gi, gb in enumerate(gt_boxes):
                if gi in matched_gt:
                    continue
                iou = compute_iou(pb, gb)
                if iou > best_iou:
                    best_iou, best_gi = iou, gi
            if best_iou >= iou_threshold and best_gi >= 0:
                tp_total += 1
                matched_gt.add(best_gi)
                region_tp[gt_regions[best_gi]] += 1
            else:
                fp_total += 1

        for gi in range(len(gt_boxes)):
            if gi not in matched_gt:
                fn_total += 1
                region_fn[gt_regions[gi]] += 1

    precision = tp_total / (tp_total + fp_total + 1e-9)
    recall = tp_total / (tp_total + fn_total + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    mae_count = float(np.mean(np.abs(count_errors)))
    me_count = float(np.mean(count_errors))

    print(f"\n{'='*60}")
    print(f"  v2 デュアルビュー検出精度評価 (n={n_test}, IoU>{iou_threshold})")
    print(f"  TP={tp_total}  FP={fp_total}  FN={fn_total}")
    print(f"  Precision : {precision:.3f}")
    print(f"  Recall    : {recall:.3f}")
    print(f"  F1        : {f1:.3f}")
    print(f"  病変数誤差: MAE={mae_count:.2f}  ME={me_count:+.2f}")
    print(f"\n  部位別 Recall (前面):")
    for rk in region_keys:
        r_tp = region_tp[rk]
        r_fn = region_fn[rk]
        r_rec = r_tp / (r_tp + r_fn + 1e-9)
        print(f"    {rk:20s}: {r_rec:.3f}  (TP={r_tp}, FN={r_fn})")
    print(f"{'='*60}")

    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp_total, "fp": fp_total, "fn": fn_total,
            "mae_count": mae_count, "me_count": me_count,
            "region_recall": {k: region_tp[k] / (region_tp[k] + region_fn[k] + 1e-9)
                              for k in region_keys}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(RUNS_DIR / "bone_scinti_detector_v2" / "weights" / "best.pt"))
    parser.add_argument("--n", type=int, default=100)
    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"ERROR: モデルが見つかりません: {args.model}")
        sys.exit(1)

    run_validation_v2(args.model, n_test=args.n)
