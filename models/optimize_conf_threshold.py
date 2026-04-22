"""
BoneScintiVision — EXP-007 conf閾値最適化スクリプト

EXP-007モデル (P=0.707, R=0.899) の conf閾値を最適化して
Precision ≥ 0.900 かつ 腹部Recall ≥ 0.800 の同時達成を探索する。

conf=0.25 (デフォルト) → 0.70 をスイープし、最適バランスを特定。
推論は conf=0.05 の1回のみ実行し、閾値フィルタで高速評価。

使い方:
  cd ~/develop/research/BoneScintiVision
  python3.12 models/optimize_conf_threshold.py
  python3.12 models/optimize_conf_threshold.py --n 300
  python3.12 models/optimize_conf_threshold.py --model path/to/best.pt
"""

import sys
import argparse
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

V7_DEFAULT_MODEL = (
    BASE_DIR / "runs" / "detect" / "bone_scinti_detector_v7-2" / "weights" / "best.pt"
)

ANT_W = POST_W = IMG_H = 256
FULL_W = 512
IOU_THRESHOLD = 0.3

CONF_THRESHOLDS = [
    0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35,
    0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70,
]

REGION_KEYS = ["head_neck", "thorax", "abdomen_pelvis", "extremities"]


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


def classify_region(y_norm: float) -> str:
    if y_norm < 0.25:
        return "head_neck"
    elif y_norm < 0.55:
        return "thorax"
    elif y_norm < 0.75:
        return "abdomen_pelvis"
    else:
        return "extremities"


def generate_test_data(n_test: int, rng: np.random.Generator):
    """Generate synthetic test images and return GT + raw predictions."""
    from synth.bone_phantom import BonePhantom
    from synth.scintigraphy_sim import ScintSim

    samples = []

    print(f"  テストデータ生成中 ({n_test}枚)...")
    for i in range(n_test):
        seed = int(rng.integers(1e9))
        phantom = BonePhantom(body_size=rng.uniform(0.85, 1.15), seed=seed)
        sim = ScintSim(counts=int(rng.integers(400_000, 1_200_000)), seed=seed + 1)

        n_les = int(rng.choice([0, 1, 2, 3, 4, 5, 6, 8]))
        lesions = phantom.sample_lesion_sites(n_les)

        lesions_post = []
        for les in lesions:
            lp = les.copy()
            lp["x"] = phantom.IMG_W - les["x"]
            lesions_post.append(lp)

        base_ant, _ = phantom.get_anterior_view(add_variation=True)
        base_post, _ = phantom.get_posterior_view(add_variation=True)
        add_phys = rng.random() > 0.5
        img_ant = sim.acquire(base_ant, lesions, view="anterior", add_physiological=add_phys)
        img_post = sim.acquire(base_post, lesions_post, view="posterior", add_physiological=add_phys)

        img_ant_pad, scale_ant, px_ant, py_ant = _resize_and_pad(img_ant, ANT_W, IMG_H)
        img_post_pad, scale_post, px_post, py_post = _resize_and_pad(img_post, POST_W, IMG_H)
        dual_img = np.hstack([img_ant_pad, img_post_pad])
        img_rgb = cv2.cvtColor(dual_img, cv2.COLOR_GRAY2RGB)

        gt_boxes = []
        gt_regions = []
        for les in lesions:
            x = les["x"] * scale_ant + px_ant
            y = les["y"] * scale_ant + py_ant
            s = les["size"] * scale_ant * 1.1
            gt_boxes.append([x - s, y - s, x + s, y + s])
            gt_regions.append(classify_region(y / IMG_H))

        samples.append({
            "img_rgb": img_rgb,
            "gt_boxes": gt_boxes,
            "gt_regions": gt_regions,
        })

        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{n_test} 枚生成完了")

    return samples


def run_inference(model, samples: List[Dict]) -> List[List[Tuple[List[float], float]]]:
    """Run inference once at low conf and store boxes with confidence scores."""
    print(f"  推論実行中 (conf=0.05, {len(samples)}枚)...")
    all_preds = []
    for i, s in enumerate(samples):
        res = model(s["img_rgb"], verbose=False, conf=0.05)
        boxes = res[0].boxes
        preds = []
        if boxes is not None and len(boxes) > 0:
            for b, c in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy()):
                cx = (b[0] + b[2]) / 2
                if cx < ANT_W:  # anterior only
                    preds.append((b[:4].tolist(), float(c)))
        all_preds.append(preds)
        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{len(samples)} 枚推論完了")
    return all_preds


def evaluate_at_threshold(
    samples: List[Dict],
    all_preds: List[List[Tuple[List[float], float]]],
    conf_thresh: float,
) -> Dict:
    """Evaluate metrics at a specific confidence threshold."""
    tp_total = fp_total = fn_total = 0
    region_tp = {k: 0 for k in REGION_KEYS}
    region_fn = {k: 0 for k in REGION_KEYS}

    for s, preds in zip(samples, all_preds):
        gt_boxes = s["gt_boxes"]
        gt_regions = s["gt_regions"]

        filtered = [p for p in preds if p[1] >= conf_thresh]
        pred_boxes = [p[0] for p in filtered]

        matched_gt = set()
        for pb in pred_boxes:
            best_iou, best_gi = 0, -1
            for gi, gb in enumerate(gt_boxes):
                if gi in matched_gt:
                    continue
                iou = compute_iou(pb, gb)
                if iou > best_iou:
                    best_iou, best_gi = iou, gi
            if best_iou >= IOU_THRESHOLD and best_gi >= 0:
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

    region_recall = {}
    for k in REGION_KEYS:
        region_recall[k] = region_tp[k] / (region_tp[k] + region_fn[k] + 1e-9)

    return {
        "conf": conf_thresh,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp_total,
        "fp": fp_total,
        "fn": fn_total,
        "region_recall": region_recall,
    }


def main():
    parser = argparse.ArgumentParser(
        description="BoneScintiVision EXP-007 conf閾値最適化"
    )
    parser.add_argument(
        "--model", default=str(V7_DEFAULT_MODEL),
        help="モデルパス (default: EXP-007 v7-2 best.pt)",
    )
    parser.add_argument("--n", type=int, default=200, help="テストサンプル数")
    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"ERROR: モデルが見つかりません: {args.model}")
        sys.exit(1)

    from ultralytics import YOLO

    print(f"=== conf閾値最適化 (n={args.n}) ===")
    print(f"  モデル: {args.model}")

    model = YOLO(args.model)
    rng = np.random.default_rng(999)

    samples = generate_test_data(args.n, rng)
    all_preds = run_inference(model, samples)

    # Sweep thresholds
    results = []
    for ct in CONF_THRESHOLDS:
        r = evaluate_at_threshold(samples, all_preds, ct)
        results.append(r)

    # Print results table
    print(f"\n{'='*90}")
    print(f"  conf閾値スイープ結果 (n={args.n}, IoU>{IOU_THRESHOLD})")
    print(f"{'='*90}")
    print(f"  {'conf':>5s}  {'Prec':>6s}  {'Recall':>6s}  {'F1':>6s}  "
          f"{'head':>6s}  {'thorax':>6s}  {'abdomen':>7s}  {'extrem':>6s}  "
          f"{'TP':>4s}  {'FP':>4s}  {'FN':>4s}  判定")
    print(f"  {'-'*5}  {'-'*6}  {'-'*6}  {'-'*6}  "
          f"{'-'*6}  {'-'*6}  {'-'*7}  {'-'*6}  "
          f"{'-'*4}  {'-'*4}  {'-'*4}  ----")

    best_f1_meeting_targets = None
    best_f1_overall = None

    for r in results:
        rr = r["region_recall"]
        p_ok = r["precision"] >= 0.900
        abd_ok = rr["abdomen_pelvis"] >= 0.800
        both_ok = p_ok and abd_ok
        mark = " ★" if both_ok else ""

        print(f"  {r['conf']:>5.2f}  {r['precision']:>6.3f}  {r['recall']:>6.3f}  {r['f1']:>6.3f}  "
              f"{rr['head_neck']:>6.3f}  {rr['thorax']:>6.3f}  {rr['abdomen_pelvis']:>7.3f}  "
              f"{rr['extremities']:>6.3f}  "
              f"{r['tp']:>4d}  {r['fp']:>4d}  {r['fn']:>4d}  "
              f"{'P≥.9' if p_ok else '    '} {'A≥.8' if abd_ok else '    '}{mark}")

        if both_ok:
            if best_f1_meeting_targets is None or r["f1"] > best_f1_meeting_targets["f1"]:
                best_f1_meeting_targets = r
        if best_f1_overall is None or r["f1"] > best_f1_overall["f1"]:
            best_f1_overall = r

    # Summary
    print(f"\n{'='*90}")
    print("  最適閾値分析:")

    if best_f1_meeting_targets:
        r = best_f1_meeting_targets
        rr = r["region_recall"]
        print(f"  ★ 両目標達成 最高F1: conf={r['conf']:.2f}")
        print(f"    P={r['precision']:.3f}  R={r['recall']:.3f}  F1={r['f1']:.3f}")
        print(f"    腹部Recall={rr['abdomen_pelvis']:.3f}")
    else:
        print(f"  ❌ Precision≥0.900 かつ 腹部Recall≥0.800 を同時達成する閾値なし")
        # Find best Precision >= 0.900
        p_ok_results = [r for r in results if r["precision"] >= 0.900]
        if p_ok_results:
            best_p = max(p_ok_results, key=lambda x: x["region_recall"]["abdomen_pelvis"])
            rr = best_p["region_recall"]
            print(f"  → P≥0.900 内で腹部最高: conf={best_p['conf']:.2f}")
            print(f"    P={best_p['precision']:.3f}  R={best_p['recall']:.3f}  "
                  f"腹部={rr['abdomen_pelvis']:.3f}")
        # Find best abdomen >= 0.800
        abd_ok_results = [r for r in results if r["region_recall"]["abdomen_pelvis"] >= 0.800]
        if abd_ok_results:
            best_a = max(abd_ok_results, key=lambda x: x["precision"])
            rr = best_a["region_recall"]
            print(f"  → 腹部≥0.800 内でP最高: conf={best_a['conf']:.2f}")
            print(f"    P={best_a['precision']:.3f}  R={best_a['recall']:.3f}  "
                  f"腹部={rr['abdomen_pelvis']:.3f}")

    r = best_f1_overall
    rr = r["region_recall"]
    print(f"\n  全体最高F1: conf={r['conf']:.2f}")
    print(f"    P={r['precision']:.3f}  R={r['recall']:.3f}  F1={r['f1']:.3f}")
    print(f"    腹部Recall={rr['abdomen_pelvis']:.3f}")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
