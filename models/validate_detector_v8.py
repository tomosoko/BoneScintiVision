"""
BoneScintiVision — EXP-008 検証スクリプト（生理的集積マスク後処理）

EXP-007 モデル (P=0.707, R=0.899, 腹部R=0.828) に
生理的集積マスク後処理 (physio_mask) を組み合わせて
Precision 改善 + 腹部Recall 維持を評価する。

手法:
  1. conf=0.05 で1回だけ推論し、全検出を保存
  2. suppress_conf をスイープして生理的集積ゾーン内 FP を除外
  3. 各 suppress_conf での Precision/Recall/腹部Recall を評価

使い方:
  cd ~/develop/research/BoneScintiVision
  python3.12 models/validate_detector_v8.py
  python3.12 models/validate_detector_v8.py --n 200
  python3.12 models/validate_detector_v8.py --model path/to/best.pt --n 300
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
IOU_THRESHOLD = 0.3
REGION_KEYS = ["head_neck", "thorax", "abdomen_pelvis", "extremities"]

# EXP-007 baseline (conf=0.25, no mask)
EXP007_BASELINE = {"precision": 0.707, "recall": 0.899, "abdomen_recall": 0.828, "f1": 0.792}

SUPPRESS_CONF_THRESHOLDS = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
INFERENCE_CONF = 0.05  # 低信頼度で1回推論 → フィルタで評価


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


def generate_test_samples(n_test: int, rng: np.random.Generator) -> List[Dict]:
    """テスト用合成画像と GT を生成する。"""
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
        lesions_post = [{**les, "x": phantom.IMG_W - les["x"]} for les in lesions]

        base_ant, _ = phantom.get_anterior_view(add_variation=True)
        base_post, _ = phantom.get_posterior_view(add_variation=True)
        add_phys = rng.random() > 0.5
        img_ant = sim.acquire(base_ant, lesions, view="anterior", add_physiological=add_phys)
        img_post = sim.acquire(base_post, lesions_post, view="posterior", add_physiological=add_phys)

        img_ant_pad, scale_ant, px_ant, py_ant = _resize_and_pad(img_ant, ANT_W, IMG_H)
        img_post_pad, _, _, _ = _resize_and_pad(img_post, POST_W, IMG_H)
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
            "has_physio": add_phys,
        })

        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{n_test} 枚完了")

    return samples


def run_inference_low_conf(
    model, samples: List[Dict]
) -> List[List[Tuple[List[float], float]]]:
    """conf=0.05 で1回だけ推論し、前面側の検出ボックスと信頼度を保存する。"""
    print(f"  推論実行中 (conf={INFERENCE_CONF}, {len(samples)}枚)...")
    all_preds = []
    for i, s in enumerate(samples):
        res = model(s["img_rgb"], verbose=False, conf=INFERENCE_CONF)
        boxes = res[0].boxes
        preds: List[Tuple[List[float], float]] = []
        if boxes is not None and len(boxes) > 0:
            for b, c in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy()):
                cx = (b[0] + b[2]) / 2
                if cx < ANT_W:
                    preds.append((b[:4].tolist(), float(c)))
        all_preds.append(preds)
        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{len(samples)} 枚推論完了")
    return all_preds


def evaluate(
    samples: List[Dict],
    all_preds: List[List[Tuple[List[float], float]]],
    conf_thresh: float,
    suppress_conf: float | None,
) -> Dict:
    """
    conf_thresh: 検出に使う信頼度閾値
    suppress_conf: physio マスクの抑制閾値 (None = マスクなし)
    """
    from models.physio_mask import filter_physio_detections

    tp_total = fp_total = fn_total = 0
    region_tp = {k: 0 for k in REGION_KEYS}
    region_fn = {k: 0 for k in REGION_KEYS}

    for s, preds in zip(samples, all_preds):
        gt_boxes = s["gt_boxes"]
        gt_regions = s["gt_regions"]

        # Step1: conf 閾値フィルタ
        filtered = [(box, c) for box, c in preds if c >= conf_thresh]

        # Step2: 生理的集積マスク後処理
        if suppress_conf is not None:
            filtered = filter_physio_detections(filtered, suppress_conf=suppress_conf)

        pred_boxes = [box for box, _ in filtered]

        matched_gt = set()
        for pb in pred_boxes:
            best_iou, best_gi = 0.0, -1
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
    abd_recall = region_tp["abdomen_pelvis"] / (
        region_tp["abdomen_pelvis"] + region_fn["abdomen_pelvis"] + 1e-9
    )
    return {
        "precision": precision, "recall": recall, "f1": f1,
        "abdomen_recall": abd_recall,
        "tp": tp_total, "fp": fp_total, "fn": fn_total,
        "region_tp": region_tp, "region_fn": region_fn,
    }


def run_exp008(model_path: str, n_test: int = 200) -> Dict:
    from ultralytics import YOLO

    print(f"\n{'='*65}")
    print(f"  EXP-008: 生理的集積マスク後処理評価")
    print(f"  モデル: {model_path}")
    print(f"  サンプル数: {n_test}")
    print(f"{'='*65}")

    model = YOLO(model_path)
    rng = np.random.default_rng(999)

    samples = generate_test_samples(n_test, rng)
    all_preds = run_inference_low_conf(model, samples)

    # ── EXP-007 baseline (conf=0.25, no mask) ──
    baseline = evaluate(samples, all_preds, conf_thresh=0.25, suppress_conf=None)

    # ── physio mask sweep ──
    # conf_thresh=0.25 固定、suppress_conf をスイープ
    results_table = []
    for sup_conf in SUPPRESS_CONF_THRESHOLDS:
        r = evaluate(samples, all_preds, conf_thresh=0.25, suppress_conf=sup_conf)
        results_table.append((sup_conf, r))

    # ── print results ──
    print(f"\n{'─'*65}")
    print(f"  {'sup_conf':>9} {'Precision':>10} {'Recall':>8} {'F1':>7} {'腹部R':>7} {'FP':>5} {'判定':>6}")
    print(f"{'─'*65}")

    # baseline row
    p = baseline["precision"]
    r = baseline["recall"]
    f = baseline["f1"]
    ab = baseline["abdomen_recall"]
    fp = baseline["fp"]
    judge = ("✅" if p >= 0.900 else "  ") + ("腹✅" if ab >= 0.800 else "腹❌")
    print(f"  {'(EXP-007)':>9} {p:>10.3f} {r:>8.3f} {f:>7.3f} {ab:>7.3f} {fp:>5}  [baseline]")

    best_row = None
    for sup_conf, res in results_table:
        p = res["precision"]
        r = res["recall"]
        f = res["f1"]
        ab = res["abdomen_recall"]
        fp = res["fp"]
        judge = ("P✅" if p >= 0.900 else "  ") + (" 腹✅" if ab >= 0.800 else " 腹❌")
        marker = " ◀ 推奨" if (ab >= 0.800 and p == max(
            rr["precision"] for _, rr in results_table if rr["abdomen_recall"] >= 0.800
        )) else ""
        print(f"  {sup_conf:>9.2f} {p:>10.3f} {r:>8.3f} {f:>7.3f} {ab:>7.3f} {fp:>5}  {judge}{marker}")
        if best_row is None and ab >= 0.800:
            best_row = (sup_conf, res)

    # find optimal: highest Precision with abdomen_recall >= 0.800
    candidates = [(sc, res) for sc, res in results_table if res["abdomen_recall"] >= 0.800]
    if candidates:
        best_sc, best_res = max(candidates, key=lambda x: x[1]["precision"])
    else:
        best_sc, best_res = None, None

    print(f"\n{'─'*65}")
    print(f"  EXP-007 baseline : P={baseline['precision']:.3f}  腹部R={baseline['abdomen_recall']:.3f}  FP={baseline['fp']}")
    if best_res:
        dp = best_res["precision"] - baseline["precision"]
        dr = best_res["abdomen_recall"] - baseline["abdomen_recall"]
        print(f"  最良マスク設定  : suppress_conf={best_sc:.2f}")
        print(f"    Precision  : {best_res['precision']:.3f} ({dp:+.3f})")
        print(f"    Recall     : {best_res['recall']:.3f}")
        print(f"    腹部Recall : {best_res['abdomen_recall']:.3f} ({dr:+.3f})")
        print(f"    F1         : {best_res['f1']:.3f}")
        print(f"    FP 削減    : {baseline['fp']} → {best_res['fp']} ({baseline['fp'] - best_res['fp']:+d})")
        p_goal = "✅" if best_res["precision"] >= 0.900 else "❌ (目標0.900未達)"
        abd_goal = "✅" if best_res["abdomen_recall"] >= 0.800 else "❌"
        print(f"    Precision≥0.900: {p_goal}")
        print(f"    腹部Recall≥0.800: {abd_goal}")
    else:
        print("  腹部Recall≥0.800 を維持できる suppress_conf が見つかりませんでした")

    print(f"{'='*65}\n")

    return {
        "baseline": baseline,
        "results_table": results_table,
        "best_suppress_conf": best_sc,
        "best_result": best_res,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BoneScintiVision EXP-008 検証")
    parser.add_argument("--model", default=str(V7_DEFAULT_MODEL),
                        help="検証するモデルのパス (default: EXP-007 best.pt)")
    parser.add_argument("--n", type=int, default=200,
                        help="テストサンプル数 (default: 200)")
    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"ERROR: モデルが見つかりません: {args.model}")
        print(f"  訓練スクリプト: models/train_detector_v7.py")
        sys.exit(1)

    run_exp008(args.model, n_test=args.n)
