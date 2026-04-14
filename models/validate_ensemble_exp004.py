"""
BoneScintiVision — EXP-004 アンサンブル検証スクリプト

EXP-002 + EXP-003b の2モデルアンサンブル（NMS統合）。
腹部骨盤Recall 0.724 → 0.800+ を目標とする。

使い方:
  cd /Users/kohei/develop/research/BoneScintiVision
  python3.12 models/validate_ensemble_exp004.py
  python3.12 models/validate_ensemble_exp004.py --n 200
  python3.12 models/validate_ensemble_exp004.py --conf 0.20 --iou_nms 0.45
"""

import sys
import argparse
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

V2_MODEL_PATH   = BASE_DIR / "runs" / "bone_scinti_detector_v2" / "weights" / "best.pt"
V3B_MODEL_PATH  = BASE_DIR / "runs" / "detect" / "bone_scinti_detector_v3b" / "weights" / "best.pt"

ANT_W = POST_W = IMG_H = 256


def compute_iou(box1, box2) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter + 1e-9)


def nms_boxes(boxes_confs: List[tuple], iou_thresh: float = 0.45) -> List[list]:
    """簡易NMS: (box, conf) リストから重複除去。confが高いものを優先。"""
    if not boxes_confs:
        return []
    boxes_confs = sorted(boxes_confs, key=lambda x: x[1], reverse=True)
    kept = []
    suppressed = set()
    for i, (box_i, conf_i) in enumerate(boxes_confs):
        if i in suppressed:
            continue
        kept.append(box_i)
        for j, (box_j, _) in enumerate(boxes_confs[i + 1:], start=i + 1):
            if j not in suppressed and compute_iou(box_i, box_j) >= iou_thresh:
                suppressed.add(j)
    return kept


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


def run_ensemble(model_path_v2: str, model_path_v3b: str,
                 n_test: int = 200, conf: float = 0.25,
                 iou_nms: float = 0.45, iou_match: float = 0.3) -> Dict:
    from ultralytics import YOLO
    from synth.bone_phantom import BonePhantom
    from synth.scintigraphy_sim import ScintSim

    model_v2  = YOLO(model_path_v2)
    model_v3b = YOLO(model_path_v3b)

    rng = np.random.default_rng(999)
    region_keys = ["head_neck", "thorax", "abdomen_pelvis", "extremities"]
    tp_total = fp_total = fn_total = 0
    region_tp = {k: 0 for k in region_keys}
    region_fn = {k: 0 for k in region_keys}
    count_errors = []

    print(f"  アンサンブル評価 ({n_test}枚, conf>{conf}, NMS IoU>{iou_nms})...")
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
        img_ant = sim.acquire(base_ant, lesions,       view="anterior",  add_physiological=True)
        img_post = sim.acquire(base_post, lesions_post, view="posterior", add_physiological=True)

        img_ant_pad, scale_ant, px_ant, py_ant = _resize_and_pad(img_ant, ANT_W, IMG_H)
        img_post_pad, _, _, _ = _resize_and_pad(img_post, POST_W, IMG_H)
        dual_img = np.hstack([img_ant_pad, img_post_pad])
        img_rgb = cv2.cvtColor(dual_img, cv2.COLOR_GRAY2RGB)

        # 両モデルで推論
        res_v2  = model_v2(img_rgb,  verbose=False, conf=conf)
        res_v3b = model_v3b(img_rgb, verbose=False, conf=conf)

        boxes_confs = []
        for res in [res_v2, res_v3b]:
            boxes = res[0].boxes
            if boxes is not None and len(boxes) > 0:
                for b, c in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy()):
                    boxes_confs.append((b[:4].tolist(), float(c)))

        # NMSで統合
        merged = nms_boxes(boxes_confs, iou_thresh=iou_nms)

        # 前面側のみ評価
        pred_ant = [pb for pb in merged if pb[0] < ANT_W and pb[2] < ANT_W + 20]

        # GT
        gt_boxes = []
        gt_regions = []
        for les in lesions:
            x = les["x"] * scale_ant + px_ant
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
            if best_iou >= iou_match and best_gi >= 0:
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
    recall    = tp_total / (tp_total + fn_total + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    mae_count = float(np.mean(np.abs(count_errors)))
    me_count  = float(np.mean(count_errors))

    print(f"\n{'='*60}")
    print(f"  EXP-004 アンサンブル評価 (EXP-002 + EXP-003b, n={n_test})")
    print(f"  conf>{conf}, NMS IoU>{iou_nms}")
    print(f"  TP={tp_total}  FP={fp_total}  FN={fn_total}")
    print(f"  Precision : {precision:.3f}")
    print(f"  Recall    : {recall:.3f}")
    print(f"  F1        : {f1:.3f}")
    print(f"  病変数誤差: MAE={mae_count:.2f}  ME={me_count:+.2f}")
    print(f"\n  部位別 Recall (前面):")
    region_recall = {}
    for rk in region_keys:
        r_tp = region_tp[rk]
        r_fn = region_fn[rk]
        r_rec = r_tp / (r_tp + r_fn + 1e-9)
        region_recall[rk] = r_rec
        target = "← 目標 0.800 ✅" if rk == "abdomen_pelvis" and r_rec >= 0.800 else \
                 "← 目標 0.800 ❌" if rk == "abdomen_pelvis" else ""
        print(f"    {rk:20s}: {r_rec:.3f}  (TP={r_tp}, FN={r_fn}) {target}")

    print(f"\n  比較:")
    print(f"    EXP-002 腹部Recall : 0.711")
    print(f"    EXP-003b 腹部Recall: 0.724")
    print(f"    EXP-004 腹部Recall : {region_recall.get('abdomen_pelvis', 0):.3f}")
    print(f"{'='*60}")

    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp_total, "fp": fp_total, "fn": fn_total,
            "mae_count": mae_count, "me_count": me_count,
            "region_recall": region_recall}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BoneScintiVision EXP-004 アンサンブル評価")
    parser.add_argument("--model_v2",  default=str(V2_MODEL_PATH))
    parser.add_argument("--model_v3b", default=str(V3B_MODEL_PATH))
    parser.add_argument("--n",        type=int,   default=200)
    parser.add_argument("--conf",     type=float, default=0.25)
    parser.add_argument("--iou_nms",  type=float, default=0.45)
    args = parser.parse_args()

    for path, name in [(args.model_v2, "EXP-002"), (args.model_v3b, "EXP-003b")]:
        if not Path(path).exists():
            print(f"ERROR: {name}モデルが見つかりません: {path}")
            sys.exit(1)

    print("BoneScintiVision EXP-004 アンサンブル評価")
    print(f"  EXP-002  : {args.model_v2}")
    print(f"  EXP-003b : {args.model_v3b}")
    run_ensemble(args.model_v2, args.model_v3b,
                 n_test=args.n, conf=args.conf, iou_nms=args.iou_nms)
