"""Find the confidence threshold that best matches what you care about.

The default 0.25 is an arbitrary inherited number. The right value depends on whether a
missed detection or a false alarm costs you more, and that is a business decision, not a
modelling one. This measures the trade-off so you can pick deliberately.

Inference runs ONCE at a very low threshold; every candidate threshold is then evaluated
against those cached detections. Sweeping 19 thresholds costs the same as one pass.

    precision = of the boxes predicted, how many were real     (low -> false alarms)
    recall    = of the real objects, how many were found       (low -> missed sludge)
    F-beta    = their harmonic mean; beta weights recall

      --beta 1.0   balanced          (default)
      --beta 2.0   recall matters 2x  <- use if MISSING sludge is the costly error
      --beta 0.5   precision matters 2x <- use if FALSE ALARMS are the costly error

Usage
-----
    python tools/tune_confidence.py \
        --model runs/train/sludge-v9s-run1/checkpoints/best-030-0.6747.ckpt \
        --model-cfg v9-s --image-size 640 640 \
        --images data/custom/images/val --labels data/custom/labels/val \
        --classes floatingsludge sludge --beta 2.0

Pick the threshold on VAL, never on test - choosing it on test turns test into a second
validation set and you lose your honest estimate.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.visualize_predictions import (  # noqa: E402
    CheckpointBackend,
    OnnxBackend,
    iou_matrix,
    load_ground_truth,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def match(gt_boxes, pred_boxes, iou_thresh=0.5):
    """Greedy highest-IoU-first matching -> number of true positives."""
    ious = iou_matrix(gt_boxes, pred_boxes)
    if ious.size == 0:
        return 0
    matched_gt, matched_pred = set(), set()
    order = np.dstack(np.unravel_index(np.argsort(-ious, axis=None), ious.shape))[0]
    for gi, pi in order:
        if ious[gi, pi] < iou_thresh:
            break
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(int(gi))
        matched_pred.add(int(pi))
    return len(matched_gt)


def evaluate(frames, threshold, num_classes, iou_thresh=0.5):
    """Per-class and overall TP/FP/FN at one confidence threshold."""
    stats = {c: {"tp": 0, "fp": 0, "fn": 0} for c in range(num_classes)}

    for gt, preds in frames:
        for cls in range(num_classes):
            gt_c = [g[1:] for g in gt if int(g[0]) == cls]
            pred_c = [p[:4] for p in preds if int(p[5]) == cls and p[4] >= threshold]
            tp = match(gt_c, pred_c, iou_thresh)
            stats[cls]["tp"] += tp
            stats[cls]["fp"] += len(pred_c) - tp
            stats[cls]["fn"] += len(gt_c) - tp

    return stats


def prf(tp, fp, fn, beta=1.0):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0:
        return precision, recall, 0.0
    b2 = beta * beta
    f = (1 + b2) * precision * recall / (b2 * precision + recall)
    return precision, recall, f


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True, help=".ckpt or .onnx")
    parser.add_argument("--model-cfg", default="v9-s", help="model config name (only for .ckpt)")
    parser.add_argument("--image-size", type=int, nargs=2, default=[640, 640])
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--classes", nargs="+", required=True)
    parser.add_argument("--beta", type=float, default=1.0,
                        help="F-beta weight: >1 favours recall, <1 favours precision")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU for counting a match")
    parser.add_argument("--nms-iou", type=float, default=0.45)
    parser.add_argument("--min-conf", type=float, default=0.01,
                        help="inference floor; thresholds below this cannot be evaluated")
    parser.add_argument("--steps", type=int, default=19, help="thresholds between 0.05 and 0.95")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None, help="write results as JSON")
    args = parser.parse_args()

    num_classes = len(args.classes)
    images = sorted(p for p in args.images.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if args.limit:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"No images under {args.images}")

    backend = (OnnxBackend(args.model) if args.model.suffix.lower() == ".onnx"
               else CheckpointBackend(args.model, args.model_cfg, num_classes, args.image_size))

    print(f"running inference once over {len(images)} frames at conf>={args.min_conf} ...")
    frames = []
    for i, path in enumerate(images, 1):
        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
            preds = backend(im, args.min_conf, args.nms_iou)
        gt = load_ground_truth(args.labels / f"{path.stem}.txt", w, h)
        frames.append((gt, preds))
        if i % 50 == 0 or i == len(images):
            print(f"  {i}/{len(images)}")

    thresholds = np.linspace(0.05, 0.95, args.steps)
    rows = []
    for t in thresholds:
        stats = evaluate(frames, t, num_classes, args.iou)
        tp = sum(s["tp"] for s in stats.values())
        fp = sum(s["fp"] for s in stats.values())
        fn = sum(s["fn"] for s in stats.values())
        p, r, f = prf(tp, fp, fn, args.beta)
        rows.append({"threshold": round(float(t), 3), "precision": p, "recall": r, "fbeta": f,
                     "tp": tp, "fp": fp, "fn": fn,
                     "per_class": {args.classes[c]: dict(stats[c],
                                   **dict(zip(("precision", "recall", "fbeta"),
                                              prf(stats[c]["tp"], stats[c]["fp"], stats[c]["fn"], args.beta))))
                                   for c in range(num_classes)}})

    label = f"F{args.beta:g}"
    print(f"\n{'conf':>6} {'precision':>10} {'recall':>8} {label:>8} {'TP':>6} {'FP':>6} {'FN':>6}")
    print("-" * 56)
    best = max(rows, key=lambda r: r["fbeta"])
    for r in rows:
        mark = "  <-- best" if r is best else ""
        print(f"{r['threshold']:>6.2f} {r['precision']:>10.3f} {r['recall']:>8.3f} "
              f"{r['fbeta']:>8.3f} {r['tp']:>6} {r['fp']:>6} {r['fn']:>6}{mark}")

    print(f"\nBest {label}: threshold {best['threshold']:.2f}  "
          f"(precision {best['precision']:.3f}, recall {best['recall']:.3f})")

    print(f"\nper class at threshold {best['threshold']:.2f}")
    print(f"  {'class':<18} {'precision':>10} {'recall':>8} {label:>8} {'TP':>5} {'FP':>5} {'FN':>5}")
    for name, s in best["per_class"].items():
        print(f"  {name:<18} {s['precision']:>10.3f} {s['recall']:>8.3f} {s['fbeta']:>8.3f} "
              f"{s['tp']:>5} {s['fp']:>5} {s['fn']:>5}")

    # Operating points someone will ask for.
    hi_p = max((r for r in rows if r["precision"] >= 0.95), key=lambda r: r["recall"], default=None)
    hi_r = max((r for r in rows if r["recall"] >= 0.95), key=lambda r: r["precision"], default=None)
    print("\noperating points")
    print(f"  balanced ({label})      : conf {best['threshold']:.2f}")
    if hi_p:
        print(f"  precision >= 0.95     : conf {hi_p['threshold']:.2f}  (recall {hi_p['recall']:.3f})")
    else:
        print("  precision >= 0.95     : not reachable at any threshold")
    if hi_r:
        print(f"  recall    >= 0.95     : conf {hi_r['threshold']:.2f}  (precision {hi_r['precision']:.3f})")
    else:
        print("  recall    >= 0.95     : not reachable at any threshold")

    if args.out:
        args.out.write_text(json.dumps({"model": str(args.model), "beta": args.beta,
                                        "iou": args.iou, "frames": len(images),
                                        "best": best, "sweep": rows}, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
