"""Draw predictions (and ground truth) on images, then tile them into contact sheets.

With a noisy validation metric, looking at the boxes beats reading mAP. This produces:

  <out>/images/<name>.jpg   one annotated image per frame
  <out>/grid_001.jpg ...    contact sheets, N frames tiled into one image
  <out>/summary.json        per-frame miss / false-positive counts

Ground truth is drawn in GREEN, predictions in RED with their confidence. Where they
overlap well you see a red box hugging a green one; a lone green box is a MISS, a lone
red box is a FALSE ALARM. That reads at a glance on a contact sheet.

Usage
-----
    # after export_edge.py (no torch needed)
    python tools/visualize_predictions.py \
        --model deploy_out/model_fp32.onnx \
        --images data/custom/images/test \
        --labels data/custom/labels/test \
        --classes floatingsludge sludge \
        --out viz_test --limit 30 --grid 6x5 --sample worst

    # or straight from a training checkpoint
    python tools/visualize_predictions.py \
        --model runs/train/run1/checkpoints/best-030-0.6747.ckpt \
        --model-cfg v9-s --image-size 640 640 \
        --images data/custom/images/test --labels data/custom/labels/test \
        --classes floatingsludge sludge --out viz_test

`--sample worst` ranks frames by how badly the model did (misses + false alarms) and
shows those first. That is almost always what you want: the confident correct frames
teach you nothing.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}

GT_COLOUR = (60, 220, 60)
PRED_COLOUR = (255, 70, 70)


# ----------------------------------------------------------------------------------
# model backends
# ----------------------------------------------------------------------------------
class OnnxBackend:
    def __init__(self, path: Path):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        shape = self.session.get_inputs()[0].shape
        self.image_size = (int(shape[3]), int(shape[2]))  # (W, H)

    def __call__(self, image: Image.Image, conf: float, iou: float):
        from deploy.rpi_infer import postprocess, preprocess

        tensor, scale, pad_x, pad_y = preprocess(image, self.image_size)
        boxes, scores = self.session.run(None, {self.input_name: tensor})
        return postprocess(boxes, scores, scale, pad_x, pad_y, conf, iou)


class CheckpointBackend:
    """Runs a Lightning .ckpt directly, using the same deploy graph the exporter builds."""

    def __init__(self, path: Path, model_cfg: str, num_classes: int, image_size):
        import torch
        from omegaconf import OmegaConf

        from yolo.tools.export import build_deploy_model

        self.torch = torch
        self.image_size = tuple(image_size)
        cfg = OmegaConf.create(
            {
                "model": OmegaConf.load(REPO_ROOT / "yolo" / "config" / "model" / f"{model_cfg}.yaml"),
                "dataset": {"class_num": num_classes},
                "image_size": list(image_size),
            }
        )
        self.model = build_deploy_model(cfg, path, prefer_ema=True)

    def __call__(self, image: Image.Image, conf: float, iou: float):
        from deploy.rpi_infer import postprocess, preprocess

        tensor, scale, pad_x, pad_y = preprocess(image, self.image_size)
        with self.torch.no_grad():
            boxes, scores = self.model(self.torch.from_numpy(tensor))
        return postprocess(boxes.numpy(), scores.numpy(), scale, pad_x, pad_y, conf, iou)


# ----------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------
def load_ground_truth(label_path: Path, width: int, height: int):
    """YOLO detection rows -> [(cls, x1, y1, x2, y2)] in pixels."""
    if not label_path.is_file():
        return []
    boxes = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls, cx, cy, bw, bh = (float(p) for p in parts)
        boxes.append(
            (int(cls),
             (cx - bw / 2) * width, (cy - bh / 2) * height,
             (cx + bw / 2) * width, (cy + bh / 2) * height)
        )
    return boxes


def iou_matrix(a, b):
    """a: (N,4), b: (M,4) xyxy -> (N,M) IoU."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def score_frame(gt, preds, iou_thresh=0.5):
    """Greedy match -> (misses, false_alarms, matched)."""
    gt_boxes = [g[1:] for g in gt]
    pred_boxes = [p[:4] for p in preds]
    ious = iou_matrix(gt_boxes, pred_boxes)

    matched_gt, matched_pred = set(), set()
    if ious.size:
        order = np.dstack(np.unravel_index(np.argsort(-ious, axis=None), ious.shape))[0]
        for gi, pi in order:
            if ious[gi, pi] < iou_thresh:
                break
            if gi in matched_gt or pi in matched_pred:
                continue
            matched_gt.add(int(gi))
            matched_pred.add(int(pi))
    return len(gt) - len(matched_gt), len(preds) - len(matched_pred), len(matched_gt)


def _font(size=16):
    for name in ("DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, int(size))
        except OSError:
            continue
    return ImageFont.load_default()


def _tag(draw, x, y, text, fill, font, anchor_top=True):
    """Draw text on an opaque plate so it stays readable over any background."""
    box = draw.textbbox((x, y), text, font=font)
    pad = 2
    draw.rectangle([box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad], fill=(0, 0, 0))
    draw.text((x, y), text, fill=fill, font=font)
    return box[3] - box[1]


def annotate(image, gt, preds, class_names, show_gt=True, target_width=None):
    """Draw GT (green) and predictions (red), both labelled with their class.

    If `target_width` is given the image is resized FIRST and the boxes scaled to match,
    so label text stays legible. Annotating at full size and shrinking afterwards — which
    is what a contact sheet would otherwise do — renders the text unreadable.
    """
    canvas = image.convert("RGB")
    scale = 1.0
    if target_width and canvas.width > target_width:
        scale = target_width / canvas.width
        canvas = canvas.resize(
            (max(1, int(canvas.width * scale)), max(1, int(canvas.height * scale))),
            Image.Resampling.LANCZOS,
        )
    else:
        canvas = canvas.copy()

    draw = ImageDraw.Draw(canvas)
    line = max(2, canvas.width // 320)
    font = _font(max(11, canvas.width / 38))

    if show_gt:
        for cls, x1, y1, x2, y2 in gt:
            x1, y1, x2, y2 = (v * scale for v in (x1, y1, x2, y2))
            draw.rectangle([x1, y1, x2, y2], outline=GT_COLOUR, width=line)
            name = class_names[int(cls)] if int(cls) < len(class_names) else str(int(cls))
            # GT label sits BELOW the box so it never collides with the prediction label.
            _tag(draw, x1 + 2, min(canvas.height - font.size - 2, y2 + 2), name, GT_COLOUR, font)

    for x1, y1, x2, y2, score, cls in preds:
        x1, y1, x2, y2 = (v * scale for v in (x1, y1, x2, y2))
        draw.rectangle([x1, y1, x2, y2], outline=PRED_COLOUR, width=line)
        name = class_names[int(cls)] if int(cls) < len(class_names) else str(int(cls))
        _tag(draw, x1 + 2, max(0, y1 - font.size - 4), f"{name} {score:.2f}", PRED_COLOUR, font)

    legend = _font(max(10, canvas.width / 46))
    _tag(draw, 4, 4, "GT", GT_COLOUR, legend)
    _tag(draw, 34, 4, "pred", PRED_COLOUR, legend)

    return canvas


def build_grid(records, class_names, cols, rows, show_gt, cell=(520, 400), pad=6, bg=(24, 24, 28)):
    """Tile frames into one contact sheet, annotating each at thumbnail size."""
    cw, ch = cell
    caption_h = 24
    sheet = Image.new("RGB", (cols * (cw + pad) + pad, rows * (ch + caption_h + pad) + pad), bg)
    draw = ImageDraw.Draw(sheet)
    font = _font(14)

    for index, rec in enumerate(records):
        r, c = divmod(index, cols)
        if r >= rows:
            break
        with Image.open(rec["path"]) as im:
            thumb = annotate(im, rec["gt"], rec["preds"], class_names,
                             show_gt=show_gt, target_width=cw)
        if thumb.height > ch:
            ratio = ch / thumb.height
            thumb = thumb.resize((int(thumb.width * ratio), ch), Image.Resampling.LANCZOS)
        x = pad + c * (cw + pad) + (cw - thumb.width) // 2
        y = pad + r * (ch + caption_h + pad)
        sheet.paste(thumb, (x, y))
        caption = f"{rec['path'].stem}  ok:{rec['matched']} miss:{rec['misses']} fp:{rec['false_alarms']}"
        draw.text((pad + c * (cw + pad), y + ch + 4), caption, fill=(210, 210, 210), font=font)

    return sheet


# ----------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True, help=".onnx or .ckpt")
    parser.add_argument("--model-cfg", default="v9-s", help="model config name (only for .ckpt)")
    parser.add_argument("--image-size", type=int, nargs=2, default=[640, 640], help="only for .ckpt")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=None, help="ground-truth .txt dir (optional)")
    parser.add_argument("--classes", nargs="+", required=True)
    parser.add_argument("--out", type=Path, default=Path("viz_out"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--limit", type=int, default=30, help="how many frames to render")
    parser.add_argument("--grid", default="6x5", help="contact sheet layout, COLSxROWS")
    parser.add_argument("--sample", choices=["worst", "random", "first"], default="worst")
    parser.add_argument("--no-gt", action="store_true", help="draw predictions only")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cols, rows = (int(v) for v in args.grid.lower().split("x"))
    per_sheet = cols * rows

    images = sorted(p for p in args.images.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise SystemExit(f"No images under {args.images}")

    if args.model.suffix.lower() == ".onnx":
        backend = OnnxBackend(args.model)
    else:
        backend = CheckpointBackend(args.model, args.model_cfg, len(args.classes), args.image_size)

    # `worst` needs every frame scored before choosing; the others do not.
    pool = images
    if args.sample == "random":
        random.seed(args.seed)
        pool = random.sample(images, min(len(images), max(args.limit * 3, args.limit)))
    elif args.sample == "first":
        pool = images[: args.limit]

    print(f"scoring {len(pool)} frames ...")
    records = []
    for i, path in enumerate(pool, 1):
        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
            preds = backend(im, args.conf, args.iou)
            gt = [] if (args.labels is None or args.no_gt) else load_ground_truth(
                args.labels / f"{path.stem}.txt", w, h)
            misses, false_alarms, matched = score_frame(gt, preds)
            records.append({"path": path, "gt": gt, "preds": preds,
                            "misses": misses, "false_alarms": false_alarms, "matched": matched})
        if i % 50 == 0 or i == len(pool):
            print(f"  {i}/{len(pool)}")

    if args.sample == "worst":
        records.sort(key=lambda r: (r["misses"] + r["false_alarms"], r["misses"]), reverse=True)
    records = records[: args.limit]

    out_images = args.out / "images"
    out_images.mkdir(parents=True, exist_ok=True)

    # Full-size annotated frames, one file each.
    for rec in records:
        with Image.open(rec["path"]) as im:
            canvas = annotate(im, rec["gt"], rec["preds"], args.classes, show_gt=not args.no_gt)
        canvas.save(out_images / f"{rec['path'].stem}.jpg", quality=92)

    # Contact sheets re-annotate at thumbnail scale so the labels stay readable.
    sheets = 0
    for start in range(0, len(records), per_sheet):
        sheet = build_grid(records[start:start + per_sheet], args.classes, cols, rows,
                           show_gt=not args.no_gt)
        sheets += 1
        sheet.save(args.out / f"grid_{sheets:03d}.jpg", quality=92)

    total_miss = sum(r["misses"] for r in records)
    total_fp = sum(r["false_alarms"] for r in records)
    total_ok = sum(r["matched"] for r in records)

    (args.out / "summary.json").write_text(json.dumps({
        "model": str(args.model), "conf": args.conf, "frames_rendered": len(records),
        "sampling": args.sample,
        "totals": {"matched": total_ok, "missed": total_miss, "false_alarms": total_fp},
        "frames": [{"image": r["path"].name, "matched": r["matched"],
                    "missed": r["misses"], "false_alarms": r["false_alarms"]} for r in records],
    }, indent=2))

    print(f"\n{'=' * 58}")
    print(f"rendered {len(records)} frames ({args.sample}) -> {out_images}")
    print(f"contact sheets: {sheets} x {cols}x{rows} -> {args.out}/grid_*.jpg")
    print(f"  matched      {total_ok}")
    print(f"  missed       {total_miss}   (green box with no red = model failed to detect)")
    print(f"  false alarms {total_fp}   (red box with no green = model invented one)")
    print("=" * 58)


if __name__ == "__main__":
    main()
