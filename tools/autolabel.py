"""Auto-label images with an existing detector, for human correction.

Purpose
-------
Bootstrap a labelled dataset: run a model you already have over unlabelled images,
write YOLO-format `.txt` labels, and hand them to a human to fix. Correcting boxes is
several times faster than drawing them from scratch.

Backends (chosen by file extension):
  *.pt   via Ultralytics  - your existing YOLOv8/v9/v11 model
  *.onnx via onnxruntime  - a model exported by tools/export_edge.py from THIS repo

Why both: once you have trained an MIT model here, you can relabel the next batch with
it and drop the Ultralytics dependency entirely. That loop gets faster each round.

Licensing note
--------------
Ultralytics is AGPL-3.0. Running it locally to produce annotations does not distribute
it, and the resulting labels describe *your* images. The model trained here from those
labels is a separate MIT artifact containing no Ultralytics code. Keep the two separate
and do not ship the Ultralytics weights.

Usage
-----
    # bootstrap from an Ultralytics model, mapping COCO 'person' onto your class 0
    python tools/autolabel.py --model yolov8n.pt --images data/raw \
        --out data/custom --split train --classes person --conf 0.35 --copy-images

    # later: relabel the next batch with your own exported MIT model
    python tools/autolabel.py --model deploy_out/model_fp32.onnx --images data/raw_batch2 \
        --out data/custom --split train --classes floc foam scum --conf 0.35
"""

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ----------------------------------------------------------------------------------
# Backends. Each returns detections as (class_name, confidence, x1, y1, x2, y2) in
# ORIGINAL image pixel coordinates.
# ----------------------------------------------------------------------------------
class UltralyticsBackend:
    def __init__(self, model_path: Path, imgsz: int):
        # Imported lazily so that ONNX-only users never load the AGPL dependency.
        from ultralytics import YOLO

        self.model = YOLO(str(model_path))
        self.imgsz = imgsz
        self.names = self.model.names
        print(f"Ultralytics backend | {len(self.names)} classes | imgsz={imgsz}")

    def class_names(self):
        return list(self.names.values())

    def predict(self, image_path: Path, conf: float, iou: float):
        result = self.model.predict(str(image_path), imgsz=self.imgsz, conf=conf, iou=iou, verbose=False)[0]
        detections = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append((self.names[int(box.cls.item())], float(box.conf.item()), x1, y1, x2, y2))
        return detections


class OnnxBackend:
    """Runs a graph exported by tools/export_edge.py (outputs boxes + sigmoid scores)."""

    def __init__(self, model_path: Path, class_names):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

        shape = self.session.get_inputs()[0].shape
        self.image_size = (int(shape[3]), int(shape[2]))  # (W, H)

        num_classes = int(self.session.get_outputs()[1].shape[2])
        if class_names is None or len(class_names) != num_classes:
            raise SystemExit(
                f"This ONNX model has {num_classes} classes but --classes supplied "
                f"{0 if class_names is None else len(class_names)}. They must match, in training order."
            )
        self.names = list(class_names)
        print(f"ONNX backend | {num_classes} classes | input={self.image_size}")

    def class_names(self):
        return list(self.names)

    def predict(self, image_path: Path, conf: float, iou: float):
        from deploy.rpi_infer import postprocess, preprocess

        with Image.open(image_path) as image:
            tensor, scale, pad_x, pad_y = preprocess(image.convert("RGB"), self.image_size)

        boxes, scores = self.session.run(None, {self.input_name: tensor})
        detections = postprocess(boxes, scores, scale, pad_x, pad_y, conf, iou)

        return [
            (self.names[int(cls_id)], float(score), float(x1), float(y1), float(x2), float(y2))
            for x1, y1, x2, y2, score, cls_id in detections
        ]


# ----------------------------------------------------------------------------------
def build_class_map(source_names, target_classes, explicit_map):
    """Map a source class NAME to a target class INDEX. Unmapped classes are dropped."""
    if explicit_map:
        mapping = {}
        for pair in explicit_map.split(","):
            if ":" not in pair:
                raise SystemExit(f"--map entries must look like 'srcname:dstidx', got {pair!r}")
            source, target = pair.rsplit(":", 1)
            target_index = int(target)
            if not 0 <= target_index < len(target_classes):
                raise SystemExit(f"--map target index {target_index} is outside --classes (0..{len(target_classes)-1})")
            mapping[source.strip()] = target_index
        return mapping

    # Default: match by name, case-insensitively.
    lowered = {name.lower(): index for index, name in enumerate(target_classes)}
    return {name: lowered[name.lower()] for name in source_names if name.lower() in lowered}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True, help=".pt (Ultralytics) or .onnx (this repo)")
    parser.add_argument("--images", type=Path, required=True, help="directory of unlabelled images")
    parser.add_argument("--out", type=Path, required=True, help="dataset root, e.g. data/custom")
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--classes", nargs="+", required=True, help="TARGET class list, in training order")
    parser.add_argument("--map", dest="class_map", default=None,
                        help="explicit 'srcname:dstidx' pairs, e.g. 'person:0,boat:1'")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640, help="Ultralytics inference size")
    parser.add_argument("--copy-images", action="store_true", help="copy images into <out>/images/<split>/")
    parser.add_argument("--review-below", type=float, default=0.6,
                        help="flag images whose best detection scores below this, for priority review")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    images = sorted(path for path in args.images.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    if args.limit:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"No images found under {args.images}")

    if args.model.suffix.lower() == ".onnx":
        backend = OnnxBackend(args.model, args.classes)
    else:
        backend = UltralyticsBackend(args.model, args.imgsz)

    class_map = build_class_map(backend.class_names(), args.classes, args.class_map)
    if not class_map:
        raise SystemExit(
            "No source class maps onto a target class.\nSource classes: "
            + ", ".join(sorted(backend.class_names()))
            + "\nConnect them explicitly with --map 'srcname:dstidx'."
        )
    print(f"class map: {class_map}")

    labels_dir = args.out / "labels" / args.split
    labels_dir.mkdir(parents=True, exist_ok=True)
    images_dir = args.out / "images" / args.split
    if args.copy_images:
        images_dir.mkdir(parents=True, exist_ok=True)

    counts, empty, review = Counter(), [], []

    for index, image_path in enumerate(images, 1):
        detections = backend.predict(image_path, args.conf, args.iou)

        with Image.open(image_path) as image:
            width, height = image.size

        rows, best_conf = [], 0.0
        for name, score, x1, y1, x2, y2 in detections:
            if name not in class_map:
                continue
            # xyxy pixels -> normalized cx cy w h, clipped to the frame
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(width), x2), min(float(height), y2)
            if x2 <= x1 or y2 <= y1:
                continue
            target = class_map[name]
            center_x, center_y = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
            box_w, box_h = (x2 - x1) / width, (y2 - y1) / height
            rows.append(f"{target} {center_x:.6f} {center_y:.6f} {box_w:.6f} {box_h:.6f}")
            counts[args.classes[target]] += 1
            best_conf = max(best_conf, score)

        # An empty .txt is a deliberate, valid background sample - always write the file.
        (labels_dir / f"{image_path.stem}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))

        if args.copy_images:
            shutil.copy2(image_path, images_dir / image_path.name)

        if not rows:
            empty.append(image_path.name)
        elif best_conf < args.review_below:
            review.append((image_path.name, best_conf))

        if index % 50 == 0 or index == len(images):
            print(f"  {index}/{len(images)} images")

    print("\n" + "=" * 64)
    print(f"Wrote {len(images)} label files to {labels_dir}")
    for name, count in counts.most_common():
        print(f"  {name:<18} {count}")
    print(f"  {'(no detections)':<18} {len(empty)} images")
    print("=" * 64)
    print("\nThese labels are a DRAFT. Review them before training:")
    print("  - the model cannot invent a class it was never taught")
    print("  - missed objects become background and actively teach the wrong thing")
    print(f"  - {len(review)} images scored below {args.review_below}; review those first")

    report_path = args.out / f"autolabel_report_{args.split}.json"
    report_path.write_text(
        json.dumps(
            {
                "model": str(args.model),
                "images": len(images),
                "conf_threshold": args.conf,
                "class_map": class_map,
                "class_counts": dict(counts),
                "empty_images": empty,
                "low_confidence": [
                    {"image": name, "best_conf": round(conf, 3)}
                    for name, conf in sorted(review, key=lambda item: item[1])
                ],
            },
            indent=2,
        )
    )
    print(f"\nReport (incl. priority-review list): {report_path}")


if __name__ == "__main__":
    main()
