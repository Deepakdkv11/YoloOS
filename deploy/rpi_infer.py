"""Standalone Raspberry Pi 4 inference for an exported YOLO ONNX model.

Deliberately depends on nothing but onnxruntime + numpy + Pillow: no torch, no
lightning, no hydra, none of the training stack. Copy this single file and your
`model_int8.onnx` onto the Pi.

Install on Raspberry Pi OS (64-bit — do not use the 32-bit image, the ARM64 wheels are
substantially faster and onnxruntime ships no optimised 32-bit build):

    sudo apt install -y python3-pip libatlas-base-dev
    pip3 install onnxruntime numpy pillow

Run:

    python3 rpi_infer.py --model model_int8.onnx --source frame.jpg --classes floc foam scum
    python3 rpi_infer.py --model model_int8.onnx --source 0 --benchmark
"""

import argparse
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw


# ----------------------------------------------------------------------------------
# pre-processing (must mirror training's PadAndResize exactly)
# ----------------------------------------------------------------------------------
def letterbox(image: Image.Image, size: Tuple[int, int], color=(114, 114, 114)):
    target_w, target_h = size
    img_w, img_h = image.size
    scale = min(target_w / img_w, target_h / img_h)
    new_w, new_h = int(img_w * scale), int(img_h * scale)

    resized = image.convert("RGB").resize((new_w, new_h), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (target_w, target_h), color)
    pad_x, pad_y = (target_w - new_w) // 2, (target_h - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))
    return canvas, scale, pad_x, pad_y


def preprocess(image: Image.Image, size: Tuple[int, int]):
    canvas, scale, pad_x, pad_y = letterbox(image, size)
    array = np.asarray(canvas, dtype=np.float32) / 255.0
    tensor = np.transpose(array, (2, 0, 1))[None]
    return np.ascontiguousarray(tensor), scale, pad_x, pad_y


# ----------------------------------------------------------------------------------
# post-processing
# ----------------------------------------------------------------------------------
def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
    """Plain greedy NMS. Runs on the few dozen boxes that clear the score threshold,
    so a vectorised numpy implementation is far cheaper here than an in-graph ONNX op."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        best = order[0]
        keep.append(int(best))
        if order.size == 1:
            break
        rest = order[1:]

        xx1 = np.maximum(x1[best], x1[rest])
        yy1 = np.maximum(y1[best], y1[rest])
        xx2 = np.minimum(x2[best], x2[rest])
        yy2 = np.minimum(y2[best], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[best] + areas[rest] - inter + 1e-9)

        order = rest[iou <= iou_threshold]
    return keep


def postprocess(
    boxes: np.ndarray,
    scores: np.ndarray,
    scale: float,
    pad_x: int,
    pad_y: int,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    max_detections: int = 300,
):
    """Graph output -> detections in ORIGINAL image pixel coordinates."""
    boxes, scores = boxes[0], scores[0]  # drop batch

    class_ids = scores.argmax(axis=1)
    confidences = scores[np.arange(len(scores)), class_ids]

    keep_mask = confidences > conf_threshold
    if not keep_mask.any():
        return np.zeros((0, 6), dtype=np.float32)

    boxes, confidences, class_ids = boxes[keep_mask], confidences[keep_mask], class_ids[keep_mask]

    # Per-class NMS via a coordinate offset, so boxes of different classes never suppress
    # each other. Offset is computed in float64 to avoid the precision collision that
    # bites the float16 path in the training repo.
    offsets = class_ids.astype(np.float64) * 8192.0
    keep = nms(boxes.astype(np.float64) + offsets[:, None], confidences, iou_threshold)[:max_detections]

    boxes, confidences, class_ids = boxes[keep], confidences[keep], class_ids[keep]

    # undo letterbox: subtract padding, then divide by scale
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale

    return np.concatenate([boxes, confidences[:, None], class_ids[:, None].astype(np.float32)], axis=1)


# ----------------------------------------------------------------------------------
# runtime
# ----------------------------------------------------------------------------------
class Detector:
    def __init__(self, model_path: Path, num_threads: int = 4):
        options = ort.SessionOptions()
        # The Pi 4 has exactly 4 A72 cores. Oversubscribing makes latency worse, and
        # leaving this at the default can spawn more threads than cores under load.
        options.intra_op_num_threads = num_threads
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        self.session = ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

        shape = self.session.get_inputs()[0].shape  # [1, 3, H, W], static
        self.image_size = (int(shape[3]), int(shape[2]))  # (W, H)
        print(f"Loaded {Path(model_path).name}  input={shape}  threads={num_threads}")

    def __call__(self, image: Image.Image, conf_threshold=0.25, iou_threshold=0.45):
        tensor, scale, pad_x, pad_y = preprocess(image, self.image_size)
        boxes, scores = self.session.run(None, {self.input_name: tensor})
        return postprocess(boxes, scores, scale, pad_x, pad_y, conf_threshold, iou_threshold)

    def benchmark(self, runs: int = 50, warmup: int = 5) -> None:
        dummy = np.random.rand(1, 3, self.image_size[1], self.image_size[0]).astype(np.float32)
        for _ in range(warmup):
            self.session.run(None, {self.input_name: dummy})

        timings = []
        for _ in range(runs):
            start = time.perf_counter()
            self.session.run(None, {self.input_name: dummy})
            timings.append((time.perf_counter() - start) * 1000)

        timings = np.array(timings)
        print(
            f"latency  mean {timings.mean():7.1f} ms | median {np.median(timings):7.1f} ms | "
            f"p90 {np.percentile(timings, 90):7.1f} ms  ->  {1000 / timings.mean():.2f} FPS"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", default=None, help="image path, directory, or camera index")
    parser.add_argument("--classes", nargs="*", default=None, help="class names, in training order")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--save", type=Path, default=None, help="write an annotated image here")
    args = parser.parse_args()

    detector = Detector(args.model, args.threads)

    if args.benchmark:
        detector.benchmark()
        if args.source is None:
            return

    if args.source is None:
        parser.error("--source is required unless --benchmark is used alone")

    image = Image.open(args.source).convert("RGB")
    start = time.perf_counter()
    detections = detector(image, args.conf, args.iou)
    elapsed = (time.perf_counter() - start) * 1000

    print(f"{len(detections)} detections in {elapsed:.1f} ms")
    for x1, y1, x2, y2, conf, cls in detections:
        name = args.classes[int(cls)] if args.classes and int(cls) < len(args.classes) else f"class_{int(cls)}"
        print(f"  {name:<12} {conf:.3f}  [{x1:7.1f} {y1:7.1f} {x2:7.1f} {y2:7.1f}]")

    if args.save:
        draw = ImageDraw.Draw(image)
        for x1, y1, x2, y2, conf, cls in detections:
            name = args.classes[int(cls)] if args.classes and int(cls) < len(args.classes) else f"class_{int(cls)}"
            draw.rectangle([x1, y1, x2, y2], outline=(255, 80, 80), width=3)
            draw.text((x1 + 2, max(0, y1 - 12)), f"{name} {conf:.2f}", fill=(255, 255, 0))
        image.save(args.save)
        print(f"saved {args.save}")


if __name__ == "__main__":
    main()
