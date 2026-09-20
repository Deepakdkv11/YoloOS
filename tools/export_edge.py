"""One-shot edge export: checkpoint -> deploy ONNX -> INT8 ONNX, with an accuracy check.

Usage
-----
    python tools/export_edge.py \
        --checkpoint runs/train/myrun/checkpoints/last.ckpt \
        --model v9-s \
        --classes 3 \
        --image-size 416 416 \
        --calib-dir data/custom/images/train \
        --out-dir deploy_out

Outputs `<out-dir>/model_fp32.onnx` and `<out-dir>/model_int8.onnx`, then reports how far
the INT8 outputs drift from FP32 on a sample of the calibration images. Treat a mean box
drift above ~2 px or a score correlation below ~0.99 as a red flag: re-run with more
calibration images, or fall back to FP32.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from yolo.tools.export import export_onnx  # noqa: E402
from yolo.tools.quantize import preprocess, quantize_static_int8  # noqa: E402
from yolo.utils.logger import logger  # noqa: E402


def build_cfg(model_name: str, num_classes: int, image_size) -> OmegaConf:
    model_cfg = OmegaConf.load(REPO_ROOT / "yolo" / "config" / "model" / f"{model_name}.yaml")
    return OmegaConf.create(
        {
            "model": model_cfg,
            "dataset": {"class_num": num_classes},
            "image_size": list(image_size),
        }
    )


def compare_models(fp32_path: Path, int8_path: Path, calib_dir: Path, image_size, num_samples: int = 20) -> None:
    """Sanity-check the quantized graph against the float one on real images."""
    import onnxruntime as ort

    suffixes = {".jpg", ".jpeg", ".png", ".bmp"}
    images = [p for p in sorted(Path(calib_dir).rglob("*")) if p.suffix.lower() in suffixes][:num_samples]
    if not images:
        logger.warning("No images available for the INT8 accuracy check; skipping")
        return

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    fp32 = ort.InferenceSession(str(fp32_path), options, providers=["CPUExecutionProvider"])
    int8 = ort.InferenceSession(str(int8_path), options, providers=["CPUExecutionProvider"])

    box_deltas, score_corrs, score_absdiffs = [], [], []
    for image_path in images:
        tensor = preprocess(image_path, tuple(image_size))
        boxes_f, scores_f = fp32.run(None, {"images": tensor})
        boxes_q, scores_q = int8.run(None, {"images": tensor})

        # Only compare where the float model actually predicts something; the background
        # anchors dominate by count and would wash out any real signal.
        confident = scores_f.max(axis=-1)[0] > 0.25
        if confident.sum() > 0:
            box_deltas.append(np.abs(boxes_f[0][confident] - boxes_q[0][confident]).mean())

        # corrcoef is undefined when either side has zero variance (an untrained model
        # emits near-constant scores). Fall back to max absolute difference so the check
        # still reports something meaningful instead of nan.
        flat_f, flat_q = scores_f.ravel(), scores_q.ravel()
        if flat_f.std() > 1e-9 and flat_q.std() > 1e-9:
            score_corrs.append(np.corrcoef(flat_f, flat_q)[0, 1])
        score_absdiffs.append(np.abs(flat_f - flat_q).max())

    logger.info("=" * 62)
    logger.info(f"INT8 vs FP32 over {len(images)} images")
    if box_deltas:
        logger.info(f"  mean box drift   : {np.mean(box_deltas):.3f} px   (want < 2.0)")
    else:
        logger.info("  mean box drift   : n/a (no confident detections)")
    if score_corrs:
        logger.info(f"  score correlation: {np.mean(score_corrs):.5f}      (want > 0.99)")
    else:
        logger.info("  score correlation: n/a (scores have no variance - is the model trained?)")
    logger.info(f"  max score delta  : {np.max(score_absdiffs):.5f}")
    logger.info("=" * 62)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Lightning .ckpt or released .pt")
    parser.add_argument("--model", default="v9-s", help="model config name, e.g. v9-t / v9-s / v9-m / v9-c")
    parser.add_argument("--classes", type=int, required=True, help="number of classes in your dataset")
    parser.add_argument("--image-size", type=int, nargs=2, default=[416, 416], metavar=("W", "H"))
    parser.add_argument("--calib-dir", type=Path, required=True, help="directory of representative images")
    parser.add_argument("--calib-images", type=int, default=300)
    parser.add_argument("--out-dir", type=Path, default=Path("deploy_out"))
    parser.add_argument("--no-ema", action="store_true", help="export raw weights instead of the EMA copy")
    parser.add_argument("--skip-int8", action="store_true")
    args = parser.parse_args()

    cfg = build_cfg(args.model, args.classes, args.image_size)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    fp32_path = args.out_dir / "model_fp32.onnx"
    export_onnx(cfg, args.checkpoint, fp32_path, prefer_ema=not args.no_ema)

    if args.skip_int8:
        return

    int8_path = args.out_dir / "model_int8.onnx"
    quantize_static_int8(
        fp32_path,
        int8_path,
        args.calib_dir,
        tuple(args.image_size),
        num_calibration_images=args.calib_images,
    )
    compare_models(fp32_path, int8_path, args.calib_dir, args.image_size)


if __name__ == "__main__":
    main()
