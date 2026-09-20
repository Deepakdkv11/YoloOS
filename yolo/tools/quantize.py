"""Post-training INT8 quantization for Raspberry Pi 4 deployment.

The Pi 4 is a Cortex-A72: 64-bit ARMv8, NEON SIMD, no NPU, no VNNI-style dot-product
instructions (those arrive with ARMv8.2 dot-product on the Pi 5 / A76). Practical
consequences, and why this module does what it does:

  * INT8 still wins, but mostly through memory bandwidth and cache pressure, not raw
    integer throughput. Expect roughly 1.5-2.5x over FP32, not the 4x you might read
    about for server CPUs. A Pi 4 also throttles hard without a heatsink.
  * **Static** (calibrated) quantization is what you want. Dynamic quantization only
    quantizes weights and leaves activations in float, which for a conv-dominated
    detector buys you size but very little speed.
  * Per-channel weight scales are essential. YOLO's depthwise-ish grouped convs have
    wildly different per-channel ranges, and per-tensor scales collapse mAP.
  * QUInt8 activations / QInt8 weights is the combination ONNX Runtime's ARM kernels are
    actually optimised for; QInt8 activations fall back to slower paths.

Calibration data must be *real* images from the deployment camera. 100-500 frames that
cover your lighting and turbidity range is plenty; more than ~1000 adds nothing.
"""

from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np
from PIL import Image

from yolo.utils.logger import logger

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def letterbox(image: Image.Image, size: tuple, color: tuple = (114, 114, 114)) -> np.ndarray:
    """Resize preserving aspect ratio, pad to `size`. Must match training preprocessing.

    This mirrors `PadAndResize` in the training pipeline. Any mismatch here (different
    padding colour, centring, or interpolation) shifts the input distribution and shows
    up as a mysterious accuracy drop after export.
    """
    target_w, target_h = size
    img_w, img_h = image.size
    scale = min(target_w / img_w, target_h / img_h)
    new_w, new_h = int(img_w * scale), int(img_h * scale)

    resized = image.convert("RGB").resize((new_w, new_h), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (target_w, target_h), color)
    canvas.paste(resized, ((target_w - new_w) // 2, (target_h - new_h) // 2))
    return np.asarray(canvas)


def preprocess(image_path: Path, size: tuple) -> np.ndarray:
    """Image file -> NCHW float32 in [0, 1], the exact tensor the graph expects."""
    with Image.open(image_path) as image:
        array = letterbox(image, size)
    array = array.astype(np.float32) / 255.0
    return np.transpose(array, (2, 0, 1))[None]  # HWC -> NCHW


def _iter_images(directory: Path, limit: int) -> Iterator[Path]:
    count = 0
    for path in sorted(directory.rglob("*")):
        if path.suffix.lower() in IMAGE_SUFFIXES:
            yield path
            count += 1
            if count >= limit:
                return


class ImageCalibrationReader:
    """Feeds calibration batches to ONNX Runtime's static quantizer."""

    def __init__(self, image_dir: Path, image_size: tuple, limit: int = 300, input_name: str = "images"):
        self.image_paths: List[Path] = list(_iter_images(Path(image_dir), limit))
        if not self.image_paths:
            raise FileNotFoundError(f"No images found under {image_dir}")
        self.image_size = image_size
        self.input_name = input_name
        self._iterator = iter(self.image_paths)
        logger.info(f":microscope: Calibrating on {len(self.image_paths)} images from {image_dir}")

    def get_next(self):
        path = next(self._iterator, None)
        if path is None:
            return None
        return {self.input_name: preprocess(path, self.image_size)}

    def rewind(self):
        self._iterator = iter(self.image_paths)


def quantize_static_int8(
    onnx_path: Path,
    output_path: Path,
    calibration_dir: Path,
    image_size: tuple,
    num_calibration_images: int = 300,
    per_channel: bool = True,
) -> Path:
    """Calibrated INT8 quantization. This is the one to use for the Pi."""
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    onnx_path, output_path = Path(onnx_path), Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Shape inference + graph cleanup first. Skipping this is the usual cause of
    # "node X not quantized" warnings and a model that ends up mostly float.
    preprocessed = output_path.with_suffix(".preproc.onnx")
    quant_pre_process(str(onnx_path), str(preprocessed), skip_symbolic_shape=False)

    # ImageCalibrationReader must come first: CalibrationDataReader declares get_next
    # abstract, so listing it first would leave the abstract method unimplemented.
    class _Reader(ImageCalibrationReader, CalibrationDataReader):
        pass

    reader = _Reader(calibration_dir, image_size, num_calibration_images)

    quantize_static(
        model_input=str(preprocessed),
        model_output=str(output_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        per_channel=per_channel,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        # Sigmoid/Concat/Resize quantize poorly and sit at the very end of the graph where
        # they cost almost nothing; leaving them float protects confidence calibration.
        nodes_to_exclude=[],
        extra_options={"ActivationSymmetric": False, "WeightSymmetric": True},
    )
    preprocessed.unlink(missing_ok=True)

    original_mb = onnx_path.stat().st_size / 1e6
    quantized_mb = output_path.stat().st_size / 1e6
    logger.info(
        f":compression: INT8 model saved to {output_path} "
        f"({original_mb:.1f} MB -> {quantized_mb:.1f} MB, {original_mb / quantized_mb:.1f}x smaller)"
    )
    return output_path


def quantize_dynamic_int8(onnx_path: Path, output_path: Path) -> Path:
    """Weights-only INT8. No calibration images needed, but much less speedup.

    Use this only as a quick size check or when you genuinely cannot collect calibration
    frames; for conv-heavy detectors the static path above is worth the extra effort.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(str(onnx_path), str(output_path), weight_type=QuantType.QInt8)
    logger.info(f":compression: Dynamic INT8 model saved to {output_path}")
    return output_path
