"""Edge-deployment export for this YOLO implementation.

The stock `FastModelLoader._create_onnx_model` has three properties that make its output
a poor fit for a Raspberry Pi:

  1. It exports the *training* graph, auxiliary branch included. The aux branch exists
     only to supply gradients (YOLOv9's "programmable gradient information") and is dead
     weight at inference; it is roughly a third of the FLOPs for nothing.
  2. It emits the six raw head tensors and leaves anchor decoding to Python, so every
     consumer has to reimplement Vec2Box.
  3. It marks the batch axis dynamic, which blocks most INT8 kernels and NCNN's
     shape inference.

`export_onnx` below fixes all three: deploy-mode graph, decoding folded in as ONNX ops,
fully static shapes. The result is a single-input, two-output model:

    input   images  float32 [1, 3, H, W]   RGB, already letterboxed, scaled to 0..1
    output  boxes   float32 [1, N, 4]      xyxy, in pixels of the letterboxed input
    output  scores  float32 [1, N, C]      per-class confidence, sigmoid applied

NMS is intentionally left out of the graph: ONNX Runtime's NMS op is slow on ARM and
`deploy/rpi_infer.py` does it in numpy on the handful of boxes that survive thresholding.
"""

from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn

from yolo.model.yolo import YOLO, create_model
from yolo.utils.bounding_box_utils import generate_anchors
from yolo.utils.logger import logger


def load_state_from_checkpoint(checkpoint_path: Path, prefer_ema: bool = True) -> Dict[str, Tensor]:
    """Pull a plain YOLO state dict out of a Lightning training checkpoint.

    A checkpoint written by `task=train` holds *two* copies of the network:
      - `model.model.*` : the raw SGD weights
      - `ema.model.*`   : the exponential moving average

    The EMA copy is the one that gets validated and is almost always the better model,
    but `YOLO.save_load_weights` only ever reads the `model.model.` prefix, so the EMA
    weights are silently discarded. Default to EMA here.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "state_dict" not in checkpoint:
        return checkpoint  # already a bare state dict (e.g. the released .pt files)

    state_dict = checkpoint["state_dict"]
    ema_keys = [key for key in state_dict if key.startswith("ema.model.")]

    if prefer_ema and ema_keys:
        logger.info(":sparkles: Using EMA weights from checkpoint")
        return {key.removeprefix("ema.model."): state_dict[key] for key in ema_keys}

    if prefer_ema:
        logger.warning(":warning: No EMA weights found in checkpoint; falling back to raw weights")
    return {
        key.removeprefix("model.model."): value
        for key, value in state_dict.items()
        if key.startswith("model.model.")
    }


class DeployYOLO(nn.Module):
    """YOLO + anchor decoding, as one traceable module with no Python-side post-processing.

    Reimplements `Vec2Box.__call__` with `einops.rearrange` replaced by plain reshape /
    permute so the graph exports cleanly, and with the anchor grid baked in as a buffer
    (which is what makes the static-shape requirement acceptable).
    """

    def __init__(self, model: YOLO, image_size: Tuple[int, int], strides: List[int], num_classes: int):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        anchor_grid, scaler = generate_anchors(list(image_size), strides)
        self.register_buffer("anchor_grid", anchor_grid, persistent=False)
        self.register_buffer("scaler", scaler, persistent=False)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        outputs = self.model(x)
        predictions = outputs["Main"]

        classes, boxes = [], []
        for layer_output in predictions:
            pred_cls, _pred_anc, pred_box = layer_output
            batch, num_cls, height, width = pred_cls.shape
            # B C h w -> B (h w) C
            classes.append(pred_cls.permute(0, 2, 3, 1).reshape(batch, height * width, num_cls))
            # B X h w -> B (h w) X
            boxes.append(pred_box.permute(0, 2, 3, 1).reshape(batch, height * width, pred_box.shape[1]))

        pred_cls = torch.cat(classes, dim=1)
        pred_box = torch.cat(boxes, dim=1)

        # distance-to-anchor (in stride units) -> absolute xyxy in input pixels
        pred_ltrb = pred_box * self.scaler.view(1, -1, 1)
        lt, rb = pred_ltrb.chunk(2, dim=-1)
        boxes_xyxy = torch.cat([self.anchor_grid - lt, self.anchor_grid + rb], dim=-1)

        return boxes_xyxy, pred_cls.sigmoid()


def build_deploy_model(
    cfg,
    weight_path: Optional[Path] = None,
    prefer_ema: bool = True,
) -> DeployYOLO:
    """Construct the auxiliary-free inference model and load weights into it."""
    model_cfg = deepcopy(cfg.model)
    # Drop the auxiliary branch. In this codebase the architecture is a dict of named
    # blocks, and "auxiliary" is one of them, so emptying it removes those layers entirely.
    if "auxiliary" in model_cfg.model:
        model_cfg.model["auxiliary"] = {}
        logger.info(":scissors: Stripped auxiliary branch for deployment")

    model = create_model(model_cfg, weight_path=False, class_num=cfg.dataset.class_num)

    if weight_path is not None:
        state_dict = load_state_from_checkpoint(Path(weight_path), prefer_ema=prefer_ema)
        missing, unexpected = model.model.load_state_dict(state_dict, strict=False)
        # Unexpected keys are expected here: they are the auxiliary layers we just removed.
        if missing:
            logger.warning(f":warning: {len(missing)} weights missing from checkpoint, e.g. {missing[:3]}")
        logger.info(f":white_check_mark: Loaded weights ({len(unexpected)} auxiliary tensors ignored)")

    model.eval()

    image_size = tuple(cfg.image_size)
    strides = _infer_strides(model, image_size)
    logger.info(f":straight_ruler: Detection strides: {strides}")

    return DeployYOLO(model, image_size, strides, cfg.dataset.class_num).eval()


@torch.no_grad()
def _infer_strides(model: YOLO, image_size: Tuple[int, int]) -> List[int]:
    width, height = image_size
    dummy = torch.zeros(1, 3, height, width)
    output = model(dummy)
    strides = []
    for head in output["Main"]:
        *_, grid_h, grid_w = head[2].shape
        strides.append(width // grid_w)
    return strides


@torch.no_grad()
def export_onnx(
    cfg,
    weight_path: Optional[Path],
    output_path: Path,
    prefer_ema: bool = True,
    opset: int = 13,
    simplify: bool = True,
) -> Path:
    """Export a static-shape, decode-fused, auxiliary-free ONNX graph."""
    model = build_deploy_model(cfg, weight_path, prefer_ema=prefer_ema)

    width, height = cfg.image_size
    dummy = torch.zeros(1, 3, height, width)

    boxes, scores = model(dummy)
    logger.info(f":package: Graph outputs: boxes {tuple(boxes.shape)}, scores {tuple(scores.shape)}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["images"],
        output_names=["boxes", "scores"],
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes=None,  # static shapes: required for good INT8 / NCNN performance
    )
    logger.info(f":inbox_tray: ONNX saved to {output_path}")

    if simplify:
        _try_simplify(output_path)

    return output_path


def _try_simplify(onnx_path: Path) -> None:
    """Run onnx-simplifier if present. Optional, but it folds a lot of reshape noise."""
    try:
        import onnx
        import onnxsim
    except ImportError:
        logger.info(":information: onnxsim not installed; skipping graph simplification "
                    "(`pip install onnxsim` for a smaller, faster graph)")
        return

    model = onnx.load(str(onnx_path))
    simplified, ok = onnxsim.simplify(model)
    if ok:
        onnx.save(simplified, str(onnx_path))
        logger.info(":broom: Simplified ONNX graph")
    else:
        logger.warning(":warning: onnxsim could not validate the simplified graph; keeping the original")
