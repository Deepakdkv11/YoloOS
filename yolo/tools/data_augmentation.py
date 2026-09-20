import math
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

# cv2 is already a hard dependency (requirements.txt: opencv-python) and is used
# for every pixel-level augmentation below because PIL has no fast warp/LUT path.
import cv2

from yolo.utils.logger import logger

__all__ = [
    "AugmentationComposer",
    "PadAndResize",
    "RemoveOutliers",
    "HorizontalFlip",
    "VerticalFlip",
    "Mosaic",
    "MixUp",
    "RandomCrop",
    "HSVJitter",
    "RandomAffine",
]


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _to_numpy(image: Image.Image) -> np.ndarray:
    """PIL RGB -> uint8 HWC numpy (RGB order)."""
    if not isinstance(image, Image.Image):
        return image
    return np.asarray(image.convert("RGB"))


def _to_pil(array: np.ndarray) -> Image.Image:
    return Image.fromarray(array)


def box_candidates(
    box_before: np.ndarray,
    box_after: np.ndarray,
    wh_thr: float = 2.0,
    ar_thr: float = 100.0,
    area_thr: float = 0.1,
    eps: float = 1e-16,
) -> np.ndarray:
    """Keep-mask for boxes that survived a geometric warp.

    Mirrors the filter used by YOLOv5/v8 (`box_candidates`). Without it, boxes that are
    warped almost entirely out of frame collapse into slivers that are still treated as
    valid positives, which poisons the assigner and the box regression loss.

    Args:
        box_before: (4, N) xyxy in pixels, before the warp.
        box_after: (4, N) xyxy in pixels, after the warp and clipping.
        wh_thr: minimum width/height in pixels.
        ar_thr: maximum aspect ratio.
        area_thr: minimum surviving area, as a fraction of the original area.
    """
    w1, h1 = box_before[2] - box_before[0], box_before[3] - box_before[1]
    w2, h2 = box_after[2] - box_after[0], box_after[3] - box_after[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)


class AugmentationComposer:
    """Composes several transforms together."""

    def __init__(self, transforms, image_size: int = [640, 640], base_size: int = 640):
        self.transforms = transforms
        # TODO: handle List of image_size [640, 640]
        self.pad_resize = PadAndResize(image_size)
        self.base_size = base_size
        # Set by Mosaic so that a following RandomAffine knows to crop the 2x canvas
        # back down to base_size instead of letting PadAndResize shrink it.
        self.mosaic_border: Tuple[int, int] = (0, 0)

        for transform in self.transforms:
            if hasattr(transform, "set_parent"):
                transform.set_parent(self)

        self._warn_mosaic_without_affine()

    def _warn_mosaic_without_affine(self) -> None:
        names = [type(t).__name__ for t in self.transforms]
        if "Mosaic" in names and "RandomAffine" not in names:
            logger.warning(
                ":warning: Mosaic is enabled without RandomAffine. Mosaic builds a 2x canvas that "
                "RandomAffine is expected to crop back; without it the canvas is downscaled to "
                "image_size and every object ends up at half resolution. Add `RandomAffine` after "
                "`Mosaic` in task.data.data_augment."
            )

    def __call__(self, image, boxes=torch.zeros(0, 5)):
        self.mosaic_border = (0, 0)
        for transform in self.transforms:
            image, boxes = transform(image, boxes)
        image, boxes, rev_tensor = self.pad_resize(image, boxes)
        image = TF.to_tensor(image)
        return image, boxes, rev_tensor


class RemoveOutliers:
    """Removes outlier bounding boxes that are too small or have invalid dimensions."""

    def __init__(self, min_box_area=1e-8):
        """
        Args:
            min_box_area (float): Minimum area for a box to be kept, as a fraction of the image area.
        """
        self.min_box_area = min_box_area

    def __call__(self, image, boxes):
        """
        Args:
            image (PIL.Image): The cropped image.
            boxes (torch.Tensor): Bounding boxes in normalized coordinates (x_min, y_min, x_max, y_max).
        Returns:
            PIL.Image: The input image (unchanged).
            torch.Tensor: Filtered bounding boxes.
        """
        box_areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 4] - boxes[:, 2])

        valid_boxes = (box_areas > self.min_box_area) & (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 4] > boxes[:, 2])

        return image, boxes[valid_boxes]


class PadAndResize:
    def __init__(self, image_size, background_color=(114, 114, 114)):
        """Initialize the object with the target image size."""
        self.target_width, self.target_height = image_size
        self.background_color = background_color

    def set_size(self, image_size: List[int]):
        self.target_width, self.target_height = image_size

    def __call__(self, image: Image, boxes):
        img_width, img_height = image.size
        scale = min(self.target_width / img_width, self.target_height / img_height)
        new_width, new_height = int(img_width * scale), int(img_height * scale)

        # BILINEAR rather than LANCZOS: letterboxing runs once per sample per epoch and
        # LANCZOS is ~5x slower for no measurable mAP benefit (v5/v8 use INTER_LINEAR).
        resized_image = image.resize((new_width, new_height), Image.Resampling.BILINEAR)

        pad_left = (self.target_width - new_width) // 2
        pad_top = (self.target_height - new_height) // 2
        padded_image = Image.new("RGB", (self.target_width, self.target_height), self.background_color)
        padded_image.paste(resized_image, (pad_left, pad_top))

        boxes[:, [1, 3]] = (boxes[:, [1, 3]] * new_width + pad_left) / self.target_width
        boxes[:, [2, 4]] = (boxes[:, [2, 4]] * new_height + pad_top) / self.target_height

        transform_info = torch.tensor([scale, pad_left, pad_top, pad_left, pad_top])
        return padded_image, boxes, transform_info


class HorizontalFlip:
    """Randomly horizontally flips the image along with the bounding boxes."""

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, boxes):
        if torch.rand(1) < self.prob:
            image = TF.hflip(image)
            boxes[:, [1, 3]] = 1 - boxes[:, [3, 1]]
        return image, boxes


class VerticalFlip:
    """Randomly vertically flips the image along with the bounding boxes."""

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, boxes):
        if torch.rand(1) < self.prob:
            image = TF.vflip(image)
            boxes[:, [2, 4]] = 1 - boxes[:, [4, 2]]
        return image, boxes


class HSVJitter:
    """Random HSV gain, matching the YOLOv5/v8 `augment_hsv` defaults.

    This is the single cheapest augmentation for photometric robustness (lighting,
    white balance, water turbidity/colour shifts) and was entirely absent here.
    """

    def __init__(self, prob: float = 1.0, hgain: float = 0.015, sgain: float = 0.7, vgain: float = 0.4):
        self.prob = prob
        self.hgain, self.sgain, self.vgain = hgain, sgain, vgain

    def __call__(self, image, boxes):
        if torch.rand(1) >= self.prob or not (self.hgain or self.sgain or self.vgain):
            return image, boxes

        array = _to_numpy(image)
        gains = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain] + 1

        hue, sat, val = cv2.split(cv2.cvtColor(array, cv2.COLOR_RGB2HSV))
        lut_base = np.arange(0, 256, dtype=gains.dtype)
        lut_hue = ((lut_base * gains[0]) % 180).astype(np.uint8)
        lut_sat = np.clip(lut_base * gains[1], 0, 255).astype(np.uint8)
        lut_val = np.clip(lut_base * gains[2], 0, 255).astype(np.uint8)

        merged = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        return _to_pil(cv2.cvtColor(merged, cv2.COLOR_HSV2RGB)), boxes


class RandomAffine:
    """Random perspective / rotation / scale / shear / translation.

    Port of YOLOv5/v8 `random_perspective`. Two jobs:
      1. Geometric augmentation (scale jitter is the strongest single knob in the v8 recipe).
      2. Cropping a Mosaic 2x canvas back down to `base_size`, which is how v5/v8 mosaic
         actually works. Run this immediately after Mosaic.
    """

    def __init__(
        self,
        prob: float = 1.0,
        degrees: float = 0.0,
        translate: float = 0.1,
        scale: float = 0.5,
        shear: float = 0.0,
        perspective: float = 0.0,
        border: Optional[Tuple[int, int]] = None,
        background_color: Tuple[int, int, int] = (114, 114, 114),
    ):
        self.prob = prob
        self.degrees, self.translate = degrees, translate
        self.scale, self.shear, self.perspective = scale, shear, perspective
        self.border = tuple(border) if border is not None else None
        self.background_color = background_color
        self.parent = None

    def set_parent(self, parent):
        self.parent = parent

    def _resolve_border(self, height: int, width: int) -> Tuple[int, int]:
        if self.border is not None:
            return self.border
        # Mosaic sets parent.mosaic_border; honour it so the 2x canvas is cropped, not squashed.
        parent_border = getattr(self.parent, "mosaic_border", (0, 0))
        if parent_border != (0, 0):
            return parent_border
        return (0, 0)

    def __call__(self, image, boxes):
        if torch.rand(1) >= self.prob:
            return image, boxes

        array = _to_numpy(image)
        src_h, src_w = array.shape[:2]
        border = self._resolve_border(src_h, src_w)
        height = src_h + border[0] * 2
        width = src_w + border[1] * 2
        if height <= 0 or width <= 0:
            return image, boxes

        # Center
        C = np.eye(3)
        C[0, 2] = -src_w / 2
        C[1, 2] = -src_h / 2

        # Perspective
        P = np.eye(3)
        P[2, 0] = np.random.uniform(-self.perspective, self.perspective)
        P[2, 1] = np.random.uniform(-self.perspective, self.perspective)

        # Rotation and scale
        R = np.eye(3)
        angle = np.random.uniform(-self.degrees, self.degrees)
        gain = np.random.uniform(1 - self.scale, 1 + self.scale)
        R[:2] = cv2.getRotationMatrix2D(angle=angle, center=(0, 0), scale=gain)

        # Shear
        S = np.eye(3)
        S[0, 1] = math.tan(np.random.uniform(-self.shear, self.shear) * math.pi / 180)
        S[1, 0] = math.tan(np.random.uniform(-self.shear, self.shear) * math.pi / 180)

        # Translation
        T = np.eye(3)
        T[0, 2] = np.random.uniform(0.5 - self.translate, 0.5 + self.translate) * width
        T[1, 2] = np.random.uniform(0.5 - self.translate, 0.5 + self.translate) * height

        M = T @ S @ R @ P @ C
        if border != (0, 0) or (M != np.eye(3)).any():
            if self.perspective:
                array = cv2.warpPerspective(array, M, dsize=(width, height), borderValue=self.background_color)
            else:
                array = cv2.warpAffine(array, M[:2], dsize=(width, height), borderValue=self.background_color)

        num_boxes = boxes.shape[0]
        if num_boxes == 0:
            return _to_pil(array), boxes

        # normalized -> source pixels
        pts = boxes[:, 1:5].clone().float().numpy()
        pts[:, [0, 2]] *= src_w
        pts[:, [1, 3]] *= src_h
        before = pts.T.copy()

        # warp the 4 corners of each box, then take the axis-aligned hull
        corners = np.ones((num_boxes * 4, 3))
        corners[:, :2] = pts[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(num_boxes * 4, 2)
        corners = corners @ M.T
        if self.perspective:
            corners = corners[:, :2] / corners[:, 2:3]
        else:
            corners = corners[:, :2]
        corners = corners.reshape(num_boxes, 8)

        x_coords = corners[:, [0, 2, 4, 6]]
        y_coords = corners[:, [1, 3, 5, 7]]
        warped = np.concatenate(
            (x_coords.min(1), y_coords.min(1), x_coords.max(1), y_coords.max(1))
        ).reshape(4, num_boxes)

        warped[[0, 2]] = warped[[0, 2]].clip(0, width)
        warped[[1, 3]] = warped[[1, 3]].clip(0, height)

        keep = box_candidates(before, warped, area_thr=0.10)
        warped = warped[:, keep]

        new_boxes = torch.zeros((int(keep.sum()), 5), dtype=boxes.dtype)
        new_boxes[:, 0] = boxes[torch.from_numpy(keep), 0]
        new_boxes[:, 1] = torch.from_numpy(warped[0] / width).to(boxes.dtype)
        new_boxes[:, 2] = torch.from_numpy(warped[1] / height).to(boxes.dtype)
        new_boxes[:, 3] = torch.from_numpy(warped[2] / width).to(boxes.dtype)
        new_boxes[:, 4] = torch.from_numpy(warped[3] / height).to(boxes.dtype)

        # the 2x mosaic canvas has been consumed
        if self.parent is not None:
            self.parent.mosaic_border = (0, 0)

        return _to_pil(array), new_boxes


class Mosaic:
    """4-image mosaic, rebuilt to match YOLOv5/v8 semantics.

    The previous implementation pasted the 4 images at fixed offsets without resizing
    them to a common scale, never clipped boxes to the canvas, and kept boxes that fell
    entirely outside it. This version resizes each tile so its long side is `base_size`,
    uses a random mosaic center, clips boxes to the canvas and drops degenerate ones.

    Returns a 2*base_size canvas; a following `RandomAffine` crops it back down.
    """

    def __init__(self, prob=0.5):
        self.prob = prob
        self.parent = None

    def set_parent(self, parent):
        self.parent = parent

    def __call__(self, image, boxes):
        if torch.rand(1) >= self.prob:
            return image, boxes

        assert self.parent is not None, "Parent is not set. Mosaic cannot retrieve image size."

        size = int(self.parent.base_size)
        canvas_size = size * 2
        data = [(image, boxes)] + self.parent.get_more_data(3)

        canvas = np.full((canvas_size, canvas_size, 3), 114, dtype=np.uint8)
        center_x = int(torch.randint(size // 2, size + size // 2 + 1, (1,)).item())
        center_y = int(torch.randint(size // 2, size + size // 2 + 1, (1,)).item())

        all_labels = []
        for index, (tile, tile_boxes) in enumerate(data):
            array = _to_numpy(tile)
            src_h, src_w = array.shape[:2]
            ratio = size / max(src_h, src_w)
            if ratio != 1:
                array = cv2.resize(
                    array,
                    (max(1, int(round(src_w * ratio))), max(1, int(round(src_h * ratio)))),
                    interpolation=cv2.INTER_LINEAR,
                )
            tile_h, tile_w = array.shape[:2]

            if index == 0:  # top-left
                x1a, y1a, x2a, y2a = max(center_x - tile_w, 0), max(center_y - tile_h, 0), center_x, center_y
                x1b, y1b, x2b, y2b = tile_w - (x2a - x1a), tile_h - (y2a - y1a), tile_w, tile_h
            elif index == 1:  # top-right
                x1a, y1a, x2a, y2a = center_x, max(center_y - tile_h, 0), min(center_x + tile_w, canvas_size), center_y
                x1b, y1b, x2b, y2b = 0, tile_h - (y2a - y1a), min(tile_w, x2a - x1a), tile_h
            elif index == 2:  # bottom-left
                x1a, y1a, x2a, y2a = max(center_x - tile_w, 0), center_y, center_x, min(canvas_size, center_y + tile_h)
                x1b, y1b, x2b, y2b = tile_w - (x2a - x1a), 0, tile_w, min(y2a - y1a, tile_h)
            else:  # bottom-right
                x1a, y1a, x2a, y2a = center_x, center_y, min(center_x + tile_w, canvas_size), min(canvas_size, center_y + tile_h)
                x1b, y1b, x2b, y2b = 0, 0, min(tile_w, x2a - x1a), min(y2a - y1a, tile_h)

            canvas[y1a:y2a, x1a:x2a] = array[y1b:y2b, x1b:x2b]
            pad_w, pad_h = x1a - x1b, y1a - y1b

            if tile_boxes.numel():
                adjusted = tile_boxes.clone().float()
                adjusted[:, [1, 3]] = adjusted[:, [1, 3]] * tile_w + pad_w
                adjusted[:, [2, 4]] = adjusted[:, [2, 4]] * tile_h + pad_h
                all_labels.append(adjusted)

        if all_labels:
            labels = torch.cat(all_labels, dim=0)
            labels[:, [1, 3]] = labels[:, [1, 3]].clamp(0, canvas_size)
            labels[:, [2, 4]] = labels[:, [2, 4]].clamp(0, canvas_size)
            keep = (labels[:, 3] - labels[:, 1] > 2) & (labels[:, 4] - labels[:, 2] > 2)
            labels = labels[keep]
            labels[:, [1, 3]] /= canvas_size
            labels[:, [2, 4]] /= canvas_size
        else:
            labels = torch.zeros((0, 5), dtype=boxes.dtype)

        # Tell the following RandomAffine to crop the canvas back to base_size.
        self.parent.mosaic_border = (-size // 2, -size // 2)

        return _to_pil(canvas), labels


class MixUp:
    """Applies the MixUp augmentation to a pair of images and their corresponding boxes."""

    def __init__(self, prob=0.5, alpha=32.0, beta=32.0):
        # v8 uses Beta(32, 32), i.e. lambda concentrated near 0.5. The previous default of
        # alpha=1.0 is a uniform mix, which is far too destructive for detection.
        self.alpha = alpha
        self.beta = beta
        self.prob = prob
        self.parent = None

    def set_parent(self, parent):
        """Set the parent dataset object for accessing dataset methods."""
        self.parent = parent

    def __call__(self, image, boxes):
        if torch.rand(1) >= self.prob:
            return image, boxes

        assert self.parent is not None, "Parent is not set. MixUp cannot retrieve additional data."

        image2, boxes2 = self.parent.get_more_data()[0]

        # The two images are almost never the same size; without this resize the tensor
        # add below raises a shape error. Boxes are normalized, so they need no rescale.
        if image2.size != image.size:
            image2 = image2.resize(image.size, Image.Resampling.BILINEAR)

        lam = float(np.random.beta(self.alpha, self.beta)) if self.alpha > 0 else 0.5

        image1_t, image2_t = TF.to_tensor(image), TF.to_tensor(image2)
        mixed_image = lam * image1_t + (1 - lam) * image2_t

        merged_boxes = torch.cat((boxes, boxes2))

        return TF.to_pil_image(mixed_image), merged_boxes


class RandomCrop:
    """Randomly crops the image to half its size along with adjusting the bounding boxes.

    NOTE: this is not part of the YOLOv5/v8 recipe and is a very aggressive default
    (it always crops to exactly 50% and leaves boxes clipped flat against the new edge).
    `RandomAffine` covers the same ground far more gently. Kept for backwards
    compatibility; it now drops boxes that were cropped away instead of flattening them.
    """

    def __init__(self, prob=0.5):
        """
        Args:
            prob (float): Probability of applying the crop.
        """
        self.prob = prob

    def __call__(self, image, boxes):
        if torch.rand(1) < self.prob:
            original_width, original_height = image.size
            crop_height, crop_width = original_height // 2, original_width // 2
            top = torch.randint(0, original_height - crop_height + 1, (1,)).item()
            left = torch.randint(0, original_width - crop_width + 1, (1,)).item()

            image = TF.crop(image, top, left, crop_height, crop_width)

            before = torch.stack(
                [
                    boxes[:, 1] * original_width,
                    boxes[:, 2] * original_height,
                    boxes[:, 3] * original_width,
                    boxes[:, 4] * original_height,
                ]
            ).numpy()

            boxes[:, [1, 3]] = boxes[:, [1, 3]] * original_width - left
            boxes[:, [2, 4]] = boxes[:, [2, 4]] * original_height - top

            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, crop_width)
            boxes[:, [2, 4]] = boxes[:, [2, 4]].clamp(0, crop_height)

            if boxes.shape[0]:
                after = boxes[:, 1:5].T.numpy()
                keep = box_candidates(before, after, area_thr=0.10)
                boxes = boxes[torch.from_numpy(keep)]

            boxes[:, [1, 3]] /= crop_width
            boxes[:, [2, 4]] /= crop_height

        return image, boxes
