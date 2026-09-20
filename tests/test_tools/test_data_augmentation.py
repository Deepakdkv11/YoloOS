import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as TF

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from yolo.tools.data_augmentation import (
    AugmentationComposer,
    HorizontalFlip,
    HSVJitter,
    Mosaic,
    RandomAffine,
    VerticalFlip,
)


def test_horizontal_flip():
    # Create a mock image and bounding boxes
    img = Image.new("RGB", (100, 100), color="red")
    boxes = torch.tensor([[1, 0.05, 0.1, 0.7, 0.9]])  # class, xmin, ymin, xmax, ymax

    flip_transform = HorizontalFlip(prob=1)  # Set probability to 1 to ensure flip
    flipped_img, flipped_boxes = flip_transform(img, boxes)

    # Assert image is flipped by comparing it to a manually flipped image
    assert TF.hflip(img) == flipped_img

    # Assert bounding boxes are flipped correctly
    expected_boxes = torch.tensor([[1, 0.3, 0.1, 0.95, 0.9]])
    assert torch.allclose(flipped_boxes, expected_boxes), "Bounding boxes were not flipped correctly"


def test_compose():
    # Define two mock transforms that simply return the inputs
    def mock_transform(image, boxes):
        return image, boxes

    compose = AugmentationComposer([mock_transform, mock_transform])
    img = Image.new("RGB", (640, 640), color="blue")
    boxes = torch.tensor([[0, 0.2, 0.2, 0.8, 0.8]])

    transformed_img, transformed_boxes, rev_tensor = compose(img, boxes)
    tensor_img = TF.pil_to_tensor(img).to(torch.float32) / 255

    assert (transformed_img == tensor_img).all(), "Image should not be altered"
    assert torch.equal(transformed_boxes, boxes), "Boxes should not be altered"


def test_mosaic():
    img = Image.new("RGB", (100, 100), color="green")
    boxes = torch.tensor([[0, 0.25, 0.25, 0.75, 0.75]])

    # Mock parent with image_size and get_more_data method
    class MockParent:
        base_size = 100

        def get_more_data(self, num_images):
            return [(img, boxes) for _ in range(num_images)]

    mosaic = Mosaic(prob=1)  # Ensure mosaic is applied
    mosaic.set_parent(MockParent())

    parent = MockParent()
    mosaic.set_parent(parent)
    mosaic_img, mosaic_boxes = mosaic(img, boxes)

    # Mosaic emits the full 2x canvas (as in YOLOv5/v8); a following RandomAffine crops
    # it back to base_size. Downscaling here instead would halve every object's resolution.
    assert mosaic_img.size == (200, 200), "Mosaic should emit a 2x canvas"
    assert parent.mosaic_border == (-50, -50), "Mosaic must flag the border for RandomAffine"
    assert len(mosaic_boxes) > 0, "Should have some bounding boxes"
    assert (mosaic_boxes[:, 1:] >= 0).all() and (mosaic_boxes[:, 1:] <= 1).all(), "Boxes must stay normalized"


def test_mosaic_then_affine_restores_base_size():
    """The Mosaic -> RandomAffine pair must end up back at base_size."""
    img = Image.new("RGB", (100, 100), color="green")
    boxes = torch.tensor([[0, 0.25, 0.25, 0.75, 0.75]])

    class MockParent:
        base_size = 100
        mosaic_border = (0, 0)

        def get_more_data(self, num_images):
            return [(img, boxes) for _ in range(num_images)]

    parent = MockParent()
    mosaic, affine = Mosaic(prob=1), RandomAffine(prob=1, degrees=0, translate=0.1, scale=0.5)
    mosaic.set_parent(parent)
    affine.set_parent(parent)

    mosaic_img, mosaic_boxes = mosaic(img, boxes)
    out_img, out_boxes = affine(mosaic_img, mosaic_boxes)

    assert out_img.size == (100, 100), "RandomAffine should crop the mosaic back to base_size"
    assert parent.mosaic_border == (0, 0), "Border must be reset after the crop is consumed"
    if len(out_boxes):
        assert (out_boxes[:, 3] > out_boxes[:, 1]).all(), "No degenerate boxes"
        assert (out_boxes[:, 4] > out_boxes[:, 2]).all(), "No degenerate boxes"


def test_hsv_jitter_preserves_geometry():
    """HSV must change pixels but never touch boxes or image size."""
    img = Image.new("RGB", (64, 48), color=(120, 90, 60))
    boxes = torch.tensor([[0, 0.2, 0.2, 0.8, 0.8]])

    jitter = HSVJitter(prob=1, hgain=0.015, sgain=0.7, vgain=0.4)
    out_img, out_boxes = jitter(img, boxes.clone())

    assert out_img.size == img.size, "HSV must not resize"
    assert torch.equal(out_boxes, boxes), "HSV must not alter boxes"
