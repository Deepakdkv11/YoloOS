"""Regression tests for .txt label parsing.

The original reader assumed every .txt row was a normalized polygon. Feeding it the
standard YOLO detection row `<cls> <cx> <cy> <w> <h>` produced a silently wrong box
rather than an error, which is the single easiest way to waste a training run.
"""

import sys
from pathlib import Path

import pytest
import torch

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from yolo.tools.data_loader import YoloDataset
from yolo.utils.dataset_utils import detect_label_format


def _parse(rows, label_format):
    """Call the parser without constructing a full dataset."""
    return YoloDataset.load_valid_labels(None, "unit-test", rows, label_format)


def test_detect_format_gives_correct_box():
    # center (0.5, 0.5), size 0.2 x 0.4  ->  xyxy (0.4, 0.3, 0.6, 0.7)
    boxes = _parse([[0, 0.5, 0.5, 0.2, 0.4]], "detect")
    assert boxes.shape == (1, 5)
    assert torch.allclose(boxes[0], torch.tensor([0.0, 0.4, 0.3, 0.6, 0.7]), atol=1e-6)


def test_segment_format_gives_correct_box():
    polygon = [0, 0.4, 0.3, 0.6, 0.3, 0.6, 0.7, 0.4, 0.7]
    boxes = _parse([polygon], "segment")
    assert torch.allclose(boxes[0], torch.tensor([0.0, 0.4, 0.3, 0.6, 0.7]), atol=1e-6)


def test_detect_and_segment_disagree_on_the_same_row():
    """Guards the actual bug: the two readers must not be interchangeable."""
    row = [0, 0.5, 0.5, 0.2, 0.4]
    assert not torch.allclose(_parse([row], "detect")[0], _parse([row], "segment")[0])


def test_detect_format_clips_to_image():
    # A box hanging off the left/top edge must be clipped, not wrapped or dropped.
    boxes = _parse([[1, 0.05, 0.05, 0.4, 0.4]], "detect")
    assert boxes[0, 1] == 0.0 and boxes[0, 2] == 0.0
    assert boxes[0, 3] == pytest.approx(0.25) and boxes[0, 4] == pytest.approx(0.25)


def test_detect_format_rejects_malformed_row():
    assert _parse([[0, 0.5, 0.5, 0.2]], "detect").shape == (0, 5)


def test_detect_format_drops_zero_area():
    assert _parse([[0, 0.5, 0.5, 0.0, 0.4]], "detect").shape == (0, 5)


@pytest.mark.parametrize(
    "rows, expected",
    [
        (["0 0.5 0.5 0.2 0.4"], "detect"),
        (["0 0.1 0.1 0.3 0.1 0.3 0.3 0.1 0.3"], "segment"),
        ([], "detect"),
    ],
)
def test_auto_detection(tmp_path, rows, expected):
    for index, row in enumerate(rows):
        (tmp_path / f"img_{index}.txt").write_text(row + "\n")
    assert detect_label_format(tmp_path) == expected
