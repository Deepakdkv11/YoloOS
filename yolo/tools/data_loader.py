from collections.abc import Mapping
from functools import partial
from pathlib import Path
from queue import Empty, Queue
from statistics import mean
from threading import Event, Thread
from typing import Generator, List, Tuple, Union

import numpy as np
import torch
from PIL import Image
from rich.progress import track
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from yolo.config.config import DataConfig, DatasetConfig
from yolo.tools.data_augmentation import *
from yolo.tools.data_augmentation import AugmentationComposer
from yolo.tools.dataset_preparation import prepare_dataset
from yolo.utils.dataset_utils import (
    create_image_metadata,
    detect_label_format,
    locate_label_paths,
    scale_segmentation,
    tensorlize,
)
from yolo.utils.logger import logger


class YoloDataset(Dataset):
    def __init__(self, data_cfg: DataConfig, dataset_cfg: DatasetConfig, phase: str = "train2017"):
        augment_cfg = data_cfg.data_augment
        self.image_size = data_cfg.image_size
        phase_name = dataset_cfg.get(phase, phase)
        self.batch_size = data_cfg.batch_size
        self.dynamic_shape = getattr(data_cfg, "dynamic_shape", False)
        self.base_size = mean(self.image_size)

        # "auto" | "detect" | "segment"; resolved against the label files in filter_data().
        self.label_format = getattr(data_cfg, "label_format", "auto")

        # Each augmentation may be configured either as `Name: <prob>` (legacy) or as
        # `Name: {prob: ..., <other kwargs>}`, which is needed for HSVJitter/RandomAffine.
        transforms = []
        for aug, params in augment_cfg.items():
            aug_class = eval(aug)
            if isinstance(params, Mapping):
                transforms.append(aug_class(**params))
            else:
                transforms.append(aug_class(params))
        self.transform = AugmentationComposer(transforms, self.image_size, self.base_size)
        self.transform.get_more_data = self.get_more_data
        self.img_paths, self.bboxes, self.ratios = tensorlize(self.load_data(Path(dataset_cfg.path), phase_name))

    def load_data(self, dataset_path: Path, phase_name: str):
        """
        Loads data from a cache or generates a new cache for a specific dataset phase.

        Parameters:
            dataset_path (Path): The root path to the dataset directory.
            phase_name (str): The specific phase of the dataset (e.g., 'train', 'test') to load or generate data for.

        Returns:
            dict: The loaded data from the cache for the specified phase.
        """
        # The cache stores already-decoded boxes, so it is only valid for the label format
        # it was built with. Encoding the format in the filename means switching formats
        # (or fixing a mis-detection) rebuilds the cache instead of silently reusing bad boxes.
        cache_path = dataset_path / f"{phase_name}.{self.label_format}.pache"
        legacy_cache = dataset_path / f"{phase_name}.pache"
        if legacy_cache.exists() and not cache_path.exists():
            logger.warning(
                f":warning: Found a legacy cache at '{legacy_cache}' built before label-format "
                "support. Ignoring it; delete it to silence this message."
            )

        if not cache_path.exists():
            logger.info(f":factory: Generating {phase_name} cache")
            data = self.filter_data(dataset_path, phase_name, self.dynamic_shape)
            torch.save(data, cache_path)
        else:
            try:
                data = torch.load(cache_path, weights_only=False)
            except Exception as e:
                logger.error(
                    f":rotating_light: Failed to load the cache at '{cache_path}'.\n"
                    ":rotating_light: This may be caused by using cache from different other YOLO.\n"
                    ":rotating_light: Please clean the cache and try running again."
                )
                raise e
            logger.info(f":package: Loaded {phase_name} cache, there are {len(data)} data in total.")
        return data

    def filter_data(self, dataset_path: Path, phase_name: str, sort_image: bool = False) -> list:
        """
        Filters and collects dataset information by pairing images with their corresponding labels.

        Parameters:
            images_path (Path): Path to the directory containing image files.
            labels_path (str): Path to the directory containing label files.
            sort_image (bool): If True, sorts the dataset by the width-to-height ratio of images in descending order.

        Returns:
            list: A list of tuples, each containing the path to an image file and its associated segmentation as a tensor.
        """
        images_path = dataset_path / "images" / phase_name
        labels_path, data_type = locate_label_paths(dataset_path, phase_name)
        file_list, adjust_path = dataset_path / f"{phase_name}.txt", False
        if file_list.exists():
            data_type, adjust_path = "txt", True
            # TODO: should i sort by name?
            with open(file_list, "r") as file:
                images_list = [dataset_path / line.rstrip() for line in file]
            labels_list = [
                Path(str(image_path).replace("images", "labels")).with_suffix(".txt") for image_path in images_list
            ]
        else:
            images_list = sorted([p.name for p in Path(images_path).iterdir() if p.is_file()])

        if data_type == "json":
            annotations_index, image_info_dict = create_image_metadata(labels_path)

        # Resolve the .txt label layout once, up front, and say out loud which one we picked.
        if data_type == "txt":
            if self.label_format == "auto":
                probe_dir = labels_path if not adjust_path else Path(labels_list[0]).parent
                self.label_format = detect_label_format(probe_dir)
                logger.info(
                    f":mag: Auto-detected label format '{self.label_format}' for {phase_name} "
                    f"({'<cls> <cx> <cy> <w> <h>' if self.label_format == 'detect' else '<cls> <x1> <y1> ... <xn> <yn>'})"
                )
            elif self.label_format not in ("detect", "segment"):
                raise ValueError(
                    f"task.data.label_format must be one of 'auto', 'detect', 'segment'; got {self.label_format!r}"
                )

        data = []
        valid_inputs = 0
        for idx, image_name in enumerate(track(images_list, description="Filtering data")):
            if not adjust_path and not image_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            image_id = Path(image_name).stem

            if data_type == "json":
                image_info = image_info_dict.get(image_id, None)
                if image_info is None:
                    continue
                annotations = annotations_index.get(image_info["id"], [])
                image_seg_annotations = scale_segmentation(annotations, image_info)
            elif data_type == "txt":
                label_path = labels_list[idx] if adjust_path else labels_path / f"{image_id}.txt"
                if not label_path.is_file():
                    image_seg_annotations = []
                else:
                    with open(label_path, "r") as file:
                        image_seg_annotations = [list(map(float, line.strip().split())) for line in file]
            else:
                image_seg_annotations = []

            row_format = self.label_format if data_type == "txt" else "segment"
            labels = self.load_valid_labels(image_id, image_seg_annotations, row_format)
            img_path = image_name if adjust_path else images_path / image_name
            if sort_image:
                with Image.open(img_path) as img:
                    width, height = img.size
            else:
                width, height = 0, 1
            data.append((img_path, labels, width / height))
            if len(image_seg_annotations) != 0:
                valid_inputs += 1

        data = sorted(data, key=lambda x: x[2], reverse=True)

        logger.info(f"Recorded {valid_inputs}/{len(images_list)} valid inputs")
        return data

    def load_valid_labels(self, label_path: str, seg_data_one_img: list, label_format: str = "segment") -> Tensor:
        """
        Converts one image's raw annotation rows into normalized xyxy boxes.

        Two row layouts are supported:
          - "segment": `<cls> <x1> <y1> ... <xn> <yn>` — a normalized polygon, collapsed to
            its axis-aligned bounding box. This is what COCO json and this repo's own
            data_conversion.py produce.
          - "detect":  `<cls> <cx> <cy> <w> <h>` — the standard YOLO detection row emitted
            by LabelImg, label-studio, CVAT and Roboflow.

        Parameters:
            label_path (str): The filepath to the label file containing annotation data.
            seg_data_one_img (list): The actual list of annotation rows.
            label_format (str): "segment" or "detect".

        Returns:
            Tensor: (N, 5) of [cls, x_min, y_min, x_max, y_max], normalized. Empty if none valid.
        """
        bboxes = []
        for seg_data in seg_data_one_img:
            cls = seg_data[0]
            values = seg_data[1:]

            if label_format == "detect":
                if len(values) != 4:
                    logger.warning(
                        f"Skipping malformed detection row in {label_path}: expected 4 values "
                        f"after the class id, got {len(values)}."
                    )
                    continue
                center_x, center_y, width, height = values
                if width <= 0 or height <= 0:
                    continue
                x_min, y_min = center_x - width / 2, center_y - height / 2
                x_max, y_max = center_x + width / 2, center_y + height / 2
                bbox = torch.tensor(
                    [cls, min(max(x_min, 0.0), 1.0), min(max(y_min, 0.0), 1.0),
                     min(max(x_max, 0.0), 1.0), min(max(y_max, 0.0), 1.0)]
                )
                if bbox[3] > bbox[1] and bbox[4] > bbox[2]:
                    bboxes.append(bbox)
                continue

            points = np.array(values).reshape(-1, 2).clip(0, 1)
            valid_points = points[(points >= 0) & (points <= 1)].reshape(-1, 2)
            if valid_points.size > 1:
                bbox = torch.tensor([cls, *valid_points.min(axis=0), *valid_points.max(axis=0)])
                bboxes.append(bbox)

        if bboxes:
            # Always float32: the polygon branch builds its tensor from float64 numpy
            # scalars, so without this the two branches return different dtypes.
            return torch.stack(bboxes).to(torch.float32)
        else:
            logger.warning(f"No valid BBox in {label_path}")
            return torch.zeros((0, 5), dtype=torch.float32)

    def get_data(self, idx):
        img_path, bboxes = self.img_paths[idx], self.bboxes[idx]
        valid_mask = bboxes[:, 0] != -1
        with Image.open(img_path) as img:
            img = img.convert("RGB")
        return img, torch.from_numpy(bboxes[valid_mask]), img_path

    def get_more_data(self, num: int = 1):
        indices = torch.randint(0, len(self), (num,))
        return [self.get_data(idx)[:2] for idx in indices]

    def _update_image_size(self, idx: int) -> None:
        """Update image size based on dynamic shape and batch settings."""
        batch_start_idx = (idx // self.batch_size) * self.batch_size
        image_ratio = self.ratios[batch_start_idx].clip(1 / 3, 3)
        shift = ((self.base_size / 32 * (image_ratio - 1)) // (image_ratio + 1)) * 32

        self.image_size = [int(self.base_size + shift), int(self.base_size - shift)]
        self.transform.pad_resize.set_size(self.image_size)

    def __getitem__(self, idx) -> Tuple[Image.Image, Tensor, Tensor, List[str]]:
        img, bboxes, img_path = self.get_data(idx)

        if self.dynamic_shape:
            self._update_image_size(idx)

        img, bboxes, rev_tensor = self.transform(img, bboxes)
        bboxes[:, [1, 3]] *= self.image_size[0]
        bboxes[:, [2, 4]] *= self.image_size[1]
        return img, bboxes, rev_tensor, img_path

    def __len__(self) -> int:
        return len(self.bboxes)


_MAX_BOX_WARNED = False


def collate_fn(batch: List[Tuple[Tensor, Tensor]], max_bbox: int = 100) -> Tuple[Tensor, List[Tensor]]:
    """
    A collate function to handle batching of images and their corresponding targets.

    Args:
        batch (list of tuples): Each tuple contains:
            - image (Tensor): The image tensor.
            - labels (Tensor): The tensor of labels for the image.
        max_bbox (int): Hard cap on ground-truth boxes kept per image. Anything beyond this
            is dropped, so crowded scenes silently lose supervision. Raise it via
            `task.data.max_bbox` if your images have more than 100 objects.

    Returns:
        Tuple[Tensor, List[Tensor]]: A tuple containing:
            - A tensor of batched images.
            - A list of tensors, each corresponding to bboxes for each image in the batch.
    """
    global _MAX_BOX_WARNED
    batch_size = len(batch)
    target_sizes = [item[1].size(0) for item in batch]

    if max(target_sizes, default=0) > max_bbox and not _MAX_BOX_WARNED:
        _MAX_BOX_WARNED = True
        logger.warning(
            f":warning: An image carries {max(target_sizes)} ground-truth boxes but max_bbox={max_bbox}; "
            "the extras are being dropped. Increase task.data.max_bbox to keep them."
        )

    batch_targets = torch.zeros(batch_size, min(max(target_sizes, default=1), max_bbox), 5)
    batch_targets[:, :, 0] = -1
    for idx, target_size in enumerate(target_sizes):
        batch_targets[idx, : min(target_size, max_bbox)] = batch[idx][1][:max_bbox]

    batch_images, _, batch_reverse, batch_path = zip(*batch)
    batch_images = torch.stack(batch_images)
    batch_reverse = torch.stack(batch_reverse)

    return batch_size, batch_images, batch_targets, batch_reverse, batch_path


def create_dataloader(data_cfg: DataConfig, dataset_cfg: DatasetConfig, task: str = "train"):
    if task == "inference":
        return StreamDataLoader(data_cfg)

    if getattr(dataset_cfg, "auto_download", False):
        prepare_dataset(dataset_cfg, task)
    dataset = YoloDataset(data_cfg, dataset_cfg, task)
    max_bbox = getattr(data_cfg, "max_bbox", 100)

    # `shuffle` was present in the config but never forwarded to the DataLoader, so training
    # always walked the dataset in the same (aspect-ratio-sorted) order every epoch.
    # dynamic_shape deliberately relies on that ordering to batch similar shapes together,
    # so shuffling stays off in that mode.
    shuffle = bool(getattr(data_cfg, "shuffle", False)) and task == "train"
    if shuffle and getattr(data_cfg, "dynamic_shape", False):
        logger.warning(
            ":warning: dynamic_shape batches images by aspect ratio, which requires a sorted "
            "sampler. Disabling shuffle for this dataloader."
        )
        shuffle = False

    # Dropping the ragged last batch keeps BatchNorm statistics stable, but only when
    # there is more than one full batch to begin with: on a small dataset (say 30 images
    # with batch_size=32) drop_last would discard *every* batch and train on nothing.
    drop_last = task == "train" and len(dataset) >= 2 * data_cfg.batch_size
    if task == "train" and not drop_last and len(dataset) < data_cfg.batch_size:
        logger.warning(
            f":warning: Only {len(dataset)} samples for a batch_size of {data_cfg.batch_size}; "
            "every step will be a partial batch. Lower task.data.batch_size."
        )

    return DataLoader(
        dataset,
        batch_size=data_cfg.batch_size,
        num_workers=data_cfg.cpu_num,
        pin_memory=data_cfg.pin_memory,
        shuffle=shuffle,
        drop_last=drop_last,
        # Deliberately NOT persistent: workers are re-forked each epoch, which is what lets
        # the CloseMosaic callback's mutation of transform.prob reach them.
        persistent_workers=False,
        collate_fn=partial(collate_fn, max_bbox=max_bbox),
    )


class StreamDataLoader:
    def __init__(self, data_cfg: DataConfig):
        self.source = data_cfg.source
        self.running = True
        self.is_stream = isinstance(self.source, int) or str(self.source).lower().startswith("rtmp://")

        self.transform = AugmentationComposer([], data_cfg.image_size)
        self.stop_event = Event()

        if self.is_stream:
            import cv2

            self.cap = cv2.VideoCapture(self.source)
        else:
            self.source = Path(self.source)
            self.queue = Queue()
            self.thread = Thread(target=self.load_source)
            self.thread.start()

    def load_source(self):
        if self.source.is_dir():  # image folder
            self.load_image_folder(self.source)
        elif any(self.source.suffix.lower().endswith(ext) for ext in [".mp4", ".avi", ".mkv"]):  # Video file
            self.load_video_file(self.source)
        else:  # Single image
            self.process_image(self.source)

    def load_image_folder(self, folder):
        folder_path = Path(folder)
        for file_path in folder_path.rglob("*"):
            if self.stop_event.is_set():
                break
            if file_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                self.process_image(file_path)

    def process_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        if image is None:
            raise ValueError(f"Error loading image: {image_path}")
        self.process_frame(image)

    def load_video_file(self, video_path):
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            self.process_frame(frame)
        cap.release()

    def process_frame(self, frame):
        if isinstance(frame, np.ndarray):
            # TODO: we don't need cv2
            import cv2

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
        origin_frame = frame
        frame, _, rev_tensor = self.transform(frame, torch.zeros(0, 5))
        frame = frame[None]
        rev_tensor = rev_tensor[None]
        if not self.is_stream:
            self.queue.put((frame, rev_tensor, origin_frame))
        else:
            self.current_frame = (frame, rev_tensor, origin_frame)

    def __iter__(self) -> Generator[Tensor, None, None]:
        return self

    def __next__(self) -> Tensor:
        if self.is_stream:
            ret, frame = self.cap.read()
            if not ret:
                self.stop()
                raise StopIteration
            self.process_frame(frame)
            return self.current_frame
        else:
            try:
                frame = self.queue.get(timeout=1)
                return frame
            except Empty:
                raise StopIteration

    def stop(self):
        self.running = False
        if self.is_stream:
            self.cap.release()
        else:
            self.thread.join(timeout=1)

    def __len__(self):
        return self.queue.qsize() if not self.is_stream else 0
