"""Ultralytics-shaped validation reporting for this trainer.

Two things Ultralytics gives you that this repo does not:

  1. a per-class table at each validation, so you can see which class is weak
  2. a `results.csv` per run, which is what most analysis notebooks read

This module adds both without touching how training works. Column names follow
Ultralytics (`metrics/mAP50-95(B)` and friends) so existing pandas code that was
written against Ultralytics runs keeps working.

HONEST LIMITS - read before comparing numbers:

  * Precision and Recall are NOT emitted. Ultralytics reports them at a chosen
    confidence threshold; torchmetrics' MeanAveragePrecision does not produce
    them at all. Inventing a column would make runs look comparable when they
    are not, so those columns are simply absent.
  * Per-class **mAP50** is unavailable. torchmetrics gives `map_per_class` and
    `mar_100_per_class` only - there is no `map_50_per_class`. The per-class
    table therefore shows mAP50-95 and mAR100; mAP50 appears on the `all` row.
  * mAP50 and mAP50-95 ARE directly comparable with Ultralytics: both are
    COCO-style averages over the same IoU thresholds.
"""

import csv
from pathlib import Path
from typing import Dict, List, Optional

from lightning.pytorch.callbacks import Callback
from rich.console import Console
from rich.table import Table

from yolo.utils.logger import logger


def _scalar(value) -> Optional[float]:
    """Pull a plain float out of a tensor / number, or None if it is neither."""
    if value is None:
        return None
    try:
        return float(value.item()) if hasattr(value, "item") else float(value)
    except (ValueError, TypeError):
        return None


class UltralyticsStyleReport(Callback):
    """Print an Ultralytics-style table after validation and append to results.csv."""

    def __init__(self, save_path: Path, class_list: Optional[List[str]] = None, quiet: bool = False):
        super().__init__()
        self.quiet = quiet
        self.csv_path = Path(save_path) / "results.csv"
        self.class_list = list(class_list) if class_list else []
        self._rows: List[Dict] = []
        self._counts: Optional[Dict[int, int]] = None

    # -- ground-truth counts, for the Instances column ----------------------
    def _instance_counts(self, pl_module) -> Dict[int, int]:
        """Boxes per class in the validation set. Best effort - never fatal."""
        if self._counts is not None:
            return self._counts
        counts: Dict[int, int] = {}
        try:
            boxes = pl_module.val_loader.dataset.bboxes
            for img_boxes in boxes:
                for box in img_boxes:
                    cls = int(box[0])
                    if cls >= 0:                      # -1 rows are padding
                        counts[cls] = counts.get(cls, 0) + 1
        except Exception:                             # dataset shape differs, skip the column
            counts = {}
        self._counts = counts
        return counts

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        m = trainer.callback_metrics
        overall = {
            "mAP50": _scalar(m.get("map_50")),
            "mAP50-95": _scalar(m.get("map")),
            "mAR100": _scalar(m.get("mar_100")),
            "mAP_small": _scalar(m.get("map_small")),
            "mAP_medium": _scalar(m.get("map_medium")),
            "mAP_large": _scalar(m.get("map_large")),
        }
        # stashed by ValidateModel.on_validation_epoch_end; absent if class_metrics is off
        per_class = getattr(pl_module, "per_class_metrics", {}) or {}

        if not self.quiet:
            self._print_table(trainer, pl_module, overall, per_class)
        self._append_csv(trainer, overall, per_class)

    # -- console table -------------------------------------------------------
    def _print_table(self, trainer, pl_module, overall, per_class):
        n_images = 0
        try:
            n_images = len(pl_module.val_loader.dataset)
        except Exception:
            pass
        counts = self._instance_counts(pl_module)

        table = Table(title=f"Validation - epoch {trainer.current_epoch}", header_style="bold magenta")
        table.add_column("Class", justify="left")
        table.add_column("Images", justify="right")
        table.add_column("Instances", justify="right")
        table.add_column("mAP50", justify="right")
        table.add_column("mAP50-95", justify="right")
        table.add_column("mAR100", justify="right")

        def fmt(v):
            return "-" if v is None else f"{v:.4f}"

        table.add_row(
            "[bold]all[/]",
            str(n_images) if n_images else "-",
            str(sum(counts.values())) if counts else "-",
            fmt(overall["mAP50"]),
            fmt(overall["mAP50-95"]),
            fmt(overall["mAR100"]),
        )
        for idx, name in enumerate(self.class_list):
            stats = per_class.get(name)
            if stats is None:
                continue
            table.add_row(
                name,
                "-",
                str(counts.get(idx, "-")),
                "-",                                   # no map_50_per_class in torchmetrics
                fmt(stats.get("mAP50-95")),
                fmt(stats.get("mAR100")),
            )

        Console().print(table)
        if not per_class and self.class_list:
            logger.warning(
                ":warning: No per-class metrics. Build MeanAveragePrecision with "
                "class_metrics=True to populate the per-class rows."
            )

    # -- results.csv ---------------------------------------------------------
    def _append_csv(self, trainer, overall, per_class):
        row = {
            "epoch": trainer.current_epoch,
            "metrics/mAP50(B)": overall["mAP50"],
            "metrics/mAP50-95(B)": overall["mAP50-95"],
            "metrics/mAR100(B)": overall["mAR100"],
            "metrics/mAP50-95_small(B)": overall["mAP_small"],
            "metrics/mAP50-95_medium(B)": overall["mAP_medium"],
            "metrics/mAP50-95_large(B)": overall["mAP_large"],
        }
        for name, stats in per_class.items():
            row[f"metrics/mAP50-95({name})"] = stats.get("mAP50-95")
            row[f"metrics/mAR100({name})"] = stats.get("mAR100")
        for key, value in trainer.callback_metrics.items():
            if key.startswith("Loss/") and key.endswith("_epoch"):
                row[f"train/{key[len('Loss/'):-len('_epoch')].lower()}"] = _scalar(value)

        # Columns are NOT stable between epochs: per-class keys only appear once a class
        # has predictions, and the loss keys arrive after the first train epoch. Appending
        # with per-row fieldnames writes a header of one width and later rows of another,
        # which makes the file unparseable ("expected 11 fields, saw 14"). So keep every
        # row in memory and rewrite the whole file against the union of all keys. It is at
        # most a few hundred rows; correctness matters far more than the write cost here.
        self._rows.append(row)
        fieldnames = ["epoch"]
        for r in self._rows:
            for key in r:
                if key not in fieldnames:
                    fieldnames.append(key)

        try:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            with self.csv_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames, restval="")
                writer.writeheader()
                writer.writerows(self._rows)
        except Exception as err:                       # logging must never kill a run
            logger.warning(f":warning: Could not write {self.csv_path}: {err}")
