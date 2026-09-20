# Training this YOLO on your own data, and shipping it to a Raspberry Pi 4

A practical guide for using `MultimediaTechLab/YOLO` (MIT-licensed YOLOv7/v9) as a
commercial replacement for Ultralytics, including the traps that cost you a training run.

---

## 1. Licensing — why you are here

| | Ultralytics | this repo |
|---|---|---|
| License | AGPL-3.0 | MIT |
| Commercial use without paying | No — AGPL obliges you to publish the source of anything that links to it, *including over a network* | Yes |
| Attribution required | — | Keep the MIT copyright notice |

One caveat worth knowing: upstream issue **#202** asks whether the MIT license is
legitimate, since the original YOLOv9 reference implementation was GPLv3 and this repo
shares authorship (Wong Kin-Yiu) with it. The repo is authored by the YOLOv7/v9 authors
themselves, which is the strongest argument that they were entitled to relicense it, but
the question is open on the tracker. If this is going into a product, have whoever does
your legal review look at it once. The weights on the releases page are a separate
question from the code license.

---

## 2. What you need to know before you start

### 2.1 It is a research codebase, not a product

Read the open issues before committing. As of this writing the tracker says, in the
maintainers' and users' own words:

- **#203** — COCO training reaches ~36 AP where the paper reports ~51.
- **#231, #222** — validation numbers do not reproduce official benchmarks; batch size
  changes metrics in ways it should not.
- **#194** — mAP looks normal after epoch 0, then collapses to near zero.
- **#197** — custom-dataset metrics below 1% until augmentations are disabled.
- **#230** — no multi-GPU support or documentation.
- **#232** — segmentation/keypoint *training* is not implemented (inference only).

Several of these have concrete causes in the code, which section 4 addresses. But go in
knowing you are adopting a codebase that does not currently reproduce its own paper, and
budget time for validation you would not need with a mature framework.

### 2.2 Prerequisites

| Area | What you actually need |
|---|---|
| PyTorch | Comfortable reading `nn.Module`, state dicts, and tracing/export |
| PyTorch Lightning | The whole training loop is Lightning. You need `LightningModule`, `Callback`, `Trainer`, and how checkpoints are structured |
| Hydra / OmegaConf | All configuration is Hydra. You must be able to read `yolo/config/**.yaml`, understand `defaults:`, and use CLI overrides (`task.data.batch_size=8`) |
| Anchor-free detection | YOLOv9 is anchor-free with DFL (distribution focal loss) and a task-aligned assigner. To debug loss behaviour you need the concepts: anchor points, stride, reg_max, TAL |
| ONNX | Graph structure, opsets, static vs dynamic shapes |
| Quantization | Difference between dynamic and static (calibrated) PTQ, per-channel vs per-tensor scales, QDQ format |
| Linux/ARM | Cross-checking wheels, 64-bit Raspberry Pi OS, thread pinning |

### 2.3 Environment traps on Windows

Three things bit during setup and will bite you:

1. **A `pip install`ed copy shadows the repo.** `yolo/lazy.py` does `sys.path.append(project_root)` — *append*, so `site-packages` wins. If you ever ran `pip install git+...`, your edits to the repo will be silently ignored. Fix:
   ```bash
   pip uninstall -y yolo
   pip install -e .          # editable, points back at your working tree
   ```
2. **Emoji logging crashes on the legacy Windows console.** The logger emits `🏭`/`📦` and cp1252 cannot encode them, producing walls of `UnicodeEncodeError`. Fix: `set PYTHONIOENCODING=utf-8` (and `chcp 65001`).
3. **`device=cpu` does not work.** It is passed straight to Lightning's `devices=`, which wants an int. Use `accelerator=cpu device=1`.

---

## 3. Labelling your data

### 3.1 The format — read this twice

This repo's `.txt` reader was written for **normalized polygons**:

```
<class> <x1> <y1> <x2> <y2> ... <xn> <yn>
```

Every common labelling tool exports the **standard YOLO detection** row instead:

```
<class> <center_x> <center_y> <width> <height>
```

Feeding detection rows to the polygon reader **does not raise an error**. `0 0.5 0.5 0.2 0.4`
gets read as the point pairs `(0.5, 0.5)` and `(0.2, 0.4)`, so the box becomes
`xyxy = (0.2, 0.4, 0.5, 0.5)` instead of the correct `(0.4, 0.3, 0.6, 0.7)`. Your labels
are quietly scrambled, the loss still decreases, and mAP sits near zero. This is almost
certainly what is behind issue #197.

The patched loader in section 4 detects the format and logs which one it chose. Check
that line on every run:

```
🔍 Auto-detected label format 'detect' for train (<cls> <cx> <cy> <w> <h>)
```

You can force it with `task.data.label_format=detect` (or `segment`).

### 3.2 Bootstrapping labels from a model you already have

**You cannot convert an Ultralytics model into this repo.** An Ultralytics YOLOv8/v11
checkpoint and this repo's YOLOv9 are different networks — C2f vs RepNCSPELAN blocks, a
different neck, a different head. No weight-renaming script bridges that. (`yolo/tools/
format_converters.py` exists, but it maps from the *original* WongKinYiu/yolov9 layout,
not from Ultralytics.)

You do not need to convert. Use the Ultralytics model purely as a **label generator**:

```
Ultralytics model  ──auto-label──▶  draft .txt  ──human fixes──▶  final labels
                                                                        │
                            COCO-pretrained v9-s.pt ──train on them──▶ MIT model
                                                                        │
                                                          export + INT8 ──▶ Pi 4
```

The Ultralytics model never enters the shipped artifact; only the labels cross over, and
labels describing your own images are data. Running Ultralytics locally does not
distribute it, so AGPL obligations are not triggered — just do not ship its weights.

`tools/autolabel.py` does this, with two interchangeable backends:

```bash
# Round 1 - bootstrap from an Ultralytics model.
# Source classes are matched to your target classes BY NAME; use --map when they differ.
python tools/autolabel.py --model yolov8n.pt --images data/raw     --out data/custom --split train     --classes floc foam scum --map "person:0,boat:1"     --conf 0.35 --copy-images

# Round 2+ - relabel the next batch with your OWN exported model, no Ultralytics at all.
python tools/autolabel.py --model deploy_out/model_fp32.onnx --images data/raw_batch2     --out data/custom --split train --classes floc foam scum --conf 0.35
```

It writes standard `<cls> <cx> <cy> <w> <h>` labels (auto-detected correctly by the
loader), keeps empty `.txt` files as background samples, and emits
`autolabel_report_<split>.json` listing images whose best detection fell below
`--review-below`. **Review those first** — that list is where the model is least certain
and where your correction time is worth the most.

Two things auto-labelling cannot do, so plan for them:

- It **cannot invent a class it was never taught**. A COCO model has no concept of
  "floc" or "scum". For genuinely new classes, hand-label a first few hundred images,
  train a rough model here, then use *that* for round 2. The loop pays off from round 2 on.
- **Missed objects are worse than no label.** An unlabelled object teaches the model that
  region is background. Always eyeball every frame, even the confident ones.

### 3.3 Directory layout

```
data/custom/
├── images/
│   ├── train/   img_0001.jpg ...
│   └── val/     img_9001.jpg ...
└── labels/
    ├── train/   img_0001.txt ...     # same stem as the image
    └── val/     img_9001.txt ...
```

All coordinates normalized to `[0, 1]`. Classes are 0-based indices into `class_list`.
**Keep empty `.txt` files** for images with no objects — they are valid background samples
and reduce false positives.

### 3.4 Open-source labelling tools

Yes, you can label your base images with free software. In rough order of what I would
pick for your case:

| Tool | Why | License |
|---|---|---|
| **Label Studio** | Best all-rounder. Web UI, multi-user, exports YOLO directly, supports pre-annotation from your own model so each round gets faster | Apache-2.0 |
| **CVAT** | Strongest for video/streams. Interpolation between keyframes is a huge time saver for camera footage. Self-host with Docker | MIT |
| **LabelImg** | Simplest possible. Single desktop app, draws boxes, writes YOLO `.txt`. Now folded into Label Studio but still works | MIT |
| **labelme** | Polygon-first. Use if you want masks now and boxes later | MIT |
| **X-AnyLabeling** | LabelImg-style UI with SAM/YOLO-assisted auto-labelling built in. Big speedup on repetitive scenes | GPL-3.0 — fine as a *tool*, it does not touch your model's license |

Avoid Roboflow's hosted free tier for commercial work unless you read their terms — the
free plan makes datasets public. The export format itself is fine.

**Practical labelling advice for a fixed camera:**

- Label every instance in the frame, every time. One missed object teaches the model that region is background.
- Boxes tight to the object, no margin.
- Include the hard frames — glare, foam, night, dirty lens. A model trained only on clean frames fails exactly when you need it.
- ~300-500 instances per class is a workable start for fine-tuning from pretrained weights; a few thousand for production.
- Split train/val **by time or by scene**, never randomly. Consecutive video frames are near-duplicates, and a random split leaks them across the split, giving you a val score that is pure fiction.

---

## 4. What was changed in this repo, and why

All changes are in-place and behaviour-compatible unless noted.

### Correctness

| File | Change |
|---|---|
| `yolo/tools/data_loader.py` | **Detection label format supported + auto-detected.** Previously only polygons parsed correctly (silent corruption otherwise) |
| `yolo/tools/data_loader.py` | **`shuffle` is now actually applied.** It was in the config but never passed to the `DataLoader`, so every epoch saw the same order. Disabled automatically under `dynamic_shape`, which needs sorted batches |
| `yolo/tools/data_loader.py` | The 100-box-per-image cap now warns instead of silently truncating, and is configurable via `task.data.max_bbox` |
| `yolo/tools/data_loader.py` | `drop_last` is enabled for training (stable BatchNorm stats) but **disengages below two full batches**, so a small dataset is not discarded entirely; warns if `batch_size` exceeds the dataset size |
| `yolo/tools/data_loader.py` | Label cache filename now includes the format, so switching formats rebuilds it instead of reusing bad boxes |
| `yolo/utils/bounding_box_utils.py` | **NMS forced to float32** (upstream #234). Under the hardcoded `16-mixed` precision, torchvision's `batched_nms` offsets boxes by `group_idx * (max_coord+1)` in float16, which loses integer exactness past 2048 and lets boxes from different images/classes suppress each other |
| `yolo/utils/model_utils.py` | EMA no longer `lerp`s integer buffers (`num_batches_tracked`), which either errors or silently corrupts them |
| `yolo/utils/logging_utils.py` | `metrics.pop("v_num")` → `.pop("v_num", None)`. Training **crashed outright** with `use_wandb=False` and `use_tensorboard=False` |

### Ultralytics parity

| File | Change |
|---|---|
| `yolo/tools/data_augmentation.py` | **`HSVJitter`** added (h=0.015, s=0.7, v=0.4). Was entirely missing; it is the cheapest robustness win for lighting/colour shift |
| `yolo/tools/data_augmentation.py` | **`RandomAffine`** added — rotation/translate/scale/shear/perspective, a port of v5/v8 `random_perspective`. Scale jitter is the single strongest augmentation in the v8 recipe |
| `yolo/tools/data_augmentation.py` | **`Mosaic` rewritten.** The old one pasted 4 unresized images at fixed offsets, never clipped boxes to the canvas, and kept boxes entirely outside it. Now: per-tile rescale, random centre, clipping, degenerate-box removal, and a proper 2× canvas that `RandomAffine` crops back |
| `yolo/tools/data_augmentation.py` | **`box_candidates`** filter after every geometric transform (min size, aspect ratio, surviving-area ratio), so warped-away slivers stop being fed to the assigner as positives |
| `yolo/tools/data_augmentation.py` | `MixUp` fixed — it crashed on differently-sized images, and its `alpha=1.0` uniform mix is replaced by v8's `Beta(32, 32)` |
| `yolo/tools/data_augmentation.py` | `PadAndResize` uses BILINEAR instead of LANCZOS (~5× faster, no measurable mAP cost) |
| `yolo/utils/model_utils.py` | **`CloseMosaic` callback** — disables Mosaic/MixUp for the final N epochs (`task.close_mosaic`, default 10), as every modern YOLO recipe does |
| `yolo/utils/logging_utils.py` | **`ModelCheckpoint` added.** There was none, so only the last epoch survived — a 300-epoch run threw away its best model |
| `yolo/config/task/train.yaml` | Augmentation defaults rewritten to the v8 recipe. Notably `HorizontalFlip` was commented out and `RandomCrop: 1` (always crop to 50%) was on — an unusual and destructive default |

### Deployment (new files)

| File | Purpose |
|---|---|
| `yolo/tools/export.py` | Deploy-mode ONNX export: strips the auxiliary branch, folds anchor decoding into the graph, static shapes. Also `load_state_from_checkpoint`, which reads the **EMA** weights |
| `yolo/tools/quantize.py` | Static (calibrated) INT8 PTQ with an image calibration reader, plus dynamic PTQ |
| `tools/export_edge.py` | One-shot CLI: checkpoint → FP32 ONNX → INT8 ONNX + accuracy drift report |
| `deploy/rpi_infer.py` | Standalone Pi runtime. onnxruntime + numpy + Pillow only — no torch |
| `tools/autolabel.py` | Auto-label unlabelled images with an existing Ultralytics `.pt` or an exported `.onnx` from this repo, for human correction; reports a priority-review list |

> **The EMA problem.** A training checkpoint contains *both* `model.model.*` (raw SGD
> weights) and `ema.model.*` (the moving average). The EMA copy is what gets validated and
> is normally the better model, but `YOLO.save_load_weights` only reads the
> `model.model.` prefix — so loading a checkpoint the built-in way silently gives you the
> **worse** weights. `export.py` defaults to EMA; pass `--no-ema` to override.

---

## 5. Training

```bash
# 1. point a config at your data
cp yolo/config/dataset/custom.yaml yolo/config/dataset/mydata.yaml
#    edit: path, class_num, class_list

# 2. fine-tune from pretrained COCO weights (v9-t / v9-s / v9-m / v9-c / v7 available)
python yolo/lazy.py task=train dataset=mydata model=v9-s weight=True \
    task.epoch=150 task.data.batch_size=16 image_size=[416,416] \
    name=myrun use_wandb=False
```

Check the first 20 lines of output for:

- `🔍 Auto-detected label format 'detect'` — **the** thing to verify
- `Recorded N/N valid inputs` — if the first number is much lower, images and labels are not pairing up
- `✅ Success load model & weight` — pretrained weights actually loaded

**Sizing for a Pi 4.** Use `v9-t` or `v9-s`, and train at the resolution you will deploy
at. 416×416 or 320×320 is realistic; 640 on a Pi 4 CPU is roughly 1-2 FPS. Resolution
costs quadratically and is usually a better thing to cut than model capacity.

> **A note on the shuffle fix.** `tests/test_tools/test_data_loader.py` previously asserted a
> *fixed* batch order, which only held because shuffling was broken. That assertion has been
> replaced with a set-membership check plus explicit tests for shuffling and for the
> `drop_last` guard. Validation loaders remain sequential and never drop a batch.

**If mAP collapses after epoch 0** (issue #194), try in this order: confirm the label
format line; drop `lr` to 0.001; set `task.close_mosaic` to most of your epoch count to
effectively disable mosaic; reduce `RandomAffine.scale` to 0.2.

---

## 6. Export and quantize for the Pi

```bash
python tools/export_edge.py \
    --checkpoint runs/train/myrun/checkpoints/last.ckpt \
    --model v9-s --classes 3 \
    --image-size 416 416 \
    --calib-dir data/custom/images/train --calib-images 300 \
    --out-dir deploy_out
```

Produces `model_fp32.onnx` and `model_int8.onnx`, then reports drift between them.
**Mean box drift under ~2 px and score correlation above ~0.99** means the quantization
is safe. Worse than that: add calibration images, or ship FP32.

Calibration images must come from the **deployment camera** and cover its real range of
lighting and conditions. 300 frames is plenty.

Then on the Pi (64-bit Raspberry Pi OS — the 32-bit image has no optimised wheels):

```bash
pip3 install onnxruntime numpy pillow
python3 rpi_infer.py --model model_int8.onnx --source frame.jpg \
    --classes floc foam scum --benchmark
```

### What to expect on a Pi 4

The Pi 4's Cortex-A72 is ARMv8.0: NEON SIMD, but **no dot-product instruction** (that
arrives with the A76 in the Pi 5). So INT8 gains come mostly from memory bandwidth and
cache pressure, not integer throughput — budget roughly **1.5-2.5×** over FP32, not the
4× quoted for server CPUs.

I could not measure the real speedup for you here: this dev machine is x86, where ONNX
Runtime takes a different kernel path entirely (FP32 and INT8 benchmarked identically at
256×416 on this box). **Benchmark on the actual Pi before committing to a resolution.**

Also:
- Use a heatsink. A bare Pi 4 throttles within minutes of sustained inference and your FPS quietly halves.
- `--threads 4` matches the core count. More threads makes latency worse.
- If ONNX Runtime is not fast enough, **NCNN** is usually the fastest option on ARM CPUs — convert with `onnx2ncnn` and use its INT8 tooling. It is a larger integration effort, so try ONNX Runtime first.

---

## 7. Honest expectations vs Ultralytics

The changes above close the *mechanical* gap — same augmentation recipe, same mosaic
semantics, correct labels, correct shuffling, EMA weights actually deployed, checkpoints
actually kept. That is most of what separates a working run from a broken one.

They do **not** guarantee parity of final mAP. Issue #203's ~36-vs-51 AP gap on COCO is
not fully explained by anything fixed here, and its root cause is still open upstream.
Ultralytics also brings things this repo does not have at all: auto-batch, auto-anchor
diagnostics, mosaic scheduling tied to the LR schedule, a validated hyperparameter set
per model scale, and years of accumulated tuning.

The pragmatic path: **train a baseline both ways on your dataset and compare mAP on your
own validation split.** Use Ultralytics purely as a local benchmark (AGPL only constrains
what you *distribute*; evaluating it internally does not oblige you to publish anything).
If this repo lands within a couple of mAP points, ship it and keep your MIT license. If
the gap is large on your data, you have a concrete number to take to the upstream issue
rather than a hunch.
