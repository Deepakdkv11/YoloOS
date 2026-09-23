# Run log

One entry per training run. `runs/` is gitignored, so without this the only record of
what produced a given checkpoint is the S3 bucket and your memory.

Copy the block below for each new run. Fill in the results even when a run fails —
a failed run you cannot reconstruct is a run you will repeat.

---

## `sludge-v9s-run1` — 2026-09-23

First full run. Baseline.

### Command

```bash
python yolo/lazy.py \
    task=train dataset=sludge model=v9-s weight=True \
    accelerator=gpu device=1 use_wandb=True \
    task.epoch=150 task.close_mosaic=10 task.patience=30 \
    task.data.batch_size=16 task.validation.data.batch_size=16 \
    task.data.cpu_num=12 task.validation.data.cpu_num=12 \
    image_size=[640,640] name=sludge-v9s-run1
```

### Environment

| | |
|---|---|
| hardware | Colab A100, High-RAM, 12 vCPU |
| trainer | `Deepakdkv11/YoloOS` @ `sludge-training` |
| dataset | `data/custom`, 3493 train / 309 val / 318 test |
| classes | `floatingsludge` (0), `sludge` (1) |
| split | `06_split_dataset.py --prune-leaks --seed-search 400`, 0.0% cross-split near-duplicates |

### Outcome

Stopped by `patience=30` at **epoch 60** of 150. Wall clock **1h 54m** (~1.9 min/epoch).
Best epoch **30**.

```
s3://digitalpaani-bht-cv-data/runs/sludge-v9s-run1/checkpoints/best-030-0.6747.ckpt
```

Peak val mAP@50:95 = 67.47 at epoch 30, then a sustained ~7-point decline through
epoch 60 — overfitting, not noise. `save_top_k=3` preserved the peak.

### Results

Same checkpoint, same command, both splits:

| metric | val | test |
|---|---|---|
| mAP@50 | 90.42 | 99.31 |
| **mAP@50:95** | **66.11** | **88.17** |
| mAR@100 | 76.07 | 90.37 |
| mAP@50:95 (medium) | 26.68 | 96.90 |
| mAP@50:95 (large) | 69.69 | 88.81 |
| `floatingsludge` mAP@50:95 | **48.74** | 81.27 |
| `sludge` mAP@50:95 | 83.47 | 95.07 |

### Findings

1. **Val and test disagree by 22 points.** Not leakage — the split's near-duplicate check
   passed at 0.0%. With ~21 effective independent observations, two clean splits simply
   draw scenes of different difficulty. Test drew the stable, well-lit cone footage.
   **Report the conservative number (~66 / ~90), or the range.**

2. **`floatingsludge` is the weak class**, and this holds on *both* splits (48.74 / 81.27
   vs sludge's 83.47 / 95.07). Unlike the mAP wobble this is reproducible signal, so it is
   the highest-value thing to work on. Open question: label inconsistency in where the
   floating layer starts, or genuine difficulty.

3. **Capacity is not the bottleneck.** The model peaked at epoch 30 and then overfitted.
   A larger model (`v9-m`, 10.5x the deploy cost of `v9-t`) would overfit sooner. Skip it.

4. **Hyperparameter tuning is not yet meaningful.** Val swings ±7 points between epochs and
   ±22 between splits, so any change worth less than ~10 points cannot be measured. Tuning
   against this signal fits noise.

### Confidence threshold (val, 309 frames, 321 objects)

Full sweep: `docs/results/conf_sweep_run1_val.json`

The two classes want **opposite** thresholds, so no single global value serves both:

| | `floatingsludge` | `sludge` |
|---|---|---|
| precision @ 0.05 | 0.945 | 0.689 |
| precision @ 0.70 | 0.937 | 0.969 |
| recall @ 0.05 | 0.895 | 0.879 |
| recall @ 0.70 | 0.517 | 0.852 |
| best F1 | **conf 0.05** | **conf 0.70** |

`floatingsludge` holds ~0.95 precision at every threshold with false positives pinned at
6-9, so raising the bar only costs recall. `sludge` sheds false positives 59 -> 4 as the
bar rises while recall barely moves.

| | global 0.05 | global 0.50 | **per-class 0.05 / 0.70** |
|---|---|---|---|
| precision | 0.807 | 0.937 | **0.956** |
| recall | 0.888 | 0.788 | 0.875 |
| false alarms | 68 | 17 | **13** |
| missed | 36 | 68 | 40 |

Per-class cuts false alarms 81% against the F2-optimal global setting for 4 lost
detections out of 321. Not yet deployable - `rpi_infer.py` takes one `--conf`.

**Recall caps at 0.888 at any threshold**: 36 of 321 objects are undetectable no matter
how low the bar goes. That is a model/data limit, not a tuning one.

**This revises finding 2.** `floatingsludge` looked like the weak class on mAP@50:95
(48.74 vs 83.47), but at its own threshold it is the *stronger* one (F1 0.919 vs 0.907).
mAP@50:95 averages IoU up to 0.95 and so is dominated by box tightness; `floatingsludge`
is a thin horizontal layer where a few pixels of vertical error wrecks IoU while
detection at IoU 0.5 stays reliable. The problem is box tightness, not detection - which
points at label boundary consistency rather than at collecting more examples.

### Next

- Inspect `floatingsludge` labels with `tools/visualize_predictions.py --sample worst` on **val**.
- Sweep the confidence threshold to set the precision/recall operating point deliberately.
- More **scene** diversity — different sites, weather, times of day. This is the real ceiling.
- k-fold CV over scenes only if a defensible single number is needed for a report.

---

## Template

```markdown
## `<name>` — <date>

<one line: what this run was testing>

### Command
<the exact command>

### Environment
hardware / trainer commit / dataset counts / split provenance

### Outcome
stopped how, at which epoch, wall clock, best epoch, checkpoint S3 path

### Results
| metric | val | test |

### Findings
<what you learned, including negative results>

### Next
<what this run implies>
```
