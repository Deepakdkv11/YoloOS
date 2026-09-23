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
