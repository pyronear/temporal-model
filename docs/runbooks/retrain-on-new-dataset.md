# Runbook — retrain the temporal model on a new pyro-dataset release

How to retrain, evaluate, and release the temporal smoke classifier when a new
[pyro-dataset](https://github.com/pyronear/pyro-dataset) version ships. Every
step below was executed for the v3.0.0 → v4.1.0 retrain (PR #65); numbers in
the examples are that run's real output.

> **CI path (preferred):** steps 1–4 are automated by
> [`.github/workflows/train.yml`](../../.github/workflows/train.yml). Push a
> branch named `train_vX.Y.Z` (the new pyro-dataset release tag) and CI spins
> up a GPU runner, bumps the dataset pointers to that tag, retrains, refreshes
> eval, pushes the DVC data, and opens a `result_*` PR with the old-vs-new
> pyro-annotator table (baseline = `results/` on main). A `train_*` branch
> whose suffix is not a version tag trains on the pointers as committed
> (code-change retrain). The manual loop below remains the reference for
> running locally and for understanding what CI does.

**Prerequisites**

- A GPU machine with this repo checked out and `make install` run (`train/` and
  `eval/` both need their `uv sync`).
- AWS credentials that can read pyro-dataset's DVC remote and read/write this
  repo's DVC remotes (`AWS_PROFILE=pyronear` below).
- Disk: the raw v4.1.0 train+val import is ~9 GB (8.2 train + 0.8 val), and
  the pipeline roughly triples that downstream (truncated copies, patches,
  checkpoints).

**The loop at a glance**

```
train/  dvc update  → dvc repro (truncate → tubes → patches → train → package)
eval/   refresh datasets + model.zip → dvc repro (train / val / pyro-annotator)
        old-vs-new metrics comparison
push    dvc push (train + eval) → commit pointers + dvc.lock → PR
release bump api/MODEL_VERSION → publish model.zip to HF → git tag
```

## 0. Snapshot the current model's metrics (before touching anything)

The eval reporting tree is overwritten in place, and **every** reporting
output is `cache: false` (metrics, results, `details/`, `sequences/` — none of
it is recoverable from the DVC remote). Snapshot the whole tree, to a location
that survives a reboot: it is the "old" column of your comparison table, and
the per-sequence artifacts are what let you ask later *which* sequences
flipped, not just how the totals moved:

```bash
cd eval
cp -r data/08_reporting ~/baselines/temporal-model-08_reporting-$(date +%F)
```

If a source's reporting tree is missing locally (it happens — `cache: false`
outputs never round-trip through the remote), regenerate it with the **old**
model before updating anything:

```bash
uv run dvc repro evaluate@train evaluate@val evaluate_pyro_annotator
```

(note the `@` names — `evaluate` is a foreach stage; pull
`data/01_raw/pyro-annotator.dvc` first if needed). Verify what you snapshotted
is canonical: the `metrics.json` md5s must match the ones recorded in
`eval/dvc.lock`.

## 1. Point the raw-data imports at the new dataset release

`train/data/01_raw/datasets_full/{train,val}.dvc` are frozen `dvc import`s of
pyro-dataset's `data/processed/sequential_train_val/{train,val}`. Bumping the
dataset is one command (it downloads from pyro-dataset's DVC remote, so the
AWS profile matters):

```bash
cd train
AWS_PROFILE=pyronear uv run dvc update \
  data/01_raw/datasets_full/train.dvc \
  data/01_raw/datasets_full/val.dvc \
  --rev vX.Y.Z          # the new pyro-dataset release tag
```

Beware: `dvc update` against a tag the pointers already record is a silent
no-op — if you forget to change the `--rev`, everything downstream "succeeds"
on the old data.

This rewrites both `.dvc` pointers (`rev: v4.1.0` + the new content hashes) and
materialises the new data. Sanity-check the sequence counts against the numbers
published in the dataset release PR before spending GPU-hours:

```bash
for split in train val; do for kind in wildfire fp; do
  echo "$split/$kind: $(ls data/01_raw/datasets_full/$split/$kind | wc -l)"
done; done
```

For v4.1.0 this printed exactly the release PR's numbers (50% FP per split by
construction):

```
train/wildfire: 1664
train/fp: 1664
val/wildfire: 175
val/fp: 175
```

and the updated pointers recorded `rev: v4.1.0`, train 8.2 GB / 137,690 files,
val 814 MB / 13,096 files. If your counts disagree with the dataset release PR,
stop here — a partial pull retrains on a corrupt split.

## 2. Retrain: `dvc repro`

```bash
cd train
uv run dvc repro
```

Stages, in order (all downstream of the import, so everything reruns):

1. `truncate` — caps sequences to `truncate.max_frames` (20) frames.
2. `build_tubes` — greedy-IoU tube linking from the label boxes.
3. `build_model_input` — 224×224 stabilized patches per tube.
4. `train` — the ViT-DINOv2 finetune (PyTorch Lightning, early stopping);
   writes `best_checkpoint.pt` + training curves.
5. `package` — fits the logistic calibrator on the train sequences, picks the
   decision threshold on val, bundles `model.zip` (checkpoint + detector
   weights + config + calibrator).

The v4.1.0 run, on an RTX 4070 Ti Super (~30 min for the whole `dvc repro`):

```
build_tubes    [train] wrote 2968/3328 tubes (smoke=1527, fp=1441, dropped=360)
               [val]   wrote 300/350 tubes  (smoke=154,  fp=146,  dropped=50)
train          8/30 epochs (~70 s each), early-stopped; best = epoch 2
               (val/f1 0.9585, precision 0.943, recall 0.974 at tube level)
package        val patches scored: n=300 threshold=0.6682
               fitting logistic calibrator on 3328 records...
               wrote model.zip | variant=vit_dinov2_finetune aggregation=logistic
```

Two things worth knowing while you watch it:

- Checkpointing and early stopping both monitor `val/f1`, and an exact F1
  *tie* does not reset patience (see issue #64) — a short run with an early
  best epoch is expected behavior, not a crash.
- Training is bitwise-reproducible on the same GPU (seeded, deterministic
  mode), so a rerun from the same inputs gives the same checkpoint.

Tube-level metrics are **not** the acceptance gate — the packaged model's
sequence-level protocol eval (next step) is.

## 3. Refresh eval and re-score

`eval/` has two inputs that do **not** update themselves:

- `eval/data/01_raw/datasets/{train,val}` — a plain copy of train's
  *truncated* datasets (`train/data/01_raw/datasets/`). Not DVC-tracked in
  eval; refresh it by hand:

  ```bash
  cd eval
  rsync -a --delete ../train/data/01_raw/datasets/ data/01_raw/datasets/
  ```

- `eval/data/06_models/vit_dinov2_finetune/model.zip` — a frozen local import
  of train's packaged model:

  ```bash
  cd eval
  make update-model   # runs: dvc update data/06_models/vit_dinov2_finetune/model.zip.dvc
  ```

  **Forgetting this step is silent.** The import is frozen and `dvc status`
  does not check frozen-stage dependencies, so eval would happily re-score the
  *old* model on all three sources — your "comparison" would be the old model
  against itself. After updating, confirm the two files are identical:

  ```bash
  md5sum data/06_models/vit_dinov2_finetune/model.zip \
         ../train/data/06_models/vit_dinov2_finetune/model.zip
  ```

Then re-score everything (train/val splits + the fixed pyro-annotator store):

```bash
uv run dvc repro
```

The pyro-annotator store is the one testbed that does *not* change with the
dataset release, so its old-vs-new delta is the cleanest read on whether the
new model actually improved.

Compare each source's fresh `metrics.json` against the snapshots from step 0.
Judge the two families of numbers differently:

- **train / val**: the *dataset itself changed*, so old-vs-new is not
  apples-to-apples — read these only as "did calibration hold?" (recall should
  sit at `package.target_recall`).
- **pyro-annotator**: fixed testbed, the honest comparison. For the v4.1.0
  retrain: false alerts 114 → 86 (−25%), precision 0.274 → 0.328, at the cost
  of one extra missed smoke (recall 0.977 → 0.955) and a slower median
  time-to-detect (1 → 3 frames). Watch precision/FPR *and* TTD — a threshold
  that suppresses FPs by waiting longer trades detection latency for it.

## 4. Push data, commit pointers

```bash
cd train && AWS_PROFILE=pyronear uv run dvc push dvc.yaml   # pipeline outputs only
cd ../eval && AWS_PROFILE=pyronear uv run dvc push
```

The `dvc.yaml` target on the train side skips the ~9 GB raw imports: those
blobs already live in pyro-dataset's own DVC remote and are re-fetchable from
the pinned `rev_lock`, so pushing them here only duplicates them into this
repo's bucket and adds an hour-plus of upload per dataset bump (a bare
`dvc push` does exactly that).

Commit (pointers and locks only — blobs never enter git):

- `train/data/01_raw/datasets_full/{train,val}.dvc` (now `rev: v4.1.0`)
- `train/dvc.lock`
- `eval/data/06_models/vit_dinov2_finetune/model.zip.dvc`
- `eval/dvc.lock`

Open the PR in the self-documenting style: every section is the real output of
the commands above (see PR #65 for the template).

## 5. Release the new model (after the PR merges)

A retrain bumps the **model version** (`api/MODEL_VERSION` → the HuggingFace
artifact), which is decoupled from the code version (git tag → Docker image).
Follow **"Cutting a release" in [`docs/releasing.md`](../releasing.md)** — that
is the canonical procedure; don't work from a copy of it here. The
retrain-specific parts:

- the `model.zip` to publish is
  `train/data/06_models/vit_dinov2_finetune/model.zip` — the artifact this
  loop just built, packaged, and evaluated;
- `publish` needs a **write** HF token (the read token in `api/.envrc` 403s at
  upload);
- `publish` is immutable per version — a re-publish of an existing version
  fails by design, so double-check the new version number before running it.
