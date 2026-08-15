# Single-Generation Vast Runbook

This runbook creates one isolated, fully provenanced generation for the IEEE Access
resubmission. It does not reuse the 18 deprecated A1/A2/P4+A4 checkpoints for primary
results; it stages them only in a separate inference audit. Commands marked **Mac** run
locally; commands marked **Vast** run on the rented server.

## 0. Fixed scope and naming

The generation contains 42 primary checkpoint families in one directory:

- 24 new runs: VN P1/P3/A1/A2/P4+A4 and VSB A1/A2/P4+A4, seeds 42/43/44.
- 18 surviving archived runs: VN Baseline/P2 and VSB Baseline/P1/P2/P3, seeds 42/43/44.

It also contains a physically separate audit namespace with the 18 deprecated
A1/A2/P4+A4 checkpoints selected using augmented validation data. These checkpoints
are inference-only, carry the role `DEPRECATED_augmented_validation_selection` in
provenance, and are never read by primary tables or the 42-checkpoint registry.

Use one immutable tag throughout:

```bash
export GENERATION_TAG=access_r1_g1
```

Never point any command in this document at `results/`, `results 2/`, or `results 3/`.

## 1. Minimal data set and transfer cost

Measured logical sizes from `revised/datasets_rebuilt/` are below. An `eval` suffix
means only `val` and `test` images/labels plus YAML/report files are required.

| Tree | GiB | Needed for | Transfer or regenerate? |
|---|---:|---|---|
| VN canonical | 0.934 | A1/A2 source; raw val/test | Required |
| VN P1 full | 0.899 | P1 training/evaluation | Required |
| VN P3 full | 0.799 | P3 training/evaluation | Required |
| VN P2 eval | 0.245 | surviving P2 val/test | Required |
| VN P4 eval | 0.268 | P4+A4 val/test | Required |
| VN A1 seeds 42-44 | 2.491 | corrected training | Required |
| VN A2 seeds 42-44 | 2.509 | corrected training | Required |
| VN P4+A4 seeds 42-44 | 2.726 | corrected training | Required |
| VSB canonical | 7.621 | A1/A2 source; raw val/test | Required |
| VSB A1 seeds 42-44 | 10.446 | corrected training | Required |
| VSB A2 seeds 42-44 | 12.386 | corrected training | Required |
| VSB P4+A4 seeds 42-44 | 9.397 | corrected training | Required |
| VSB P1 eval | 0.754 | surviving P1 val/test | Required |
| VSB P2 eval | 0.681 | surviving P2 val/test | Required |
| VSB P3 eval | 0.709 | surviving P3 val/test | Required |
| VSB P4 eval | 0.746 | P4+A4 val/test | Required |
| VSB strict-clean canonical | 17.511 | 5,976-tile clean evaluation | Required |
| VSB strict-clean P1 | 2.230 | clean P1 inference | Required |
| VSB strict-clean P2 | 1.801 | clean P2 inference | Required |
| VSB strict-clean P3 | 1.913 | clean P3 inference | Required |
| VSB strict-clean P4 | 2.183 | clean P4+A4 inference | Required |

The minimal prebuilt selection is **79.25 GiB** if hardlinks are expanded and
**65.26 GiB** when transferred in one `rsync -aH` file list. Preserving hardlinks
saves **13.99 GiB**. Do not transfer A-variant trees with separate independent
commands unless duplication is acceptable.

The source-only alternative is about **51.7 GiB**:

| Source | Approximate size |
|---|---:|
| VN raw plus archived canonical val/test pixels | 0.94 GiB |
| `revised/data/main_dataset/images` | 35 GiB |
| `revised/data/clean_data` | 16 GiB |
| authoritative manifests/configs in git | under 0.03 GiB |

The source-only total is therefore about **52.0 GiB**. The complete derived tree is
about 78 GiB. Keeping source and derived data together
therefore uses about 130 GiB before checkpoints and caches.

### Ranked transfer choices

1. **Recommended: Google Drive source pull plus server materialization (Options B+C).**
   The connected Drive was checked: `workspace_20260624/data` contains the
   `main_dataset` and `vnwoodknot` source folders. No `datasets_rebuilt` folder was
   found. A metadata search also did not find a folder named `clean_data`. That 16 GiB
   folder is not needed for training, so upload it from the Mac in parallel with the
   24-job queue rather than delaying STOP 2.
   At 300-800 Mbit/s, the Drive portion should take about 10-35 minutes; allow 20-60
   minutes for Drive overhead. The 16 GiB Mac upload adds roughly 1.8/0.7/0.4 hours at
   20/50/100 Mbit/s before overhead and can run while the Drive pull is active. The
   observed Mac materialization time was 52 minutes; allow 25-60 minutes on a 16+ vCPU
   server.
2. **Fallback: Mac-to-Vast rsync of the 65.26 GiB minimal prebuilt selection.** At a
   20/50/100 Mbit/s home upload this is about 7.3/2.9/1.5 hours before small-file and
   network overhead. Use `-aH --partial` and keep all selected paths in one invocation.
3. **Do not pull a rebuilt tree from Drive unless it is uploaded and verified first.**
   Its presence has not been established. `gdown` is also a poor fit for hundreds of
   thousands of files; use `rclone` for Drive folders.

### Prebuilt fallback: exact minimal rsync

If source rematerialization is not practical, generate one file list on the Mac and
transfer it in a single hardlink-preserving operation. This selects full training trees
only where needed and only val/test from preprocessing trees used for evaluation:

```bash
cd /Users/ntkhanh/PycharmProjects/wood_defect_datacentric
python scripts/build_minimal_transfer_manifest.py \
  --rebuilt-root revised/datasets_rebuilt \
  --output-list revised/minimal_transfer_files.txt \
  --output-summary revised/minimal_transfer_summary.csv

rsync -aH --partial --info=progress2 \
  --files-from=revised/minimal_transfer_files.txt \
  revised/datasets_rebuilt/ \
  <VAST_SSH>:/workspace/data/datasets_rebuilt/
```

On Vast, skip Sections 4.4-4.6, run the YAML relocation command in Appendix A, then run
the 84/84 training-data gate from Section 4.7. The strict-clean source split in Section
7.2 is still mandatory even when its canonical/preprocessed pixels arrive prebuilt.
The manifest generator prints both logical and physical size with hardlinks preserved;
compare those values with Section 1 before starting the transfer.

## 2. Freeze code and stage surviving checkpoints

### 2.1 Mac: commit and record the exact code revision

Review all changes before committing; do not commit dataset pixels or old results.

```bash
cd /Users/ntkhanh/PycharmProjects/wood_defect_datacentric
git status --short
git diff -- RUNBOOK.md analysis/vsb_negative_aware.py scripts configs tests
git add RUNBOOK.md analysis/vsb_negative_aware.py scripts tests \
  configs/experiments/vn_yolov8s_p1_clahe_e50.yaml \
  configs/experiments/vn_yolov8s_p3_unsharp_e50.yaml
git commit -m "Add single-generation training and provenance workflow"
git push origin main
git rev-parse HEAD
```

Record the printed 40-character SHA as `PINNED_COMMIT`.

### 2.2 Mac: stage only the 18 surviving checkpoints

`results 2/` is used because all 18 selected `best.pt` files were checked and have
normal sizes of 22,600,739-22,604,771 bytes. The known truncated file is in another
snapshot and is not selected.

```bash
cd /Users/ntkhanh/PycharmProjects/wood_defect_datacentric
export BUNDLE=/tmp/${GENERATION_TAG}_survivors
python scripts/generation_checkpoint_registry.py \
  --generation-root "$BUNDLE" \
  --survivor-root "results 2" \
  --stage-survivors \
  --mode copy \
  --allow-missing
```

Expected pre-training result: `18/42`; this is not an error at this stage.

### 2.3 Mac: stage the 18 deprecated checkpoints in an audit-only namespace

```bash
python scripts/stage_deprecated_checkpoints.py \
  --source-root "results 2" \
  --generation-root "$BUNDLE" \
  --mode copy
```

Expected: `DEPRECATED CHECKPOINT REGISTRY: PASS (18/18)`. The files are stored below
`$BUNDLE/deprecated_checkpoints/`, never below the primary `multiseed/` tree.

### 2.4 Mac: test Google Drive authorization before renting Vast

Run this now, using the same `gdrive` remote that Vast will use. It performs a real
download of the file that previously returned `appNotAuthorizedToFile`, then checks it
against the restored local copy:

```bash
cd /Users/ntkhanh/PycharmProjects/wood_defect_datacentric
rclone version
rclone config reconnect gdrive:

export RCLONE_SMOKE="$(mktemp -d)"
rclone copyto \
  "gdrive:2.Work/1.PTIT/1.Ca_nhan/2.Research/2026/workspace_20260624/data/vnwoodknot/images/test/knot_free/IMG_4832.jpg" \
  "$RCLONE_SMOKE/IMG_4832.jpg" -P
test -s "$RCLONE_SMOKE/IMG_4832.jpg"
cmp "$RCLONE_SMOKE/IMG_4832.jpg" \
  revised/data/vnwoodknot/images/test/knot_free/IMG_4832.jpg

rclone lsf \
  "gdrive:2.Work/1.PTIT/1.Ca_nhan/2.Research/2026/workspace_20260624/data/vnwoodknot/benchmarks/vnwoodknot_live_dead_2class_yolo/images" \
  --dirs-only
```

Expected: `cmp` exits 0 and the last command lists `val/` and `test/`. If either fails,
repair the OAuth grant before renting; a metadata listing alone is not sufficient.

## 3. Rent and configure Vast

Choose an **on-demand**, not interruptible, offer with:

- exactly 2 x RTX 3090, 24 GB each;
- at least 16 vCPU and 64 GB RAM;
- 300 GB local NVMe recommended, 250 GB minimum;
- host reliability at least 0.98 and strong measured upload/download bandwidth;
- NVIDIA driver 550.54 or newer, and both GPUs visible to the container.

The training queue is resumable, but an interruptible instance can lose the local
source, 78 GiB derived tree, and unreturned checkpoints. The small spot discount is not
worth that provenance risk.

### 3.1 Vast: clone the exact revision

```bash
cd /workspace
git clone https://github.com/khanhnt/wood_defect_datacentric.git
cd /workspace/wood_defect_datacentric
git fetch --all --tags
git checkout <PASTE_PINNED_COMMIT_SHA_HERE>
test "$(git rev-parse HEAD)" = "<PASTE_PINNED_COMMIT_SHA_HERE>"
```

### 3.2 Vast: create the pinned Python 3.12 environment

Use the template's Python 3.12 when available:

```bash
python3.12 -V
python3.12 -m venv /workspace/wood_env
source /workspace/wood_env/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.6.0 torchvision==0.21.0
python -m pip install -r requirements.txt
python -m pip check
```

If `python3.12` is absent, use Miniforge rather than silently changing Python:

```bash
cd /workspace
curl -L -o Miniforge3.sh \
  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3.sh -b -p /workspace/miniforge3
source /workspace/miniforge3/bin/activate
conda create -y -p /workspace/wood_env python=3.12 pip
conda activate /workspace/wood_env
cd /workspace/wood_defect_datacentric
python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.6.0 torchvision==0.21.0
python -m pip install -r requirements.txt
python -m pip check
```

Create the isolated generation root and verify every package and GPU:

```bash
export GENERATION_TAG=access_r1_g1
export GEN=/workspace/generations/$GENERATION_TAG
export DATA=/workspace/data/datasets_rebuilt
mkdir -p "$GEN/provenance" /workspace/source

nvidia-smi
python scripts/verify_generation_runtime.py \
  --expected-gpus 2 \
  --output "$GEN/provenance/runtime_preflight.json"
```

**STOP 1:** continue only if runtime preflight is `PASS`, reports Python 3.12,
Ultralytics 8.4.60, torch 2.6.0 with CUDA available, and two RTX 3090 GPUs.

## 4. Transfer source data and rebuild datasets on Vast

### 4.1 Vast: authorize and test Google Drive

Install or update rclone and reconnect the remote interactively:

```bash
curl https://rclone.org/install.sh | bash
rclone config
# Create or reconnect a Drive remote named gdrive with full read access.
rclone config reconnect gdrive:
```

Test one of the files that previously produced `appNotAuthorizedToFile`:

```bash
mkdir -p /tmp/rclone_smoke
rclone copyto \
  "gdrive:2.Work/1.PTIT/1.Ca_nhan/2.Research/2026/workspace_20260624/data/vnwoodknot/images/test/knot_free/IMG_4832.jpg" \
  /tmp/rclone_smoke/IMG_4832.jpg -P
test -s /tmp/rclone_smoke/IMG_4832.jpg
```

If this still returns HTTP 403, stop and fix OAuth/app access. Do not start a 35 GiB
copy that is already known to be unauthorized.

Pull the Drive-confirmed source folders. VN is intentionally assembled from two
folders: its raw `images` folder contains all training images plus one restored test
image, while the archived canonical `images` folder contains the 226 validation and
remaining 228 test images. Their union is the verified 1,515-image manifest source.

```bash
mkdir -p /workspace/source/vnwoodknot /workspace/source/main_dataset
rclone copy \
  "gdrive:2.Work/1.PTIT/1.Ca_nhan/2.Research/2026/workspace_20260624/data/vnwoodknot/images" \
  /workspace/source/vnwoodknot/images \
  --transfers 16 --checkers 32 --drive-chunk-size 128M --fast-list -P
rclone copy \
  "gdrive:2.Work/1.PTIT/1.Ca_nhan/2.Research/2026/workspace_20260624/data/vnwoodknot/benchmarks/vnwoodknot_live_dead_2class_yolo/images" \
  /workspace/source/vnwoodknot/images \
  --transfers 16 --checkers 32 --drive-chunk-size 128M --fast-list -P
rclone copy \
  "gdrive:2.Work/1.PTIT/1.Ca_nhan/2.Research/2026/workspace_20260624/data/main_dataset/images" \
  /workspace/source/main_dataset/images \
  --transfers 16 --checkers 32 --drive-chunk-size 128M --fast-list -P
```

### 4.2 Mac: transfer checkpoint bundles and pretrained weights

Replace `<VAST_SSH>` with the configured ssh host/alias:

```bash
cd /Users/ntkhanh/PycharmProjects/wood_defect_datacentric
rsync -avh --partial --info=progress2 \
  "$BUNDLE/" \
  <VAST_SSH>:/workspace/generations/$GENERATION_TAG/
rsync -avh --partial \
  revised/wood_defect_datacentric/yolov8s.pt \
  <VAST_SSH>:/workspace/wood_defect_datacentric/yolov8s.pt
```

### 4.3 Vast: source-count preflight

```bash
cd /workspace/wood_defect_datacentric
source /workspace/wood_env/bin/activate 2>/dev/null || conda activate /workspace/wood_env

find /workspace/source/vnwoodknot/images -type f | wc -l
find /workspace/source/main_dataset/images -type f | wc -l
df -h /workspace
```

The VN count must be exactly 1,515. The larger VSB reservoir need not contain all
48,156 historical tiles, but the materializer must resolve all 9,628 rare-first
manifest rows. Keep at least 150 GiB free before rebuilding.
Verify that every new run starts from the same archived pretrained file:

```bash
echo "1f47a78bf100391c2a140b7ac73a1caae18c32779be7d310658112f7ac9aa78a  yolov8s.pt" | sha256sum -c -
```

### 4.4 Vast: canonical materialization

All writes are under the new `$DATA` tree.

```bash
mkdir -p "$DATA/canonical" "$DATA/variants" "$DATA/reports"

PYTHONDONTWRITEBYTECODE=1 python scripts/materialize_yolo_from_manifest.py \
  --manifest data/vnwoodknot_split/manifest.jsonl \
  --images-root /workspace/source/vnwoodknot/images \
  --output-root "$DATA/canonical/vnwoodknot" \
  --dataset-name vnwoodknot \
  --classes live_knot dead_knot \
  --split-strategy manifest \
  --link-mode hardlink \
  --exclude-image-id train/2/img_3671

PYTHONDONTWRITEBYTECODE=1 python scripts/materialize_yolo_from_manifest.py \
  --manifest data/vsb_rarefirst_split/manifest.jsonl \
  --images-root /workspace/source/main_dataset/images \
  --output-root "$DATA/canonical/vsb_rarefirst" \
  --dataset-name vsb_rarefirst \
  --classes live_knot dead_knot resin knot_with_crack crack marrow knot_missing \
  --split-strategy manifest \
  --link-mode hardlink

```

### 4.5 Vast: all-split preprocessing materialization

```bash
for dataset in vnwoodknot vsb_rarefirst; do
  for variant in \
    P1_CLAHE_luminance \
    P2_illumination_normalization \
    P3_mild_unsharp \
    P4_combined_safe; do
    PYTHONDONTWRITEBYTECODE=1 python scripts/materialize_preprocessed_yolo.py \
      --source-yaml "$DATA/canonical/$dataset/dataset.yaml" \
      --variant-config "configs/preprocessing/$variant.yaml" \
      --output-root "$DATA/variants/$dataset/preprocessing/$variant" \
      --image-format jpg \
      --jpg-quality 95 \
      --splits train val test
  done
done
```

### 4.6 Vast: train-only A variants

```bash
for dataset in vnwoodknot vsb_rarefirst; do
  for seed in 42 43 44; do
    for variant in A1_defect_preserving_crop A2_texture_aware_color_jitter; do
      PYTHONDONTWRITEBYTECODE=1 python scripts/materialize_augmented_yolo.py \
        --source-yaml "$DATA/canonical/$dataset/dataset.yaml" \
        --variant-config "configs/augmentation/$variant.yaml" \
        --output-root "$DATA/variants/$dataset/augmentation/seed$seed/$variant" \
        --seed "$seed" \
        --image-format jpg \
        --jpg-quality 95 \
        --splits train \
        --passthrough-mode hardlink
    done

    PYTHONDONTWRITEBYTECODE=1 python scripts/materialize_augmented_yolo.py \
      --source-yaml "$DATA/variants/$dataset/preprocessing/P4_combined_safe/dataset.yaml" \
      --variant-config configs/augmentation/A4_combined_best.yaml \
      --output-root "$DATA/variants/$dataset/augmentation/seed$seed/P4_combined_safe__A4_combined_best" \
      --seed "$seed" \
      --image-format jpg \
      --jpg-quality 95 \
      --splits train \
      --passthrough-mode hardlink
  done
done
```

### 4.7 Vast: mandatory training-data gate

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/verify_rebuilt_datasets.py \
  --root "$DATA" \
  --datasets vnwoodknot vsb_rarefirst \
  --output-csv "$DATA/reports/verification_gate.csv" \
  --output-md "$DATA/reports/verification_gate.md"
```

**STOP 2:** do not train unless the final line is exactly
`VERIFICATION GATE: PASS (84/84)`. Also inspect that the VN materialization report
lists `train/2/img_3671` under explicit exclusions. Strict-clean is deliberately absent
from this gate because its source upload runs concurrently with training; it receives a
separate 15/15 gate before clean inference.

## 5. Smoke test

Run one VN A1 seed for two epochs in a disposable output root:

```bash
export SMOKE=/workspace/generations/${GENERATION_TAG}_smoke
PYTHONDONTWRITEBYTECODE=1 python scripts/run_all_experiments.py \
  --job-set corrected24 \
  --dataset vnwoodknot \
  --variants a1_crop \
  --seeds 42 \
  --batch-size 40 \
  --epochs 2 \
  --imgsz 1024 \
  --workers 4 \
  --gpus 0 \
  --rebuilt-root "$DATA" \
  --vn-yaml "$DATA/canonical/vnwoodknot/dataset.yaml" \
  --vsb-yaml "$DATA/canonical/vsb_rarefirst/dataset.yaml" \
  --results-root "$SMOKE"

test -s "$SMOKE/multiseed/vnwoodknot/per_seed/runs/a1_crop_seed42/ultralytics/train/weights/best.pt"
test -s "$SMOKE/multiseed/vnwoodknot/per_seed/runs/a1_crop_seed42/ultralytics/train/weights/last.pt"
```

**STOP 3:** confirm nonzero labels, GPU use below 22 GiB, `best.pt`, `last.pt`, and an
`ok` run-log row. The two-epoch smoke metrics are not paper results.

## 6. Run the corrected 24-job queue

### 6.1 Vast: final dry run

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/run_all_experiments.py \
  --job-set corrected24 \
  --dataset all \
  --batch-size 40 \
  --epochs 50 \
  --imgsz 1024 \
  --workers 4 \
  --gpus 0,1 \
  --rebuilt-root "$DATA" \
  --vn-yaml "$DATA/canonical/vnwoodknot/dataset.yaml" \
  --vsb-yaml "$DATA/canonical/vsb_rarefirst/dataset.yaml" \
  --results-root "$GEN" \
  --dry-run
```

The dry run must print exactly 24 jobs: 15 VN and 9 VSB.

### 6.2 Vast: launch under nohup

```bash
nohup env PYTHONDONTWRITEBYTECODE=1 \
  /workspace/wood_env/bin/python scripts/run_all_experiments.py \
  --job-set corrected24 \
  --dataset all \
  --batch-size 40 \
  --epochs 50 \
  --imgsz 1024 \
  --workers 4 \
  --gpus 0,1 \
  --rebuilt-root "$DATA" \
  --vn-yaml "$DATA/canonical/vnwoodknot/dataset.yaml" \
  --vsb-yaml "$DATA/canonical/vsb_rarefirst/dataset.yaml" \
  --results-root "$GEN" \
  > "$GEN/launcher.log" 2>&1 &
echo $! | tee "$GEN/launcher.pid"
```

If using Conda instead of the venv, replace `/workspace/wood_env/bin/python` with the
path printed by `which python`.

### 6.3 Mac: upload strict-clean sources while training runs

This transfer is independent of training and must not block STOP 2. Start it from a
second Mac terminal immediately after the 24-job queue is running:

```bash
cd /Users/ntkhanh/PycharmProjects/wood_defect_datacentric
export CLEAN_UPLOAD_LOG="/tmp/${GENERATION_TAG}_clean_upload.log"
nohup rsync -avh --partial --info=progress2 \
  revised/data/clean_data/ \
  <VAST_SSH>:/workspace/source/clean_data/ \
  > "$CLEAN_UPLOAD_LOG" 2>&1 &
echo $!
tail -f "$CLEAN_UPLOAD_LOG"
```

The upload is complete only when Vast contains exactly 1,992 readable source BMPs.
Training needs none of these files; Section 7 waits for them only before strict-clean
materialization and inference.

Monitor without attaching to an interactive terminal:

```bash
tail -f "$GEN/launcher.log"
watch -n 5 nvidia-smi
python scripts/generation_status.py --generation-root "$GEN" --gpus 2
```

### 6.4 Vast: resume after interruption

Run the exact launch command again with `--resume` before the redirection. Completed
jobs are skipped; incomplete jobs with `last.pt` resume from it; jobs that died before
the first checkpoint restart.

```bash
nohup env PYTHONDONTWRITEBYTECODE=1 \
  /workspace/wood_env/bin/python scripts/run_all_experiments.py \
  --job-set corrected24 --dataset all --batch-size 40 --epochs 50 --imgsz 1024 \
  --workers 4 --gpus 0,1 --rebuilt-root "$DATA" \
  --vn-yaml "$DATA/canonical/vnwoodknot/dataset.yaml" \
  --vsb-yaml "$DATA/canonical/vsb_rarefirst/dataset.yaml" \
  --results-root "$GEN" --resume \
  >> "$GEN/launcher.log" 2>&1 &
```

### 6.5 Vast: checkpoint gate

```bash
python scripts/generation_checkpoint_registry.py \
  --generation-root "$GEN" \
  --output-csv "$GEN/provenance/checkpoint_registry.csv"
```

**STOP 4:** require `CHECKPOINT REGISTRY: PASS (42/42)`. The registry requires both
`best.pt` and `last.pt` for all 24 new runs and validates all 18 survivor `best.pt`
files. This is the first point at which training is complete.

## 7. Complete strict-clean data and run fair evaluation

### 7.1 Vast: materialize and verify the VSB strict-clean data

Do not run this step until the parallel Mac upload reports completion:

```bash
find /workspace/source/clean_data -type f -iname '*.bmp' | wc -l
```

The count must be exactly 1,992. Build the canonical 5,976-tile tree, preserving the
actual per-tile origins and overlap values:

```bash
python analysis/vsb_negative_aware.py \
  --clean-set-only \
  --clean-images-root /workspace/source/clean_data \
  --clean-ids-file configs/datasets/vsb_clean_source_ids.txt \
  --clean-output-root "$DATA/canonical/vsb_strict_clean" \
  --rare-first-manifest data/vsb_rarefirst_split/manifest.jsonl \
  --rare-first-yaml "$DATA/canonical/vsb_rarefirst/dataset.yaml" \
  --output-dir "$DATA/reports/vsb_strict_clean" \
  --clean-tile-size 1024 \
  --clean-tile-overlap 128 \
  --link-mode hardlink \
  --overwrite-clean-set

for variant in \
  P1_CLAHE_luminance \
  P2_illumination_normalization \
  P3_mild_unsharp \
  P4_combined_safe; do
  python scripts/materialize_preprocessed_yolo.py \
    --source-yaml "$DATA/canonical/vsb_strict_clean/dataset.yaml" \
    --variant-config "configs/preprocessing/$variant.yaml" \
    --output-root "$DATA/variants/vsb_strict_clean/preprocessing/$variant" \
    --image-format jpg \
    --jpg-quality 95 \
    --splits test
done

python scripts/verify_rebuilt_datasets.py \
  --root "$DATA" \
  --datasets vsb_strict_clean \
  --output-csv "$DATA/reports/verification_gate_strict_clean.csv" \
  --output-md "$DATA/reports/verification_gate_strict_clean.md"
```

Require `VERIFICATION GATE: PASS (15/15)` before clean inference.

### 7.2 Vast: split VSB clean sources for threshold selection and final test

Use a deterministic 50/50 split by source ID, never by tile. With seed 42, exactly
996 source images (2,988 tiles) form the threshold-selection `val` half and the other
996 sources (2,988 tiles) form the untouched final `test` half. All three tiles from
one source remain together. A 50/50 ratio gives equal statistical support to threshold
selection and final reporting while preventing source-level leakage.

```bash
python scripts/split_vsb_clean_sources.py \
  --rebuilt-root "$DATA" \
  --output-root "$DATA/eval_views/vsb_strict_clean" \
  --seed 42 \
  --selection-sources 996 \
  --mode hardlink \
  --overwrite

cat "$DATA/eval_views/vsb_strict_clean/partition_report.json"
```

Require `VSB CLEAN SOURCE SPLIT: PASS`, `source_overlap=0`, and 2,988 images in each
half for every variant. The source and tile assignments are frozen in
`source_partition_manifest.csv` and `tile_partition_manifest.csv`.

### 7.3 Vast: freeze the explicit evaluation maps

```bash
python scripts/build_generation_eval_map.py \
  --rebuilt-root "$DATA" \
  --output-dir "$GEN/eval_maps"

cp "$DATA/eval_views/vsb_strict_clean/eval_dataset_map.csv" \
  "$GEN/eval_maps/vsb_strict_clean_source_disjoint_map.csv"
```

### 7.4 Vast: evaluate all 42 primary checkpoints on val and test

```bash
python scripts/evaluate_corrected_common.py \
  --results-root "$GEN" \
  --output-csv "$GEN/fair_eval/fair_metrics.csv" \
  --eval-map-csv "$GEN/eval_maps/fair_eval_dataset_map.csv" \
  --dataset all \
  --splits val test \
  --seeds 42 43 44 \
  --imgsz 1024 \
  --batch 32 \
  --conf 0.001 \
  --iou 0.7 \
  --device 0
```

Expected: 84 metric rows, each carrying its exact `data_yaml`. A1/A2 use canonical
val/test; P variants use their full-image preprocessing tree; P4+A4 uses P4-only
val/test without A4.

### 7.5 Vast: fair evaluation of the 18 deprecated checkpoints

These results quantify the effect of selecting checkpoints with augmented validation.
They are never merged into the primary table.

```bash
python scripts/evaluate_corrected_common.py \
  --results-root "$GEN/deprecated_checkpoints" \
  --output-csv "$GEN/deprecated_audit/fair_eval/fair_metrics.csv" \
  --eval-map-csv "$GEN/eval_maps/fair_eval_dataset_map.csv" \
  --dataset all \
  --variants a1_crop a2_colorjitter p4_a4_combined \
  --splits val test \
  --seeds 42 43 44 \
  --imgsz 1024 --batch 32 --conf 0.001 --iou 0.7 --device 0

python scripts/compare_deprecated_checkpoints.py \
  --corrected "$GEN/fair_eval/fair_metrics.csv" \
  --deprecated "$GEN/deprecated_audit/fair_eval/fair_metrics.csv" \
  --output-dir "$GEN/deprecated_audit/comparison"
```

Expected: 36 deprecated metric rows plus per-seed and mean/std deltas. Every row and
checkpoint is labelled `DEPRECATED_augmented_validation_selection` in provenance.

## 8. Low-confidence prediction exports

The exporter now accepts `--split`; every command below sets `augment=False`,
`conf=0.001`, `iou=0.7`, and `imgsz=1024`.

Set reusable paths:

```bash
export VN_RUNS="$GEN/multiseed/vnwoodknot/per_seed/runs"
export VSB_RUNS="$GEN/multiseed/vsb_rarefirst/per_seed/runs"
export VARIANTS="baseline p1_clahe p2_illumination p3_unsharp a1_crop a2_colorjitter p4_a4_combined"
```

### 8.1 VN validation and test

```bash
for split in val test; do
  python scripts/threshold_sweep_inference.py \
    --dataset-name vnwoodknot \
    --split "$split" \
    --checkpoint-root "$VN_RUNS" \
    --output-dir "$GEN/predictions/vnwoodknot/$split" \
    --gpus 0,1 \
    --variants $VARIANTS \
    --seeds 42 43 44 \
    --conf 0.001 --iou 0.7 --imgsz 1024 --batch 32 --max-det 300 \
    --variant-data-yaml "baseline=$DATA/canonical/vnwoodknot/dataset.yaml" \
    --variant-data-yaml "p1_clahe=$DATA/variants/vnwoodknot/preprocessing/P1_CLAHE_luminance/dataset.yaml" \
    --variant-data-yaml "p2_illumination=$DATA/variants/vnwoodknot/preprocessing/P2_illumination_normalization/dataset.yaml" \
    --variant-data-yaml "p3_unsharp=$DATA/variants/vnwoodknot/preprocessing/P3_mild_unsharp/dataset.yaml" \
    --variant-data-yaml "a1_crop=$DATA/canonical/vnwoodknot/dataset.yaml" \
    --variant-data-yaml "a2_colorjitter=$DATA/canonical/vnwoodknot/dataset.yaml" \
    --variant-data-yaml "p4_a4_combined=$DATA/variants/vnwoodknot/preprocessing/P4_combined_safe/dataset.yaml"
done
```

### 8.2 VSB rare-first validation and test

```bash
for split in val test; do
  python scripts/threshold_sweep_inference.py \
    --dataset-name vsb_rarefirst --split "$split" \
    --checkpoint-root "$VSB_RUNS" \
    --output-dir "$GEN/predictions/vsb_rarefirst/$split" \
    --gpus 0,1 --variants $VARIANTS --seeds 42 43 44 \
    --conf 0.001 --iou 0.7 --imgsz 1024 --batch 32 --max-det 300 \
    --variant-data-yaml "baseline=$DATA/canonical/vsb_rarefirst/dataset.yaml" \
    --variant-data-yaml "p1_clahe=$DATA/variants/vsb_rarefirst/preprocessing/P1_CLAHE_luminance/dataset.yaml" \
    --variant-data-yaml "p2_illumination=$DATA/variants/vsb_rarefirst/preprocessing/P2_illumination_normalization/dataset.yaml" \
    --variant-data-yaml "p3_unsharp=$DATA/variants/vsb_rarefirst/preprocessing/P3_mild_unsharp/dataset.yaml" \
    --variant-data-yaml "a1_crop=$DATA/canonical/vsb_rarefirst/dataset.yaml" \
    --variant-data-yaml "a2_colorjitter=$DATA/canonical/vsb_rarefirst/dataset.yaml" \
    --variant-data-yaml "p4_a4_combined=$DATA/variants/vsb_rarefirst/preprocessing/P4_combined_safe/dataset.yaml"
done
```

### 8.3 VSB source-disjoint strict-clean validation and test

```bash
export CLEAN_MAP_ROOT="$DATA/eval_views/vsb_strict_clean"
for split in val test; do
  python scripts/threshold_sweep_inference.py \
    --dataset-name vsb_strict_clean --split "$split" \
    --checkpoint-root "$VSB_RUNS" \
    --output-dir "$GEN/predictions/vsb_strict_clean/$split" \
    --gpus 0,1 --variants $VARIANTS --seeds 42 43 44 \
    --conf 0.001 --iou 0.7 --imgsz 1024 --batch 32 --max-det 300 \
    --variant-data-yaml "baseline=$CLEAN_MAP_ROOT/baseline/dataset.yaml" \
    --variant-data-yaml "p1_clahe=$CLEAN_MAP_ROOT/p1_clahe/dataset.yaml" \
    --variant-data-yaml "p2_illumination=$CLEAN_MAP_ROOT/p2_illumination/dataset.yaml" \
    --variant-data-yaml "p3_unsharp=$CLEAN_MAP_ROOT/p3_unsharp/dataset.yaml" \
    --variant-data-yaml "a1_crop=$CLEAN_MAP_ROOT/a1_crop/dataset.yaml" \
    --variant-data-yaml "a2_colorjitter=$CLEAN_MAP_ROOT/a2_colorjitter/dataset.yaml" \
    --variant-data-yaml "p4_a4_combined=$CLEAN_MAP_ROOT/p4_a4_combined/dataset.yaml"
done
```

For VSB, select each operating threshold only from rare-first `val` plus the 996-source
strict-clean `val` half. Freeze that threshold, then report retained metrics on
rare-first `test` plus the source-disjoint strict-clean `test` half. Never inspect the
clean test half while selecting a threshold.

### 8.4 Deprecated-checkpoint prediction exports

Export all 18 deprecated checkpoints on the same non-augmented positive val/test data.
These 36 JSON files support audit and diagnosis only:

```bash
export DEPRECATED_ROOT="$GEN/deprecated_checkpoints/multiseed"
for dataset in vnwoodknot vsb_rarefirst; do
  if [ "$dataset" = vnwoodknot ]; then
    RUNS="$DEPRECATED_ROOT/vnwoodknot/per_seed/runs"
    RAW="$DATA/canonical/vnwoodknot/dataset.yaml"
    P4="$DATA/variants/vnwoodknot/preprocessing/P4_combined_safe/dataset.yaml"
  else
    RUNS="$DEPRECATED_ROOT/vsb_rarefirst/per_seed/runs"
    RAW="$DATA/canonical/vsb_rarefirst/dataset.yaml"
    P4="$DATA/variants/vsb_rarefirst/preprocessing/P4_combined_safe/dataset.yaml"
  fi
  for split in val test; do
    python scripts/threshold_sweep_inference.py \
      --dataset-name "$dataset" --split "$split" \
      --checkpoint-root "$RUNS" \
      --output-dir "$GEN/deprecated_audit/predictions/$dataset/$split" \
      --gpus 0,1 \
      --variants a1_crop a2_colorjitter p4_a4_combined \
      --seeds 42 43 44 \
      --conf 0.001 --iou 0.7 --imgsz 1024 --batch 32 --max-det 300 \
      --variant-data-yaml "a1_crop=$RAW" \
      --variant-data-yaml "a2_colorjitter=$RAW" \
      --variant-data-yaml "p4_a4_combined=$P4"
  done
done
```

Expected primary predictions: 42 VN, 42 VSB rare-first, and 42 VSB strict-clean,
for 126 JSON exports. Expected deprecated predictions: 36 JSON exports in the separate
audit namespace.

## 9. Reproduction and provenance gates

### 9.1 Vast: diagnose saved-prediction AP against fair evaluation

```bash
python scripts/verify_prediction_map_reproduction.py \
  --predictions-root "$GEN/predictions" \
  --fair-summary "$GEN/fair_eval/fair_metrics.csv" \
  --checkpoint-registry "$GEN/provenance/checkpoint_registry.csv" \
  --output-csv "$GEN/fair_eval/prediction_ap_reproduction.csv" \
  --diagnostics-csv "$GEN/fair_eval/prediction_ap_matching_diagnostics.csv" \
  --exact-tolerance 0.002 \
  --review-tolerance 0.005

python scripts/verify_prediction_map_reproduction.py \
  --predictions-root "$GEN/deprecated_audit/predictions" \
  --fair-summary "$GEN/deprecated_audit/fair_eval/fair_metrics.csv" \
  --checkpoint-registry "$GEN/deprecated_checkpoints/deprecated_checkpoint_registry.csv" \
  --output-csv "$GEN/deprecated_audit/fair_eval/prediction_ap_reproduction.csv" \
  --diagnostics-csv "$GEN/deprecated_audit/fair_eval/prediction_ap_matching_diagnostics.csv" \
  --exact-tolerance 0.002 \
  --review-tolerance 0.005
```

This is a diagnostic gate, not a reason to leave paid GPUs idle. The script verifies
checkpoint hash, dataset-YAML hash, and image count first, then compares an
Ultralytics-style IoU-priority matcher with the confidence-ordered greedy matcher used
by the offline analysis.

- `abs residual <= 0.002`: `EXACT_PASS`, accepted as direct reproduction.
- `0.002 < abs residual <= 0.005`: `METHOD_REVIEW`, acceptable only when all provenance
  hashes/counts match and the per-image diagnostic attributes the difference to matching
  or numerical/AP interpolation convention. Report it as method sensitivity, not exact
  equality.
- `abs residual > 0.005`, or any hash/count mismatch: `INVESTIGATE`; not acceptable for
  a paper table until resolved.

Regardless of status, finish provenance, checksums, and rsync before releasing the Vast
instance. Use `--strict` only in a later offline CI gate, not during the paid run.

### 9.2 Vast: write per-artifact provenance

```bash
python scripts/write_generation_provenance.py \
  --generation-root "$GEN" \
  --fair-summary "$GEN/fair_eval/fair_metrics.csv" \
  --checkpoint-registry "$GEN/provenance/checkpoint_registry.csv" \
  --prediction-root "$GEN/predictions" \
  --deprecated-checkpoint-registry "$GEN/deprecated_checkpoints/deprecated_checkpoint_registry.csv" \
  --deprecated-fair-summary "$GEN/deprecated_audit/fair_eval/fair_metrics.csv" \
  --deprecated-prediction-root "$GEN/deprecated_audit/predictions" \
  --pretrained-weights yolov8s.pt \
  --vn-manifest data/vnwoodknot_split/manifest.jsonl \
  --vsb-manifest data/vsb_rarefirst_split/manifest.jsonl \
  --vsb-clean-manifest "$DATA/eval_views/vsb_strict_clean/source_partition_manifest.csv" \
  --extra-manifest data/vsb_clean_manifest/clean_tile_manifest.csv \
  --extra-manifest "$DATA/canonical/vsb_strict_clean/clean_materialized_samples.csv" \
  --extra-manifest "$DATA/eval_views/vsb_strict_clean/tile_partition_manifest.csv" \
  --extra-manifest "$DATA/eval_views/vsb_strict_clean/partition_report.json"
```

This records checkpoint, manifest, evaluation-YAML, and prediction SHA-256 values,
copies every evaluation YAML into the generation, and writes a portable
`provenance/SHA256SUMS`. The 18 audit checkpoints and their outputs are explicitly
labelled DEPRECATED. VN and VSB operating thresholds must be selected from their
validation exports and applied unchanged to test; test must not select its own
threshold.

## 10. Return results to the Mac

### 10.1 Mac: copy the complete generation

```bash
cd /Users/ntkhanh/PycharmProjects/wood_defect_datacentric
mkdir -p "revised/generations/$GENERATION_TAG"
rsync -avzh --partial --info=progress2 \
  <VAST_SSH>:/workspace/generations/$GENERATION_TAG/ \
  "revised/generations/$GENERATION_TAG/"
```

Bring back the complete generation, including new `best.pt`/`last.pt`, the 18
survivors, run metadata, fair metrics, all prediction JSONs, provenance, logs, and
the reproduction CSV. Do not bring back `/workspace/source`, `$DATA`, Ultralytics
caches, or the smoke directory; the corresponding pixels already exist locally.

### 10.2 Mac: verify before destroying Vast

```bash
cd "/Users/ntkhanh/PycharmProjects/wood_defect_datacentric/revised/generations/$GENERATION_TAG"
shasum -a 256 -c provenance/SHA256SUMS

find multiseed -path '*/weights/best.pt' -size +10M | wc -l
find multiseed -path '*/weights/last.pt' -size +10M | wc -l
find predictions -name '*_predictions.json' | wc -l
find deprecated_checkpoints -path '*/weights/best.pt' -size +10M | wc -l
find deprecated_audit/predictions -name '*_predictions.json' | wc -l
wc -l fair_eval/fair_metrics.csv
wc -l deprecated_audit/fair_eval/fair_metrics.csv
wc -l fair_eval/prediction_ap_reproduction.csv
wc -l deprecated_audit/fair_eval/prediction_ap_reproduction.csv
```

Expected counts:

- 42 `best.pt` files;
- 24 `last.pt` files (surviving archived runs did not retain `last.pt`);
- 18 separately registered deprecated `best.pt` files;
- 126 primary prediction JSON files and 36 deprecated audit prediction JSON files;
- 85 lines in `fair_metrics.csv` (header + 84 rows);
- 37 lines in the deprecated fair-metrics CSV (header + 36 rows);
- 85 and 37 lines in the primary and deprecated reproduction CSVs;
- all checksum lines report `OK`; any non-`EXACT_PASS` reproduction row has a recorded
  diagnosis and follows the tolerance policy in Section 9.1.

Archive a second copy of the complete returned generation before destroying the Vast
instance. Only after both local checks and the backup pass should the instance be
deleted.

## 11. Time, cost, and likely failure mode

Times below come from the final cumulative `time` field in the 36 archived batch-40
run logs, not from a generic estimate.

| Work | Runs | GPU-hours | Two-GPU wall estimate |
|---|---:|---:|---:|
| VN A1/A2/P4+A4 retraining | 9 | 2.002 | about 1.0 h |
| New VN P1/P3 | 6 | 1.344 | about 0.7 h |
| VSB A1/A2/P4+A4 retraining | 9 | 12.340 | about 6.2 h |
| **Training total** | **24** | **15.686** | **about 7.8 h** |

The deprecated audit adds no training. Its 36 fair-evaluation jobs and 36 low-confidence
exports are expected to add roughly 0.8-1.5 GPU-hours, or about 25-50 minutes with two
RTX 3090s. The comparison script is CPU-only and should finish in seconds.

Allow 11-13 rented hours for source transfer, 25-60 minutes of materialization, smoke,
training, primary and deprecated fair evaluation, 162 prediction exports, hashing, and
one retry margin. The 16 GiB clean-data upload runs in parallel with the 7.8-hour
training window and normally adds no critical-path time.

Vast prices are marketplace-dependent rather than fixed. A practical planning range
for two on-demand RTX 3090s is about USD 0.26-0.60 per instance-hour before storage,
giving roughly USD 3-8 for an 11-13 hour session; confirm the actual offer total in the
Vast UI. Pricing references: [Vast pricing documentation](https://docs.vast.ai/guides/instances/pricing)
and [Vast live pricing guide](https://vast.ai/article/how-much-does-it-cost-to-rent-a-gpu-in-the-cloud-live-pricing-guide).

The most likely failure is an incomplete or unauthorized data transfer that leaves a
plausible-looking but wrong dataset tree. The mitigation is deliberately strict:
rclone single-file smoke, source counts, new output-only materialization, the 84/84
training-data gate plus a later 15/15 clean-data gate, source-disjoint clean val/test
views, explicit eval maps, and provenance-aware AP diagnostics before any manuscript
number is accepted.

## Appendix A. Prebuilt rsync fallback

If Drive authorization cannot be repaired, transfer the prebuilt data with hardlinks
preserved. The safest simple fallback is the complete rebuilt tree (about 78 GiB actual)
because it can run both verification gates without rematerializing pixels:

```bash
cd /Users/ntkhanh/PycharmProjects/wood_defect_datacentric/revised
rsync -aHh --partial --info=progress2 \
  datasets_rebuilt/ \
  <VAST_SSH>:/workspace/data/datasets_rebuilt/
```

Then relocate the Mac-absolute YAML roots on Vast and run the gate:

```bash
export DATA=/workspace/data/datasets_rebuilt
python scripts/relocate_dataset_yamls.py --root "$DATA"
python scripts/verify_rebuilt_datasets.py \
  --root "$DATA" \
  --output-csv "$DATA/reports/verification_gate_server.csv" \
  --output-md "$DATA/reports/verification_gate_server.md"
```

Expect `VERIFICATION GATE: PASS (99/99)`, then continue at Section 7.2 to create the
source-disjoint 996/996 clean evaluation views.

Do not use plain `scp -r`: it expands hardlinks and loses resume/checksum behavior.
