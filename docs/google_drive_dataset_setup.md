# Google Drive Dataset Setup For Vast.ai

Datasets must not be committed to GitHub. Keep raw images, YOLO folders, archives, checkpoints, and generated variants under `/workspace/data` or another external data volume.

Expected server layout:

```text
/workspace/data/
  main_dataset/benchmarks/vsb7_3600_rare_first_yolo/dataset.yaml
  vnwoodknot/benchmarks/vnwoodknot_live_dead_2class_yolo/dataset.yaml
  wood_defect_datacentric/generated_yolo/
```

Do not hard-code private Google Drive URLs into repo files. Put private links or remote names in `.env` on the server only.

## Option A: `gdown`

Install:

```bash
pip install gdown
mkdir -p /workspace/data/downloads
```

For a shared file:

```bash
gdown "<GOOGLE_DRIVE_FILE_URL_OR_ID>" -O /workspace/data/downloads/dataset.zip
```

For a shared folder:

```bash
gdown --folder "<GOOGLE_DRIVE_FOLDER_URL>" -O /workspace/data/downloads
```

Then unpack into the expected layout:

```bash
mkdir -p /workspace/data
unzip /workspace/data/downloads/dataset.zip -d /workspace/data
```

Verify that `dataset.yaml` exists for each YOLO dataset before training.

## Option B: `rclone`

Install or use an image that already includes `rclone`:

```bash
curl https://rclone.org/install.sh | sudo bash
rclone config
```

Configure a Google Drive remote, for example `gdrive:`. Then copy:

```bash
mkdir -p /workspace/data
rclone copy gdrive:/path/to/vsb7_3600_rare_first_yolo \
  /workspace/data/main_dataset/benchmarks/vsb7_3600_rare_first_yolo \
  --progress

rclone copy gdrive:/path/to/vnwoodknot_live_dead_2class_yolo \
  /workspace/data/vnwoodknot/benchmarks/vnwoodknot_live_dead_2class_yolo \
  --progress
```

Use `rclone check` when possible:

```bash
rclone check gdrive:/path/to/vsb7_3600_rare_first_yolo \
  /workspace/data/main_dataset/benchmarks/vsb7_3600_rare_first_yolo
```

## Option C: `rsync` Or `scp`

If the datasets are already on a local workstation:

```bash
rsync -avh --progress /local/path/vsb7_3600_rare_first_yolo/ \
  root@<VAST_HOST>:/workspace/data/main_dataset/benchmarks/vsb7_3600_rare_first_yolo/

rsync -avh --progress /local/path/vnwoodknot_live_dead_2class_yolo/ \
  root@<VAST_HOST>:/workspace/data/vnwoodknot/benchmarks/vnwoodknot_live_dead_2class_yolo/
```

For a compressed archive:

```bash
scp dataset_bundle.tar.gz root@<VAST_HOST>:/workspace/data/downloads/
ssh root@<VAST_HOST>
tar -xzf /workspace/data/downloads/dataset_bundle.tar.gz -C /workspace/data
```

## Required Verification

After copying data:

```bash
cd /workspace/wood_defect_datacentric
PYTHONDONTWRITEBYTECODE=1 python scripts/check_server_ready.py
```

Training should not start until this check passes.
