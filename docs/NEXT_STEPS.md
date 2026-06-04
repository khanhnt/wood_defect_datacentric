# Next Steps

Current status: the independent IJACSA data-centric repository is prepared for GitHub/Vast.ai server setup. It contains fixed-detector YOLOv8s experiment configs, dataset adapters, preprocessing and augmentation variants, negative-aware VNWoodKnot evaluation, server checks, and controlled batch scripts.

Next:

1. Push this repository to GitHub.
2. Create a Vast.ai RTX 3090 24 GB instance.
3. Clone the repository on Vast.
4. Copy VSB and VNWoodKnot datasets to `/workspace/data`.
5. Configure `.env`.
6. Run `./scripts/setup_server.sh`.
7. Run `PYTHONDONTWRITEBYTECODE=1 python scripts/check_server_ready.py`.
8. Run safe baseline dry-runs with `scripts/server_setup_sanity.py`.
9. Start training with `./scripts/run_batch1_baselines.sh` inside tmux.
10. Aggregate and inspect Batch 1 before continuing.

Do not start full data-centric batches until baselines are verified.
