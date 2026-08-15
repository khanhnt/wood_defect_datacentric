import unittest
import json
from pathlib import Path
import tempfile

from scripts.run_all_experiments import build_jobs, job_completed
from scripts.split_vsb_clean_sources import partition_source_ids
from scripts.stage_deprecated_checkpoints import DATASETS as DEPRECATED_DATASETS
from scripts.stage_deprecated_checkpoints import SEEDS as DEPRECATED_SEEDS
from scripts.stage_deprecated_checkpoints import VARIANTS as DEPRECATED_VARIANTS


class GenerationQueueTest(unittest.TestCase):
    def test_corrected_queue_contains_exactly_24_runs(self) -> None:
        jobs = build_jobs(
            dataset_filter="all",
            job_set="corrected24",
            seeds=(42, 43, 44),
            variants=None,
        )
        self.assertEqual(len(jobs), 24)
        self.assertEqual(sum(job.spec.dataset == "vnwoodknot" for job in jobs), 15)
        self.assertEqual(sum(job.spec.dataset == "vsb_rarefirst" for job in jobs), 9)

    def test_smoke_filter_selects_one_job(self) -> None:
        jobs = build_jobs(
            dataset_filter="vnwoodknot",
            job_set="corrected24",
            seeds=(42,),
            variants={"a1_crop"},
        )
        self.assertEqual([job.job_id for job in jobs], ["vnwoodknot_a1_crop_seed42"])

    def test_corrected_completion_requires_last_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            weights = run_dir / "ultralytics" / "train" / "weights"
            weights.mkdir(parents=True)
            (weights / "best.pt").write_bytes(b"best")
            (run_dir / "validation_metrics.json").write_text(json.dumps({"status": "ok"}), encoding="utf-8")
            self.assertTrue(job_completed(run_dir))
            self.assertFalse(job_completed(run_dir, require_last=True))
            (weights / "last.pt").write_bytes(b"last")
            self.assertTrue(job_completed(run_dir, require_last=True))

    def test_vsb_clean_partition_is_deterministic_and_source_disjoint(self) -> None:
        source_ids = [f"source_{index:04d}" for index in range(1992)]
        first = partition_source_ids(source_ids, seed=42, selection_sources=996)
        second = partition_source_ids(list(reversed(source_ids)), seed=42, selection_sources=996)
        self.assertEqual(first, second)
        selection, final_test, order = first
        self.assertEqual(len(selection), 996)
        self.assertEqual(len(final_test), 996)
        self.assertEqual(len(order), 1992)
        self.assertFalse(selection & final_test)
        self.assertEqual(selection | final_test, set(source_ids))

    def test_deprecated_audit_registry_has_exactly_18_runs(self) -> None:
        self.assertEqual(
            len(DEPRECATED_DATASETS) * len(DEPRECATED_VARIANTS) * len(DEPRECATED_SEEDS),
            18,
        )


if __name__ == "__main__":
    unittest.main()
