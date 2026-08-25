import unittest

from analysis.fasterrcnn_negative_aware import (
    audit_operational_stability,
    audit_zero_fp_binding,
    select_threshold,
    spearman_rank_correlation,
)


class FasterRCNNNegativeAwareTest(unittest.TestCase):
    def test_threshold_is_selected_from_seed_mean_validation_fp_rate(self) -> None:
        thresholds = [0.05, 0.10, 0.15]
        indexed = {}
        rates = {
            42: [0.04, 0.00, 0.00],
            43: [0.02, 0.00, 0.00],
            44: [0.00, 0.00, 0.00],
        }
        for seed, seed_rates in rates.items():
            for threshold, rate in zip(thresholds, seed_rates):
                indexed[("baseline", seed, threshold)] = ({"fp_image_rate": str(rate)}, None)

        self.assertEqual(select_threshold(thresholds, indexed, "baseline", 0.00), 0.10)
        self.assertEqual(select_threshold(thresholds, indexed, "baseline", 0.02), 0.05)

    def test_binding_audit_counts_seed_image_pairs_at_previous_grid_point(self) -> None:
        val_sets = []
        for variant in ("baseline", "a1_crop", "a2_colorjitter"):
            for seed in (42, 43, 44):
                confidence = 0.0
                if variant == "baseline" and seed == 42:
                    confidence = 0.72
                val_sets.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "images": [
                            {
                                "canonical_id": f"clean_{seed}",
                                "is_knot_free": True,
                                "predictions": [{"conf": confidence}] if confidence else [],
                            }
                        ],
                    }
                )
        selected = [
            {
                "variant": variant,
                "epsilon": "0.00",
                "validation_selected_threshold": "0.75" if variant == "baseline" else "0.05",
            }
            for variant in ("baseline", "a1_crop", "a2_colorjitter")
        ]
        rows = audit_zero_fp_binding(val_sets=val_sets, thresholds=[0.05, 0.70, 0.75], selected_rows=selected)
        baseline = next(row for row in rows if row["variant"] == "baseline")
        self.assertEqual(baseline["binding_seed_image_pairs"], 1)
        self.assertTrue(baseline["single_binding_pair"])

    def test_stability_audit_counts_fp_pairs_and_rank_change(self) -> None:
        binding = [
            {"variant": "baseline", "single_binding_pair": True},
            {"variant": "a1_crop", "single_binding_pair": False},
            {"variant": "a2_colorjitter", "single_binding_pair": False},
        ]
        per_seed = []
        for variant in ("baseline", "a1_crop", "a2_colorjitter"):
            for seed in (42, 43, 44):
                per_seed.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "epsilon": "0.00",
                        "test_clean_FP_images": 1 if (variant, seed) in {("baseline", 42), ("a1_crop", 43)} else 0,
                    }
                )
        summary = [
            {"variant": "baseline", "variant_label": "Baseline", "epsilon": "0.00", "retained_AP50_mean": "0.80"},
            {"variant": "a1_crop", "variant_label": "A1 crop", "epsilon": "0.00", "retained_AP50_mean": "0.60"},
            {"variant": "a2_colorjitter", "variant_label": "A2", "epsilon": "0.00", "retained_AP50_mean": "0.70"},
            {"variant": "baseline", "variant_label": "Baseline", "epsilon": "0.01", "retained_AP50_mean": "0.80"},
            {"variant": "a1_crop", "variant_label": "A1 crop", "epsilon": "0.01", "retained_AP50_mean": "0.60"},
            {"variant": "a2_colorjitter", "variant_label": "A2", "epsilon": "0.01", "retained_AP50_mean": "0.82"},
        ]
        _, report = audit_operational_stability(
            binding_rows=binding,
            per_seed_rows=per_seed,
            summary_rows=summary,
        )
        self.assertEqual(report["single_clean_image_binding"]["count"], 1)
        self.assertEqual(report["test_fp_positive_variant_seed_pairs_at_epsilon0"]["count"], 2)
        self.assertAlmostEqual(report["spearman_epsilon0_vs_epsilon001"]["rho"], 0.5)

    def test_spearman_rank_correlation(self) -> None:
        self.assertAlmostEqual(spearman_rank_correlation([1, 2, 3], [2, 1, 3]), 0.5)


if __name__ == "__main__":
    unittest.main()
