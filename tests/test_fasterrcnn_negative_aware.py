import unittest

from analysis.fasterrcnn_negative_aware import select_threshold


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


if __name__ == "__main__":
    unittest.main()
