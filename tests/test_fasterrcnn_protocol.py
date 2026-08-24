import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from analysis.fasterrcnn_prediction_adapter import adapt_fasterrcnn_image
from second_detector.data import read_dataset_yaml, split_paths
from second_detector.metrics import encode_tp_masks, match_predictions
from second_detector.protocol import build_jobs, locked_config
from scripts.train_fasterrcnn import parse_args as parse_train_args


class FasterRCNNProtocolTest(unittest.TestCase):
    def test_training_cli_maps_device_to_engine_keyword(self) -> None:
        argv = [
            "train_fasterrcnn.py",
            "--data-yaml",
            "/datasets/dataset.yaml",
            "--output-dir",
            "/generation/run",
            "--variant",
            "baseline",
            "--seed",
            "42",
            "--device",
            "cuda:1",
        ]
        with patch("sys.argv", argv):
            args = parse_train_args()
        self.assertEqual(args.device_name, "cuda:1")
        self.assertFalse(hasattr(args, "device"))

    def test_matrix_has_exactly_nine_jobs(self) -> None:
        jobs = build_jobs(rebuilt_root=Path("/datasets"), results_root=Path("/generation"))
        self.assertEqual(len(jobs), 9)
        self.assertEqual({job.variant for job in jobs}, {"baseline", "a1_crop", "a2_colorjitter"})
        self.assertEqual({job.seed for job in jobs}, {42, 43, 44})
        self.assertIn("seed42/A1_defect_preserving_crop", str(jobs[3].data_yaml))

    def test_locked_settings(self) -> None:
        config = locked_config()
        self.assertEqual(config["architecture"], "fasterrcnn_mobilenet_v3_large_fpn")
        self.assertEqual(config["image_size"], 1024)
        self.assertEqual(config["trainable_backbone_layers"], 3)
        self.assertEqual(config["epochs"], 50)
        self.assertEqual(config["best_metric"], "mAP50_95")
        self.assertEqual(config["score_threshold"], 0.001)
        self.assertEqual(config["nms_threshold"], 0.7)
        self.assertEqual(config["max_detections"], 300)
        self.assertEqual(config["online_augmentation"], "none")

    def test_ultralytics_greedy_matching_semantics(self) -> None:
        iou = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        correct = match_predictions(np.asarray([0, 0]), np.asarray([0, 0]), iou)
        self.assertEqual(encode_tp_masks(correct), [1023, 1023])

    def test_torchvision_labels_are_mapped_back_to_zero_based(self) -> None:
        image = adapt_fasterrcnn_image(
            image="example.jpg",
            canonical_id="example.jpg",
            image_path="/tmp/example.jpg",
            width=100,
            height=100,
            gt_boxes_xyxy=np.asarray([[10, 10, 20, 20]], dtype=np.float32),
            gt_class_ids=np.asarray([0]),
            pred_boxes_xyxy=np.asarray([[10, 10, 20, 20]], dtype=np.float32),
            pred_scores=np.asarray([0.9]),
            pred_class_ids=np.asarray([1]),
            class_names=["live_knot", "dead_knot"],
        )
        self.assertEqual(image["predictions"][0]["class_id"], 0)
        self.assertEqual(image["predictions"][0]["validator_tp_mask"], 1023)

    def test_dataset_yaml_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images" / "val").mkdir(parents=True)
            (root / "labels" / "val").mkdir(parents=True)
            yaml_path = root / "dataset.yaml"
            yaml_path.write_text(
                "\n".join(
                    [
                        f"path: {root}",
                        "train: images/train",
                        "val: images/val",
                        "test: images/test",
                        "names:",
                        "  0: live_knot",
                        "  1: dead_knot",
                    ]
                ),
                encoding="utf-8",
            )
            payload = read_dataset_yaml(yaml_path)
            image_root, label_root, names = split_paths(yaml_path, "val")
            self.assertEqual(payload["names"], ["live_knot", "dead_knot"])
            self.assertEqual(image_root, (root / "images" / "val").resolve())
            self.assertEqual(label_root, (root / "labels" / "val").resolve())
            self.assertEqual(names, ["live_knot", "dead_knot"])


if __name__ == "__main__":
    unittest.main()
