from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.materialize_yolo_from_manifest import build_image_index, resolve_image_path_with_status


class SplitConsistentResolverTest(unittest.TestCase):
    def test_duplicate_stem_across_splits_resolves_within_requested_split(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train" / "dead_knot" / "img_3671.jpg"
            test = root / "test" / "dead_knot" / "img_3671.jpg"
            train.parent.mkdir(parents=True)
            test.parent.mkdir(parents=True)
            train.write_bytes(b"train")
            test.write_bytes(b"test")
            index = build_image_index(root)

            train_path, train_status = resolve_image_path_with_status(
                {"split": "train", "image_path": "images/train/dead_knot/img_3671.jpg"}, index
            )
            test_path, test_status = resolve_image_path_with_status(
                {"split": "test", "image_path": "images/test/dead_knot/img_3671.jpg"}, index
            )

            self.assertEqual((train_path, train_status), (train, "resolved"))
            self.assertEqual((test_path, test_status), (test, "resolved"))

    def test_ambiguous_stem_within_split_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "train" / "class_a" / "duplicate.jpg"
            second = root / "train" / "class_b" / "duplicate.png"
            first.parent.mkdir(parents=True)
            second.parent.mkdir(parents=True)
            first.write_bytes(b"a")
            second.write_bytes(b"b")
            index = build_image_index(root)

            path, status = resolve_image_path_with_status(
                {"split": "train", "image_id": "duplicate"}, index
            )

            self.assertIsNone(path)
            self.assertEqual(status, "ambiguous")

    def test_cross_split_only_match_is_not_substituted(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            test = root / "test" / "dead_knot" / "only_in_test.jpg"
            test.parent.mkdir(parents=True)
            test.write_bytes(b"test")
            index = build_image_index(root)

            path, status = resolve_image_path_with_status(
                {"split": "train", "image_id": "only_in_test"}, index
            )

            self.assertIsNone(path)
            self.assertEqual(status, "unresolved")


if __name__ == "__main__":
    unittest.main()
