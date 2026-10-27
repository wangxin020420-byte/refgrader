import tempfile
import unittest
from pathlib import Path

from scripts.label_free_input_inventory import build_inventory


def create_fixture(root, newline):
    files = {
        "exam_database.json": b"[\n]\n",
        "answer_metadata.jsonl": b"{}\n",
        "rubrics/initial/Q/r.json": b"[]\n",
        "splits/by_question/Q.json": b"{}\n",
        "student_images/Q/a.png": b"student image bytes",
        "reference_images/q.png": b"reference image bytes",
    }
    for name, data in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data.replace(b"\n", newline) if path.suffix in (".json", ".jsonl") else data)


class InventoryTests(unittest.TestCase):
    def test_normalizes_text_but_detects_image_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            first, second = Path(tmp) / "a", Path(tmp) / "b"
            create_fixture(first, b"\n")
            create_fixture(second, b"\r\n")
            result = build_inventory(first)
            self.assertEqual(result, build_inventory(second))
            self.assertFalse(result["image_annotation_audited"])
            (second / "student_images/Q/a.png").write_bytes(b"changed")
            self.assertNotEqual(result["groups"]["student_images"], build_inventory(second)["groups"]["student_images"])

    def test_never_requires_teacher_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_fixture(root, b"\n")
            expected = build_inventory(root)
            (root / "teacher_scores.json").write_text("not even JSON", encoding="utf-8")
            self.assertEqual(expected, build_inventory(root))

    def test_rejects_missing_images_and_lfs_pointers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_fixture(root, b"\n")
            image = root / "student_images/Q/a.png"
            image.write_bytes(b"version https://git-lfs.github.com/spec/v1\n")
            with self.assertRaisesRegex(ValueError, "LFS pointer"):
                build_inventory(root)
            image.unlink()
            with self.assertRaisesRegex(ValueError, "No input files"):
                build_inventory(root)


if __name__ == "__main__":
    unittest.main()
