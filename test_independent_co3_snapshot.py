import json
from pathlib import Path
import tempfile
import unittest

from scripts.build_independent_co3_snapshot import build, SOURCE_HASH
from scripts.verify_label_free_question_snapshot import verify


class IndependentCO3SnapshotTests(unittest.TestCase):
    def test_derived_snapshot_preserves_original_and_explicit_provenance(self):
        source = Path(__file__).parent / "data/label_free_snapshots/csbench_3531732_candidate_v1"
        self.assertEqual(verify(source)["snapshot_sha256_normalized"], SOURCE_HASH)
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "derived"
            report = build(source, dest)
            self.assertEqual(report["questions"], 45)
            self.assertEqual(report["flagged_questions"], [])
            self.assertFalse(report["approved_for_formal_experiment"])
            data = json.loads((dest / "snapshot.json").read_text(encoding="utf-8"))
            provenance = data["derived_reference_provenance"]["CO_3_step_1"]
            self.assertFalse(provenance["teacher_scores_used"])
            self.assertFalse(provenance["official_reference_verified"])
            self.assertFalse(data["source_provenance_audited"])
            with self.assertRaises(FileExistsError):
                build(source, dest)
        self.assertEqual(verify(source)["snapshot_sha256_normalized"], SOURCE_HASH)


if __name__ == "__main__":
    unittest.main()
