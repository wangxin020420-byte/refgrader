import json
from pathlib import Path
import tempfile
import unittest
from scripts.assisted_label_results import review_bucket, load_assisted_results
from scripts.review_existing_teacher_evidence import digest, record_reviews, publish, VERSION


class ExistingEvidenceTests(unittest.TestCase):
    def test_human_decision_takes_precedence(self):
        row = {'suspicion_level': 'low'}
        self.assertEqual(review_bucket(row, {'decision': 'retained_hard_case'}, {'category': 'teacher_suspect'}), 'human_reviewed')
        self.assertEqual(review_bucket(row, None, {'category': 'teacher_suspect'}), 'teacher_suspect')
        self.assertEqual(review_bucket(row, None, {'category': 'uncertain'}), 'needs_review')
        self.assertEqual(review_bucket(row, None, {'category': 'label_supported'}), 'auto_reviewed')

    def test_only_individually_recorded_evidence_counts(self):
        import hashlib
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / 'student.jpg'
            image.write_bytes(b'image fixture')
            task = {'question_id':'Q', 'answer_id':'A', 'input_sha256':'frozen', 'teacher_score':4,
                'images': {'original': {'path': str(image), 'sha256':hashlib.sha256(image.read_bytes()).hexdigest()}},
                'model_evidence': {'source_path':'old_checkpoint.json'}}
            manifest = {'version':VERSION,'review_run_id':'run','target_count':1}
            (root/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
            self.assertEqual(publish(root,manifest,[task])['completed'],0)
            review = {'question_id':'Q','answer_id':'A','category':'label_supported','summary':'已有方法分可成立',
                'evidence':['原式正确，末步算错，教师已扣分'], 'original_image_observations':'公式与最终值已核对'}
            self.assertEqual(record_reviews(root,manifest,[task],[review])['completed'],1)
            rows,summary,_ = load_assisted_results(root)
            self.assertEqual(summary['completed'],1)
            self.assertFalse(rows[('Q','A')]['new_experiment'])
            with self.assertRaises(ValueError):
                record_reviews(root,manifest,[task],[review])
            with self.assertRaises(ValueError):
                record_reviews(root,manifest,[task],[{**review,'answer_id':'OUTSIDE'}])


if __name__ == '__main__':
    unittest.main()
