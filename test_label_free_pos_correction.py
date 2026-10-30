import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from label_free_pos_correction import apply_candidate, fit_pos


def pos(answer, base, blind, item='s1', cap=5):
    return {'question_id': 'Q1', 'answer_id': answer, 'route': 'POS',
            'items': [{'item_id': item, 'max_score': cap, 'base_score': base, 'blind_score': blind}]}


def target(answer, route='BND', base=3, cap=5, item='s1'):
    return {'question_id': 'Q1', 'answer_id': answer, 'route': route,
            'items': [{'item_id': item, 'max_score': cap, 'base_score': base}]}


class PosCorrectionTests(unittest.TestCase):
    def test_fit_shrinkage_then_bnd_deferred(self):
        fit = fit_pos([pos('A1', 3, 4), pos('A2', 2, 3)], min_count=2, shrinkage=2, cap=0.5)
        self.assertAlmostEqual(fit['groups'][0]['delta'], 0.1)
        out = apply_candidate(fit, target('B1'))
        self.assertAlmostEqual(out['candidate_total'], 3.5)
        self.assertTrue(out['requires_human_review'])
        self.assertIsNone(out['automatic_score'])
        self.assertFalse(out['score_change_authorized'])
        self.assertFalse(out['bnd_arbitration_applied'])

    def test_neg_remains_review_and_bounds_and_unseen_fallback(self):
        fit = fit_pos([pos('A1', 0, 5)], min_count=1, shrinkage=0, cap=0.5)
        out = apply_candidate(fit, target('B1', route='NEG', base=4.8))
        self.assertEqual(out['candidate_total'], 5)
        self.assertTrue(out['requires_human_review'])
        self.assertEqual(out['original_route'], 'NEG')
        missing = apply_candidate(fit, dict(target('B2'), question_id='Q2'))
        self.assertEqual(missing['candidate_total'], 3)
        self.assertFalse(missing['items'][0]['correction_available'])
        with self.assertRaisesRegex(ValueError, 'rubric items'):
            apply_candidate(fit, target('B3', item='unseen'))

    def test_small_groups_and_zero_cap_fallback(self):
        record = pos('A1', 1, 2)
        record['items'].append({'item_id': 'zero', 'max_score': 0, 'base_score': 0, 'blind_score': 0})
        fit = fit_pos([record],
                      min_count=2, shrinkage=1, cap=0.2)
        self.assertTrue(all(not group['usable'] and group['delta'] == 0 for group in fit['groups']))

    def test_inconsistent_question_rubrics_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Inconsistent question rubric'):
            fit_pos([pos('A1', 1, 2), pos('A2', 0, 0, item='other')],
                    min_count=1, shrinkage=1, cap=0.2)

    def test_reject_labels_nonpos_overlap_and_invalid_numbers(self):
        record = pos('A1', 3, 4)
        for bad in [dict(record, teacher_score=5), dict(record, route='NEG')]:
            with self.assertRaises(ValueError):
                fit_pos([bad], min_count=1, shrinkage=1, cap=0.1)
        with self.assertRaises(ValueError):
            fit_pos([record, record], min_count=1, shrinkage=1, cap=0.1)
        with self.assertRaises(ValueError):
            fit_pos([pos('A2', 3, float('nan'))], min_count=1, shrinkage=1, cap=0.1)
        fit = fit_pos([record], min_count=1, shrinkage=1, cap=0.1)
        with self.assertRaisesRegex(ValueError, 'fit sample'):
            apply_candidate(fit, target('A1', route='POS'))
        with self.assertRaises(ValueError):
            apply_candidate(fit, target('A3', cap=10))

    def test_cli_new_files_and_tamper_detection(self):
        runner = Path(__file__).parent / 'scripts' / 'run_pos_correction_prototype.py'
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            inp, tar, fit, out = [root / name for name in ('in.json', 'target.json', 'fit.json', 'out.json')]
            inp.write_text(json.dumps({'records': [pos('A1', 3, 4)]}), encoding='utf-8')
            tar.write_text(json.dumps({'records': [target('B1')]}), encoding='utf-8')
            command = [sys.executable, str(runner), 'fit', '--input', str(inp), '--output', str(fit),
                       '--min-count', '1', '--shrinkage', '1', '--cap', '0.1']
            self.assertEqual(subprocess.run(command, capture_output=True).returncode, 0)
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            apply = [sys.executable, str(runner), 'apply', '--fit', str(fit), '--input', str(tar), '--output', str(out)]
            self.assertEqual(subprocess.run(apply, capture_output=True).returncode, 0)
            self.assertTrue(json.loads(out.read_text(encoding='utf-8'))['results'][0]['candidate_only'])
            envelope = json.loads(fit.read_text(encoding='utf-8'))
            envelope['fit']['groups'][0]['delta'] = 0.9
            fit.write_text(json.dumps(envelope), encoding='utf-8')
            apply[-1] = str(root / 'tampered.json')
            self.assertNotEqual(subprocess.run(apply, capture_output=True).returncode, 0)


if __name__ == '__main__':
    unittest.main()
