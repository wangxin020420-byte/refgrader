import tempfile
from pathlib import Path
import unittest
import abc_experiment as m


class Checks(unittest.TestCase):
    def fixture(self):
        train = [{'sample_id': 'x', 'question_id': 'q', 'answer': 'original', 'gold_raw': 6,
                  'content_hash': 'old', 'origin': 'original'}]
        candidate = {'candidate_id': '01-1', 'sample_id': 'x', 'question_id': 'q',
                     'original_answer': 'original', 'generated_answer': 'paraphrase',
                     'source_gold_raw': 6, 'flags': [], 'review_status': 'PROVISIONAL_EXPERIMENT_ONLY'}
        return train, candidate

    def test_matched_control(self):
        train, c = self.fixture()
        result = m.groups(train, [c])
        self.assertEqual([len(result[k]) for k in 'ABC'], [1, 2, 2])
        self.assertEqual(result['B'][-1]['answer'], 'original')
        self.assertEqual(result['C'][-1]['answer'], 'paraphrase')
        for k in ['sample_id', 'gold_raw', 'source_sample_id']:
            self.assertEqual(result['B'][-1][k], result['C'][-1][k])
        self.assertNotIn('content_hash', result['C'][-1])

    def test_provenance(self):
        train, c = self.fixture()
        with self.assertRaises(ValueError):
            m.groups(train, [dict(c, source_gold_raw=10)])
        with self.assertRaises(ValueError):
            m.groups(train, [dict(c, review_status='HOLD')])
        with self.assertRaises(ValueError):
            m.groups(train, [c, c])

    def test_schedule_resume(self):
        full = [m.batch_indices(42, s, 899) for s in range(1, 301)]
        resumed = [m.batch_indices(42, s, 899) for s in range(61, 301)]
        self.assertEqual(full[60:], resumed)
        self.assertTrue(all(0 <= i < 899 for batch in full for i in batch))

    def test_selection(self):
        self.assertGreater(m.selection_key({'qwk': -.5, 'rmse_raw': 9}),
                           m.selection_key({'qwk': None, 'rmse_raw': 1}))
        self.assertGreater(m.selection_key({'qwk': .5, 'rmse_raw': 1}),
                           m.selection_key({'qwk': .5, 'rmse_raw': 2}))

    def test_write_guard(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'data'
            m.immutable(p, b'a')
            m.immutable(p, b'a')
            with self.assertRaises(ValueError):
                m.immutable(p, b'b')

    def test_join(self):
        inp = [{'sample_id': 'x', 'question_id': 'q'}]
        lab = [{'sample_id': 'x', 'question_id': 'q', 'gold_raw': 6}]
        self.assertEqual(m.join(inp, lab)[0]['gold_raw'], 6)
        with self.assertRaises(ValueError):
            m.join(inp, [dict(lab[0], sample_id='z')])
        with self.assertRaises(ValueError):
            m.join(inp, lab + lab)


if __name__ == '__main__':
    unittest.main()
