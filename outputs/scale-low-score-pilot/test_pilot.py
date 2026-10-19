import json
from pathlib import Path
import tempfile
import unittest
import pilot


def response(content, finish='stop'):
    return {'choices': [{'finish_reason': finish, 'message': {'content': content}}]}


class Checks(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(pilot.parse_response(response('{"variants":["a","b"]}')), ['a', 'b'])

    def test_reject_bad_output(self):
        for text in ['not JSON', '{"variants":["a"]}', '{"variants":["a",""]}', '{"variants":[1,2]}']:
            with self.assertRaises(ValueError):
                pilot.parse_response(response(text))
        with self.assertRaises(ValueError):
            pilot.parse_response(response('{"variants":["a","b"]}', 'length'))

    def test_no_label_or_reference_in_prompt(self):
        body = pilot.payload({'question': 'q', 'answer': 'a', 'gold_raw': 6, 'references': ['r']}, pilot.config())
        self.assertEqual(json.loads(body['messages'][1]['content']), {'question': 'q', 'student_answer': 'a'})

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'file.json'
            pilot.write_once(path, {'a': 1})
            pilot.write_once(path, {'a': 1})
            with self.assertRaises(ValueError):
                pilot.write_once(path, {'a': 2})

    def test_summary_flags_without_assigning_gold(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            rows = [{'sample_id': 'x', 'question_id': 'q', 'question': 'question',
                     'answer': 'answer', 'gold_raw': 6, 'references': ['partial']}]
            pilot.write_once(run / 'response_01.json', response('{"variants":["answer","answer"]}'))
            pilot.summarize(run, rows, rows)
            candidates = pilot.read_rows(run / 'candidates.jsonl')
            self.assertEqual(len(candidates), 2)
            self.assertIsNone(candidates[0]['candidate_gold_raw'])
            self.assertIn('duplicate_of_training_answer', candidates[0]['flags'])
            self.assertIn('duplicate_of_candidate', candidates[1]['flags'])
            self.assertEqual(json.loads((run / 'summary.json').read_text())['accepted_for_training'], 0)


if __name__ == '__main__':
    unittest.main()
