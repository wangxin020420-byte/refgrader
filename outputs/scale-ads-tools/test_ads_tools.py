import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import argparse
import json
from ads_tools import metrics, partition, evaluate, lines, write_once


class Checks(unittest.TestCase):
    def test_perfect(self):
        r = metrics([0, 5, 10], [0, 5, 10])
        self.assertEqual(r['qwk'], 1)
        self.assertEqual(r['pearson'], 1)
        self.assertEqual(r['rmse_raw'], 0)

    def test_reversed(self):
        r = metrics([0, 10], [10, 0])
        self.assertEqual(r['qwk'], -1)
        self.assertEqual(r['pearson'], -1)
        self.assertEqual(r['rmse_normalized'], 1)

    def test_constant(self):
        self.assertIsNone(metrics([5, 5], [5, 5])['qwk'])
        self.assertIsNone(metrics([0, 10], [5, 5])['pearson'])
        self.assertEqual(metrics([0, 10], [5, 5])['qwk'], 0)

    def test_invalid(self):
        for pred in ([float('nan')], [float('inf')], [-1], [11], [True]):
            with self.assertRaises(ValueError):
                metrics([0], pred)
        with self.assertRaises(ValueError):
            metrics([0.5], [0.5])
        with self.assertRaises(ValueError):
            metrics([], [])

    def test_group_partition(self):
        rows = [{'row': {'q_id': '1'}, 'content_hash': str(i // 2), 'source_line': i}
                for i in range(20)]
        train, val, _ = partition(rows, 42, 0.2)
        self.assertEqual(len(train) + len(val), 20)
        self.assertTrue(train and val)
        self.assertFalse({r['content_hash'] for r in train} & {r['content_hash'] for r in val})
        self.assertEqual(partition(rows, 42, 0.2)[:2], (train, val))

    def test_ids_and_no_overwrite(self):
        # TemporaryDirectory owns only these synthetic test files.
        with TemporaryDirectory() as temp:
            root = Path(temp)
            labels = root / 'labels.jsonl'
            preds = root / 'predictions.jsonl'
            output = root / 'metrics.json'
            labels.write_bytes(lines([{'sample_id': 'a', 'question_id': 'q', 'gold_raw': 0},
                                      {'sample_id': 'b', 'question_id': 'q', 'gold_raw': 10}]))
            args = argparse.Namespace(labels=labels, predictions=preds, output=output, evaluation_label='TEST_FIXTURE')
            preds.write_bytes(lines([{'sample_id': 'b', 'prediction_raw': 10}, {'sample_id': 'a', 'prediction_raw': 0}]))
            evaluate(args)
            self.assertEqual(json.loads(output.read_text())['mae_raw'], 0)
            before = output.read_bytes()
            with self.assertRaises(ValueError):
                write_once(output, b'changed')
            self.assertEqual(output.read_bytes(), before)
            for rows in ([{'sample_id': 'a', 'prediction_raw': 0}],
                         [{'sample_id': 'a', 'prediction_raw': 0}, {'sample_id': 'a', 'prediction_raw': 0}],
                         [{'sample_id': 'a', 'prediction_raw': 0}, {'sample_id': 'z', 'prediction_raw': 0}]):
                preds.write_bytes(lines(rows))
                with self.assertRaises(ValueError):
                    evaluate(args)


if __name__ == '__main__':
    unittest.main()
