import importlib.util
from pathlib import Path
import unittest

from run_controls import audit_predictions


class ControlTests(unittest.TestCase):
    def test_out_of_range_directions_and_bias(self):
        labels = [{'sample_id':str(i),'question_id':'q','gold_raw':g}
                  for i,g in enumerate([0,5,10])]
        predictions = [{'sample_id':str(i),'question_id':'q','prediction_raw':v}
                       for i,v in enumerate([-1,5,12])]
        result = audit_predictions(predictions, labels)['overall']
        self.assertEqual(result['below_zero'], 1)
        self.assertEqual(result['above_ten'], 1)
        self.assertAlmostEqual(result['mean_signed_error'], 1/3)
        self.assertEqual(result['out_of_range_ids'], ['0','2'])
        predictions[0]['question_id'] = 'wrong'
        with self.assertRaises(ValueError):
            audit_predictions(predictions, labels)

    def test_original_only_step_count_and_coverage(self):
        path = Path(__file__).resolve().parents[1]/'scale-sae-train/train_sae.py'
        spec = importlib.util.spec_from_file_location('trainer', str(path))
        trainer = importlib.util.module_from_spec(spec); spec.loader.exec_module(trainer)
        rows = [{'kind':'original_training'} for _ in range(885)]
        for epoch in range(10):
            schedule = trainer.batches(rows, epoch, 42)
            self.assertEqual(len(schedule), 56)
            self.assertEqual(sorted(i for group in schedule for i in group), list(range(885)))

    def test_duplicate_predictions_rejected(self):
        labels = [{'sample_id':'a','question_id':'q','gold_raw':0}]
        predictions = [{'sample_id':'a','question_id':'q','prediction_raw':0}]*2
        with self.assertRaises(ValueError):
            audit_predictions(predictions, labels)


if __name__ == '__main__':
    unittest.main()
