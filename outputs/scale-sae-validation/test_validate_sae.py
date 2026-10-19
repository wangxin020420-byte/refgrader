import math
import unittest

from validate_sae import metrics


class MetricTests(unittest.TestCase):
    def test_perfect_predictions(self):
        result = metrics([0, 5, 10], [0, 5, 10])
        self.assertEqual(result['qwk_clipped_half_up'], 1)
        self.assertEqual(result['pearson_raw'], 1)
        self.assertEqual(result['rmse_raw'], 0)

    def test_clipping_not_applied_to_raw_metrics(self):
        result = metrics([0, 5, 10], [-1, 5, 11])
        self.assertAlmostEqual(result['rmse_raw'], math.sqrt(2/3))
        self.assertEqual(result['rmse_clipped'], 0)
        self.assertEqual(result['qwk_clipped_half_up'], 1)
        self.assertEqual(result['out_of_range_predictions'], 2)

    def test_half_up_and_undefined(self):
        result = metrics([0, 1], [0.5, 0.5])
        self.assertEqual(result['qwk_clipped_half_up'], 0)
        self.assertIsNone(result['pearson_raw'])
        self.assertIsNone(metrics([1, 1], [1, 1])['qwk_clipped_half_up'])
        with self.assertRaises(ValueError):
            metrics([0], [float('nan')])
        with self.assertRaises(ValueError):
            metrics([0.5], [0.5])


if __name__ == '__main__':
    unittest.main()
