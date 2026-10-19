import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import train_sae as s


class ProtocolTests(unittest.TestCase):
    def test_schedule_coverage_sizes_and_resume_suffix(self):
        rows = [{'kind': 'original_training'} for _ in range(885)]
        rows += [{'kind': 'synthetic_candidate'} for _ in range(1327)]
        schedule = s.batches(rows, 0, 42)
        self.assertEqual(len(schedule), 77)
        flat = [i for batch in schedule for i in batch]
        self.assertEqual(sorted(flat), list(range(2212)))
        self.assertEqual(len(flat), len(set(flat)))
        for batch in schedule:
            kind = rows[batch[0]]['kind']
            self.assertTrue(all(rows[i]['kind'] == kind for i in batch))
            self.assertLessEqual(len(batch), 16 if kind == 'original_training' else 64)
        self.assertEqual(schedule[20:], s.batches(rows, 0, 42)[20:])
        self.assertNotEqual(schedule, s.batches(rows, 1, 42))

    def test_gpu_chooses_free_device_by_uuid(self):
        output = '0, GPU-a, 22687, 0\n1, GPU-b, 24244, 0\n2, GPU-c, 2211, 100\n'
        with patch.object(s.subprocess, 'check_output', return_value=output), patch.dict(os.environ), patch('builtins.print'):
            self.assertEqual(s.select_gpu('auto')['physical_index'], 1)
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], 'GPU-b')
            self.assertEqual(s.select_gpu('0')['physical_index'], 0)
            with self.assertRaisesRegex(RuntimeError, 'No selected GPU'):
                s.select_gpu('2')

    def test_frozen_config_and_atomic_updates(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'config.json'
            s.save(path, {'seed': 42}, frozen=True)
            s.save(path, {'seed': 42}, frozen=True)
            with self.assertRaisesRegex(ValueError, 'Frozen'):
                s.save(path, {'seed': 43}, frozen=True)
            self.assertEqual(s.read(path), {'seed': 42})
            s.save(path, {'seed': 43})
            self.assertEqual(s.read(path), {'seed': 43})
            self.assertFalse(path.with_suffix('.json.tmp').exists())


if __name__ == '__main__':
    unittest.main()
