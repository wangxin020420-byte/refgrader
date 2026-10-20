import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import main_pipeline
from sample_quality import SampleQualityPolicy
from scripts import run_csbench


class CSBenchGradePolicyPathTests(unittest.TestCase):
    def test_validation_and_test_pass_dataset_teacher_path_before_model_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            teacher = root / 'teacher_scores.json'
            teacher.write_text('{"A":{"CO_1":5}}', encoding='utf-8')
            metadata = root / 'answer_metadata.jsonl'
            metadata.write_text('{"answer_id":"A","question_id":"CO_1"}\n', encoding='utf-8')
            policy_path = root / 'policy.json'
            policy_path.write_text(json.dumps({
                'schema_version': 1, 'policy_id': 'reviewed-test',
                'dataset_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in (teacher, metadata)},
                'excluded': {}, 'corrected_scores': {},
            }), encoding='utf-8')
            ctx = SimpleNamespace(root=root, question_id='CO_1', teacher_db=teacher,
                database=root/'exam_database.json', answer_metadata=metadata,
                initial_dir=root/'initial', optimized_dir=root/'optimized',
                validate_optimized=Mock())
            for split in ('validation', 'test'):
                with self.subTest(split=split):
                    flags = ['grade', 'CO_1', '--split', split, '--dry-run', '--no-artifacts']
                    if split == 'test':
                        flags += ['--no-active-a3wa']
                    args = run_csbench.build_parser().parse_args(flags)
                    with (
                        patch.object(run_csbench, 'build_contexts', return_value=[ctx]),
                        patch.object(run_csbench, 'validate_active_configuration'),
                        patch.object(run_csbench, 'select_grading_run', return_value=(root/'run', 'fixed')),
                        patch.object(run_csbench, 'completed_questions_for_split', return_value=set()),
                        patch.object(run_csbench, '_read_run_state', return_value=None),
                        patch.object(run_csbench, 'execute', return_value=0) as execute,
                    ):
                        self.assertEqual(run_csbench.grade(args), 0)
                    command = execute.call_args.args[0]
                    with patch('sys.argv', command[1:]):
                        parsed = main_pipeline.parse_args()
                    self.assertEqual(Path(parsed.teacher_db), teacher)
                    policy = SampleQualityPolicy.load(Path(parsed.teacher_db).parent,
                        explicit_path=policy_path, force_raw=False)
                    self.assertEqual(policy.policy_id, 'reviewed-test')


if __name__ == '__main__':
    unittest.main()
