import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.benchmarks import scale4_data as data
from scripts.benchmarks.run_scale4 import freeze, selected_questions, metric_report
from benchmark_datasets.contract import write_json, sha256_file, audit_prepared_benchmark
from model_runtime import runtime_model_config

SOURCE = Path(__file__).parent / "tmp/refgrader_scale4_source"


class ScaleFourTests(unittest.TestCase):
    def test_profile_is_opt_in(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(runtime_model_config()["text_model"], "glm-5.2")
            self.assertEqual(runtime_model_config(text_provider="glm53")["text_model"], "glm-5.3")

    def test_freeze_rejects_changed_protocol(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'lock.json'
            freeze(p, {"model":"glm-5.3"})
            freeze(p, {"model":"glm-5.3"})
            with self.assertRaises(ValueError):
                freeze(p, {"model":"glm-5.2"})

    @unittest.skipUnless((SOURCE/'tree.json').exists(), "pinned public source fixture absent")
    def test_actual_four_datasets_identity_isolation_and_resume(self):
        for ds in data.DATASETS:
            with self.subTest(dataset=ds), tempfile.TemporaryDirectory() as folder:
                root=Path(folder)/ds
                audit=data.prepare(SOURCE,ds,root)
                self.assertEqual(audit['answer_count'],sum(data.EXPECTED[ds]))
                self.assertEqual(audit['split_counts']['validation'],50)
                self.assertEqual(audit['split_counts']['test'],data.EXPECTED[ds][1]-50)
                self.assertEqual(audit['split_counts']['train']+audit['split_counts']['calibration'],data.EXPECTED[ds][0])
                before=sha256_file(root/'teacher_scores.json')
                self.assertEqual(data.prepare(SOURCE,ds,root),audit)
                self.assertEqual(before,sha256_file(root/'teacher_scores.json'))
                metadata=[json.loads(x) for x in (root/'answer_metadata.jsonl').read_text(encoding='utf8').splitlines()]
                self.assertFalse(any(set(r)&{'score','label','source_score','source_max'} for r in metadata))
                by_id={r['answer_id']:r for r in metadata}
                for q in json.loads((root/'exam_database.json').read_text(encoding='utf8')):
                    split=json.loads((root/q['rubric_split_path']).read_text())
                    self.assertLessEqual(len(split['calibration']),5)
                    self.assertTrue(all(by_id[i]['source_split']=='train' for i in split['calibration']))
                    self.assertTrue(all(by_id[i]['source_split']=='test' for i in split['validation']+split['test']))
                if ds=='ADS':
                    self.assertTrue(any(r['blank_passthrough'] for r in metadata))
                    self.assertEqual(audit['question_count'],15)

    def test_metric_has_fixed_qwk_and_explicit_normalization(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            write_json(root/'teacher_scores.json',{'a':{'q':0},'b':{'q':10}})
            write_json(root/'split.json',{'test':['a','b']})
            write_json(root/'q_grading_checkpoint.json',[{'student_id':'a','final_calibrated_score':1},{'student_id':'b','final_calibrated_score':9}])
            c={'root':root,'questions':[{'question_id':'q','rubric_split_path':'split.json'}]}
            result=metric_report(c,root)
            self.assertEqual(result['records'],2)
            self.assertEqual(result['rmse_raw_0_10'],1)
            self.assertEqual(result['rmse_normalized'],.1)
            self.assertAlmostEqual(result['qwk'],.9756097561)

    def test_text_variance_selection_uses_calibration_not_images(self):
        import main_pipeline as main
        q={'question_id':'q','rubric_calibration_ids':['cal']}
        with patch.object(main,'load_answer_metadata',return_value={'cal':{'question_id':'q','raw_text':'a'},'test':{'question_id':'q','raw_text':'b'}}):
            self.assertEqual(main.select_question_text_answers(q,None,'calibration'),['cal.txt'])


if __name__=='__main__':
    unittest.main()
