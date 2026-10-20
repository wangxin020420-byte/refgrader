import argparse
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import main_pipeline
from scripts import run_initial_rubric_ablation as ablation


class InitialRubricAblationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.snapshot = self.root/'inputs'
        self.job = self.root/'results/unique_initial_run'
        self.snapshot.mkdir()
        self.state = {'run_id':'unique_initial_run', 'job':str(self.job),'questions':['CO_1']}
        for name, value in (('ROOT',self.root),('SNAPSHOT',self.snapshot)):
            mock = patch.object(ablation,name,value)
            mock.start()
            self.addCleanup(mock.stop)
        self.rubric = self.snapshot/'rubrics/initial/CO/CO_1_rubric_standard.json'
        self.rubric.parent.mkdir(parents=True)
        ablation.common.write(self.rubric,[{'id':'s1','item':'one criterion','points':5}])
        ablation.common.write(self.snapshot/'exam_database.json',[
            {'question_id':'CO_1','total_score':5,'initial_rubric_path':'wrong/old/path.json',
             'optimized_rubric_path':'wrong/optimized.json','rubric_split_path':'wrong/split.json'}])

    def test_actual_parser_accepts_explicit_initial_paths_for_both_splits(self):
        for split in ('validation','test'):
            command = ablation.grade_command(self.state,split)
            with patch('sys.argv',command[2:]):
                parsed = main_pipeline.parse_args()
            self.assertTrue(parsed.allow_initial_rubric)
            self.assertEqual(parsed.mode,'FULL')
            self.assertEqual(parsed.answer_split,split)
            self.assertEqual(Path(parsed.teacher_db),self.snapshot/'teacher_scores.json')
            self.assertEqual(Path(parsed.rubric_dir),self.job/'no_optimized_rubrics')
            self.assertNotIn('--force-rerun',command)

    def test_runtime_paths_and_actual_rubric_resolution_use_initial_only(self):
        rows = ablation.runtime_database(self.job)
        self.assertEqual(Path(rows[0]['initial_rubric_path']),self.rubric)
        ablation.validate_commands(self.state)

    def test_accidental_optimized_rubric_is_not_silently_accepted(self):
        path = self.job/'no_optimized_rubrics/CO/CO_1_rubric_standard.json'
        path.parent.mkdir(parents=True)
        path.write_bytes(self.rubric.read_bytes())
        with self.assertRaises((ValueError,RuntimeError,FileNotFoundError)):
            ablation.validate_commands(self.state)

    def test_plan_skips_optimization_and_shared_activation(self):
        plan = ablation.stages(self.state)
        self.assertEqual([p['name'] for p in plan],['grade_validation','calibrate','grade_test','evaluate_test'])
        joined = ' '.join(str(part) for p in plan for part in p['command'])
        self.assertNotIn('VARIANCE_OPT',joined)
        self.assertNotIn('activate',joined)
        self.assertNotIn('--push',joined)
        calibration = plan[1]['command']
        self.assertIn(str(self.job/'validation/CO_1_grading_checkpoint.json'),calibration)
        self.assertIn(str(self.job/'a3wa_config.json'),calibration)
        self.assertIn('--score-calibration',calibration)

    def test_evaluation_parser_accepts_plan_without_model_calls(self):
        import evaluate
        command = ablation.stages(self.state)[3]['command']
        captured = []
        original = argparse.ArgumentParser.parse_args
        class Parsed(Exception): pass
        def parse(parser,*args,**kwargs):
            captured.append(original(parser,*args,**kwargs))
            raise Parsed()
        with patch('sys.argv',command[2:]), patch.object(argparse.ArgumentParser,'parse_args',parse):
            with self.assertRaises(Parsed): evaluate.main()
        self.assertEqual(captured[0].result_source,'checkpoint')
        self.assertEqual(Path(captured[0].summary_output),self.job/'reports/summary.json')

    def test_calibration_parser_accepts_plan_without_reading_scores(self):
        from scripts import calibrate_a3wa
        command = ablation.stages(self.state)[1]['command']
        captured = []
        original = argparse.ArgumentParser.parse_args
        class Parsed(Exception): pass
        def parse(parser,*args,**kwargs):
            captured.append(original(parser,*args,**kwargs))
            raise Parsed()
        with patch('sys.argv',command[2:]), patch.object(argparse.ArgumentParser,'parse_args',parse):
            with self.assertRaises(Parsed): calibrate_a3wa.main()
        self.assertTrue(captured[0].score_calibration)
        self.assertEqual(captured[0].text_model,'glm-5.2')

    def test_test_requires_own_calibration_and_validation_clears_inherited_config(self):
        with patch.dict(os.environ,{'A3WA_CALIBRATION_CONFIG':'wrong/optimized/config.json'}):
            self.assertEqual(ablation.child_environment(self.state,'validation')['A3WA_CALIBRATION_CONFIG'],'')
            with self.assertRaises(FileNotFoundError): ablation.child_environment(self.state,'test')
            ablation.common.write(self.job/'a3wa_config.json',{})
            self.assertEqual(ablation.child_environment(self.state,'test')['A3WA_CALIBRATION_CONFIG'],str(self.job/'a3wa_config.json'))

    def test_completeness_includes_neg_and_rejects_missing_or_duplicate_answers(self):
        split = self.snapshot/'splits/by_question/CO_1.json'
        ablation.common.write(split,{'validation':['a','b'],'test':['c']})
        checkpoint=self.job/'validation/CO_1_grading_checkpoint.json'
        rows=[{'student_id':'a','3wd_route':'POS','model_avg_score':2,'three_way_core_score':2,'final_calibrated_score':2},
              {'student_id':'b','3wd_route':'NEG','model_avg_score':1,'three_way_core_score':1,'final_calibrated_score':1}]
        policy=SimpleNamespace(filter_ids=lambda q,ids:ids)
        with patch.object(ablation.SampleQualityPolicy,'load',return_value=policy):
            ablation.common.write(checkpoint,rows)
            ablation.verify_complete(self.state,'validation')
            ablation.common.write(checkpoint,rows[:1])
            with self.assertRaises(RuntimeError): ablation.verify_complete(self.state,'validation')
            ablation.common.write(checkpoint,rows+[rows[0]])
            with self.assertRaises(RuntimeError): ablation.verify_complete(self.state,'validation')


if __name__=='__main__':
    unittest.main()
