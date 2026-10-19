import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import run_stage1 as a

def fixture(root):
    source=root/'data/csbench_within_question_v1'
    visual=root/'runs/online_experiments/csbench_scale_online_v1/visual'
    source.mkdir(parents=True)
    (visual/'text_view').mkdir(parents=True)
    rows=[dict(sample_id=str(i),question_id='q',question='question',references=['reference'],answer='' if i<5 else 'answer '+str(i),blank_passthrough=i<5) for i in range(345)]
    labels=[dict(sample_id=str(i),question_id='q',score_original=2,score_max_original=4,score_0_10=5) for i in range(345)]
    m=a.contract()
    (source/'train.inputs.jsonl').write_bytes(m.jsonlines(rows))
    (source/'train.labels.jsonl').write_bytes(m.jsonlines(labels))
    (visual/'text_view/train.inputs.jsonl').write_bytes(m.jsonlines(rows))
    a.save(source/'manifest.json',dict(counts=dict(train=345,validation=343,test=2723),files_sha256={n:a.sha((source/n).read_bytes()) for n in ('train.inputs.jsonl','train.labels.jsonl')}))
    a.save(visual/'experiment_summary.json',dict(experiment_id='csbench_scale_online_v1',inputs_complete=True,pending=[],files_sha256={'train.inputs.jsonl':a.sha(m.jsonlines(rows))}))
    return source,visual

def client(calls,mode='valid'):
    def create(**kw):
        calls.append(kw)
        if mode=='400':
            exc=RuntimeError('hidden error');exc.status_code=400;exc.body={'error':{'code':'1301'}};raise exc
        text=json.loads(kw['messages'][1]['content'])['text']
        obj={'triples':[dict(t=text.split()[0],c='term',w='meaning')]}
        envelope={'choices':[dict(finish_reason='stop',message={'content':json.dumps(obj)})]}
        return SimpleNamespace(model_dump=lambda **_:envelope)
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

class Tests(unittest.TestCase):
    def test_full_resume_and_graph(self):
        with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()):
            root=Path(d);source,visual=fixture(root);calls=[];c=client(calls)
            before={p.name:p.read_bytes() for p in source.iterdir()}
            self.assertEqual(a.execute(root,'csbench_scale_online_v1',limit=2,client=c),2)
            self.assertEqual(len(calls),2)
            self.assertEqual(a.execute(root,'csbench_scale_online_v1',client=c),0)
            self.assertEqual(len(calls),341)
            out=visual.parent/'scale_stage1'
            self.assertEqual(a.execute(root,'csbench_scale_online_v1',client=c),0)
            self.assertEqual(len(calls),341)
            self.assertEqual(a.read(out/'summary.json')['remaining'],0)
            self.assertEqual(len(a.lines(out/'extractions.jsonl')),346)
            self.assertTrue(all((source/n).read_bytes()==raw for n,raw in before.items()))
            self.assertTrue(all('score' not in kw['messages'][1]['content'] for kw in calls))
            # Recover a valid saved envelope after an interrupted accepted-file write.
            accepted=next(out.glob('responses/*/accepted.json'));accepted.unlink()
            self.assertEqual(a.execute(root,'csbench_scale_online_v1',client=c),0)
            self.assertEqual(len(calls),341)

    def test_changed_inputs_rejected(self):
        with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()):
            root=Path(d);source,visual=fixture(root)
            a.prepare(root,'csbench_scale_online_v1')
            path=visual/'text_view/train.inputs.jsonl'
            path.write_bytes(path.read_bytes()+b'\n')
            with self.assertRaises(ValueError):a.prepare(root,'csbench_scale_online_v1')

    def test_permanent_error_not_retried_and_invalid_span_rejected(self):
        with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()):
            root=Path(d);_,visual=fixture(root);calls=[]
            self.assertEqual(a.execute(root,'csbench_scale_online_v1',limit=1,client=client(calls,'400')),2)
            self.assertEqual(len(calls),1)
            self.assertEqual(a.execute(root,'csbench_scale_online_v1',limit=1,client=client(calls)),2)
            self.assertEqual(len(calls),2)
            m=a.contract()
            with self.assertRaises(ValueError):
                a.parse(m,dict(unit_id='x',text='answer'),{'choices':[dict(finish_reason='stop',message={'content':json.dumps({'triples':[dict(t='invented',c='term',w='x')]})})]})

if __name__=='__main__':unittest.main()
