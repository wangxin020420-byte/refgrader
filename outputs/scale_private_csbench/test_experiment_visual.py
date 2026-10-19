import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from PIL import Image
import experiment_visual_v1 as w

class Tests(unittest.TestCase):
    def test_new_experiment_resume_and_export(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            inputs = root / 'inputs'
            inputs.mkdir()
            image = root / 'a.png'
            Image.new('RGB', (8,8)).save(image)
            m = w.load_base(Path(w.__file__).with_name('transcribe_csbench_coding_v2.py'))
            row = dict(sample_id='a', question_id='q', question='question', references=['ref'],
                       answer='如图', blank_passthrough=False, student_image=str(image))
            blank = dict(row, sample_id='b', answer='', blank_passthrough=True)
            data = {'train.inputs.jsonl':[row,blank], 'validation.inputs.jsonl':[], 'test.inputs.jsonl':[],
                    'questions.jsonl':[], 'nonblank_text_review_queue.jsonl':[]}
            hashes = {}
            for name, values in data.items():
                raw = ''.join(json.dumps(x)+'\n' for x in values).encode()
                (inputs/name).write_bytes(raw)
                hashes[name] = m.sha(raw)
            m.write(inputs/'manifest.json', dict(files_sha256=hashes))
            w.install_plan(m, 'e1')
            calls = []
            def create(**kwargs):
                calls.append(kwargs)
                response = {'choices':[{'finish_reason':'stop','message':{'content':json.dumps({'text':'A -> B', 'issues':[]})}}]}
                return SimpleNamespace(model_dump=lambda **k:response)
            client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
            out = root/'e1'/'visual'
            self.assertEqual(m.run(inputs,out,0,1,client),0)
            self.assertEqual(w.export(m,inputs,out),0)
            self.assertEqual(m.run(inputs,out,0,1,client),0)
            self.assertEqual(len(calls),1)
            result = w.rows(out/'text_view/train.inputs.jsonl')
            self.assertEqual(result[0]['answer'],'A -> B')
            self.assertEqual(result[1]['answer'],'')
            self.assertEqual(w.rows(inputs/'train.inputs.jsonl')[0]['answer'],'如图')
            other = w.load_base(Path(w.__file__).with_name('transcribe_csbench_coding_v2.py'))
            w.install_plan(other,'e2')
            self.assertEqual(other.run(inputs,root/'e2'/'visual',0,1,client),0)
            self.assertEqual(len(calls),2)
            self.assertTrue(all(m.sha((inputs/n).read_bytes()) == h for n,h in hashes.items()))

if __name__ == '__main__':
    unittest.main()
