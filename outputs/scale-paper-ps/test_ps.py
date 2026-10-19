import contextlib
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import ps


class Checks(unittest.TestCase):
    def fixture(self):
        q=[{'sample_id':'a','q':'question','a':'original','r':['reference'],'l':7},
           {'sample_id':'b','q':'question','a':'other','r':['reference'],'l':3}]
        weights=[{'sample_id':'a','probability':.5},{'sample_id':'b','probability':.5}]
        rep=[{'sample_id':'a','record_id':'r','source_answer_sha256':ps.sha(b'original'),'modified_answer':'modified','answer_changed':True}]
        return q,weights,rep

    def test_plan_provenance_and_no_label_prompt(self):
        args=self.fixture();a,skip=ps.make_plan(*args,10,42);b,_=ps.make_plan(*args,10,42)
        self.assertEqual(a,b);self.assertFalse(skip)
        self.assertEqual(len(a),10)
        for req in a:
            body=json.loads(req['messages'][1]['content'])
            self.assertEqual(set(body),{'question','reference_answers','tasks'})
            self.assertEqual(len(body['tasks']),5)
            self.assertEqual([r['branch'] for r in req['provenance'][:2]],['original','original'])

    def test_parser_rejects_missing_and_truncated(self):
        request,_=ps.make_plan(*self.fixture(),1,42)
        for raw in [{'choices':[{'finish_reason':'length'}]}, {'choices':[{'finish_reason':'stop','message':{'content':'{"answers":[]}'}}]}]:
            with self.assertRaises(ValueError):ps.parse_response(raw,request[0])

    def test_resume_and_offline(self):
        with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()):
            root=Path(d);base=root/'paper_method_v1';prev=base/'artifacts/retrieval_v1';rep=base/'artifacts/replacement_inputs_v1'
            q,w,replacements=self.fixture()
            blobs={'training_quadruples.json':ps.enc(q),'sampling_weights.json':ps.enc(w)}
            ps.save(prev,dict(blobs,**{'summary.json':ps.enc({'output_sha256':{n:ps.sha(b) for n,b in blobs.items()}})}))
            raw=ps.jl(replacements)
            ps.save(rep,{'replacement_records.jsonl':raw,'summary.json':ps.enc({'output_sha256':{'replacement_records.jsonl':ps.sha(raw)}})})
            ps.save(base/'runs/pe_glm_full_v1',{'config.json':ps.enc({'model':'mock','base_url':'https://mock.invalid'})})
            ps.prepare(root,4,42);ps.prepare(root,4,42)
            def response(req,timeout):
                payload=json.loads(json.loads(req.data)['messages'][1]['content'])
                content=json.dumps({'answers':[{'variant_id':t['variant_id'],'answer':t['input_answer']} for t in payload['tasks']]})
                return io.StringIO(json.dumps({'model':'mock','choices':[{'finish_reason':'stop','message':{'content':content}}]}))
            fake=types.SimpleNamespace(flock=lambda *a:None,LOCK_EX=1,LOCK_NB=2)
            with patch.dict('sys.modules',{'fcntl':fake}),patch.object(ps.getpass,'getpass',return_value='TEST_SECRET'),patch.object(ps.time,'sleep'),patch.object(ps.urllib.request,'urlopen',side_effect=response) as api:
                ps.execute(root,False,1);self.assertEqual(api.call_count,1)
                ps.execute(root,True,None);self.assertEqual(api.call_count,1)
                ps.execute(root,False,None);self.assertEqual(api.call_count,4)
                ps.execute(root,False,None);self.assertEqual(api.call_count,4)
            summary=ps.read(base/'runs/ps_glm_v1/summary.json')
            self.assertEqual(summary['candidates'],20)
            self.assertEqual(summary['status'],'PS_GENERATION_COMPLETE_UNLABELED')
            self.assertFalse(any('TEST_SECRET' in f.read_text(encoding='utf8') for f in base.rglob('*.json')))


if __name__=='__main__':unittest.main()
