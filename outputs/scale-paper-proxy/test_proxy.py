import contextlib
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import urllib.error
import proxy as p


class Checks(unittest.TestCase):
    def sources(self):
        return [{'sample_id':'s'+str(i),'question_id':'q','q':'Question','a':('' if i==1 else 'Wrong formula '+str(i)),
                 'r':['Reference'],'l':i} for i in range(4)]

    def fixture(self,root):
        base=root/'paper_method_v1/artifacts';retr=base/'retrieval_v1';flt=base/'initial_filter_k5_b5000_v1';ps=base/'ps_inputs_v1'
        p.save(retr/'training_quadruples.json',p.enc(self.sources()))
        p.save(retr/'summary.json',p.enc({'output_sha256':{'training_quadruples.json':p.sha(retr/'training_quadruples.json')}}))
        p.save(flt/'config.json',p.enc({'k':5}))
        p.save(flt/'summary.json',p.enc({'status':'INITIAL_FILTER_COMPLETE_UNLABELED','selected_candidates':5000,
                                      'config_sha256':p.sha(flt/'config.json'),'files_sha256':{}}))
        p.save(ps/'config.json',p.enc({'model':'fixture','base_url':'https://mock.invalid/api','max_tokens':16384}))
        p.save(ps/'summary.json',p.enc({'files_sha256':{'config.json':p.sha(ps/'config.json')}}))

    def response(self,request,timeout):
        body=json.loads(request.data)
        item=json.loads(body['messages'][1]['content'])
        self.assertEqual(set(item),{'question','original_answer'})
        raw={'choices':[{'finish_reason':'stop','message':{'content':'```json\n'+json.dumps({'proxy_answer':item['original_answer']})+'\n```'}}],
             'usage':{'total_tokens':9},'model':'fixture'}
        return io.StringIO(json.dumps(raw))

    def test_plan_labels_omitted_and_blank_preserved(self):
        reqs,blank=p.make_plan(self.sources())
        self.assertEqual(len(reqs),3);self.assertEqual(blank[0]['sample_id'],'s1')
        for request in reqs:
            payload=json.loads(request['messages'][1]['content'])
            self.assertEqual(set(payload),{'question','original_answer'})
        with self.assertRaises(ValueError):p.make_plan(self.sources()*2)

    def test_parser(self):
        raw={'choices':[{'finish_reason':'stop','message':{'content':'```json\n{"proxy_answer":"do not correct 2+2=5"}\n```'}}]}
        self.assertEqual(p.parse(raw),('do not correct 2+2=5',True))
        for content in ['{"proxy_answer":"","score":3}','{"proxy_answer":""}','explain {"proxy_answer":"x"}']:
            raw['choices'][0]['message']['content']=content
            with self.assertRaises(ValueError):p.parse(raw)
        raw['choices'][0]['finish_reason']='length'
        with self.assertRaises(ValueError):p.parse(raw)

    def test_small_run_offline_resume_and_no_repeat(self):
        with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()):
            root=Path(d);self.fixture(root)
            with patch.dict('sys.modules',{'fcntl':types.SimpleNamespace(LOCK_EX=2,LOCK_NB=4,flock=lambda *a:None)}),patch.object(p.getpass,'getpass',return_value='secret'),patch.object(p.time,'sleep'):
                p.prepare(root);p.prepare(root)
                with patch.object(p.urllib.request,'urlopen',side_effect=self.response) as api:
                    p.execute(root,False,1,False);self.assertEqual(api.call_count,1)
                    p.execute(root,True,None,False);self.assertEqual(api.call_count,1)
                    p.execute(root,False,None,False);self.assertEqual(api.call_count,3)
                    p.execute(root,False,None,False);self.assertEqual(api.call_count,3)
                run=root/'paper_method_v1/runs/proxy_glm_v1'
                report=p.read(run/'summary.json')
                self.assertEqual(report['status'],'PROXY_GENERATION_COMPLETE_REVIEW_REQUIRED')
                self.assertEqual(report['proxy_records'],4)
                self.assertEqual(report['new_api_calls_this_invocation'],0)
                self.assertTrue(all(r['assigned_label'] is None for r in p.rows(run/'proxy_records.jsonl')))
                self.assertTrue((root/'paper_method_v1/artifacts/proxy_generated_v1/manifest.json').exists())

    def test_network_failure_explicit_retry_preserves_attempt_history(self):
        with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()):
            root=Path(d);self.fixture(root)
            with patch.dict('sys.modules',{'fcntl':types.SimpleNamespace(LOCK_EX=2,LOCK_NB=4,flock=lambda *a:None)}),patch.object(p.getpass,'getpass',return_value='secret'),patch.object(p.time,'sleep'):
                p.prepare(root)
                with patch.object(p.urllib.request,'urlopen',side_effect=urllib.error.URLError('failure secret')):
                    p.execute(root,False,1,False)
                with patch.object(p.urllib.request,'urlopen',side_effect=self.response) as api:
                    p.execute(root,False,None,False);self.assertEqual(api.call_count,2)
                    p.execute(root,False,None,True);self.assertEqual(api.call_count,3)
                run=root/'paper_method_v1/runs/proxy_glm_v1';rid=p.rows(root/'paper_method_v1/artifacts/proxy_inputs_v1/requests.jsonl')[0]['request_id']
                self.assertEqual(len(list((run/'attempts'/rid).glob('*.json'))),2)
                self.assertNotIn('secret',(run/'errors'/rid/'0001.json').read_text())
                self.assertEqual(p.read(run/'summary.json')['status'],'PROXY_GENERATION_COMPLETE_REVIEW_REQUIRED')


if __name__=='__main__':unittest.main()
