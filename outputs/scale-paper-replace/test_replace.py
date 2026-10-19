import unittest
import replace as r


class Checks(unittest.TestCase):
    def test_source_span_and_determinism(self):
        nodes={'w:b':{'text':'B'},'t:b':{'text':'beta'}}
        args=('alpha alpha',{'t':'alpha','w':'A'},['w:b'],[.5],{'w:b':['t:b']},nodes,5,'42|a|1')
        got=r.select(*args)
        self.assertEqual(got,r.select(*args))
        self.assertEqual(got['modified_answer'],'beta alpha')
        self.assertEqual(got['effective_k'],1)
        self.assertEqual(got['source_span'],[0,5])

    def test_noop_retained(self):
        w='w:'+r.sha(b'A');nodes={w:{'text':'A'},'t:a':{'text':'alpha'}}
        got=r.select('alpha',{'t':'alpha','w':'A'},[w],[1.],{w:['t:a']},nodes,1,'42')
        self.assertFalse(got['answer_changed']);self.assertTrue(got['same_summary_selected'])
        self.assertEqual(got['selected_rank'],1)

    def test_topk_only(self):
        nodes={'w:a':{'text':'A'},'w:b':{'text':'B'},'t:a':{'text':'a'},'t:b':{'text':'b'}}
        for seed in range(10):
            got=r.select('a',{'t':'a','w':'A'},['w:a','w:b'],[1.,.5],{'w:a':['t:a'],'w:b':['t:b']},nodes,1,str(seed))
            self.assertEqual(got['replacement_summary_id'],'w:a')


if __name__=='__main__':unittest.main()
