import copy
import unittest
import run_filter as r

class FilterTests(unittest.TestCase):
    def fixture(self):
        sources = [dict(sample_id=f's{i}', question_id='q', q='Question', r=['Reference'], a='Original', l=0.0) for i in range(345)]
        candidates = [dict(variant_id=f'v{i}', sample_id='s0', answer='Candidate', source_score=0.0) for i in range(5000)]
        scored = [dict(x, assigned_label=0.3, annotator_seed=42, absolute_gap_raw=0.3, label_origin='annotator_raw_regression') for x in candidates]
        return sources, candidates, scored

    def test_boundary_and_raw_prediction(self):
        s,c,v = self.fixture()
        for i,value in enumerate([0.2, 0.2000001, -0.1]):
            v[i].update(assigned_label=value, absolute_gap_raw=abs(value))
        kept,aug,audit = r.c.select(v,s,c,42)
        self.assertEqual([x['variant_id'] for x in kept], ['v0','v2'])
        self.assertEqual(aug[-1]['label'], -0.1)
        self.assertEqual(len(aug),347)

    def test_wrong_seed_rejected(self):
        s,c,v=self.fixture();v[1]['annotator_seed']=43
        with self.assertRaises(ValueError): r.c.select(v,s,c,42)

    def test_changed_candidate_rejected(self):
        s,c,v=self.fixture();v[1]['answer']='Changed'
        with self.assertRaises(ValueError): r.c.select(v,s,c,42)

    def test_original_not_proxy(self):
        train=[dict(sample_id=f's{i}',question_id='q',question='Q',references=['R'],answer='Polished',label=1) for i in range(345)]
        visual=copy.deepcopy(train)
        for v in visual: v['answer']='Unpolished'
        sources=r.prepare_sources(train,visual)
        self.assertTrue(all(s['a']=='Unpolished' for s in sources))
        visual[0]['references']=['Changed']
        with self.assertRaises(ValueError):r.prepare_sources(train,visual)

if __name__=='__main__':unittest.main()
