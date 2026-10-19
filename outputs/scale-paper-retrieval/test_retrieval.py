import unittest
import retrieval as r


class Checks(unittest.TestCase):
    def test_balanced_mass(self):
        weights=r.sampling_weights([{'sample_id':str(i),'gold_raw':0 if i==0 else 10} for i in range(5)])
        self.assertAlmostEqual(weights[0]['probability'],.5)
        self.assertAlmostEqual(sum(x['probability'] for x in weights[1:]),.5)
        self.assertEqual(r.weighted_draw(weights,20,42),r.weighted_draw(weights,20,42))
        with self.assertRaises(ValueError):r.sampling_weights([{'sample_id':'a','gold_raw':float('nan')}])

    def test_equations_and_type_context(self):
        nodes=[{'id':x,'kind':'w'} for x in ['a','b','z','context']]+[{'id':x,'kind':'c'} for x in ['c1','c2']]+[{'id':'phrase','kind':'t'}]
        edges=[{'kind':k,'source':a,'target':b} for k,a,b in [('Er','a','c1'),('Er','a','c2'),('Er','b','c1'),('Er','z','c2'),('Es','a','context'),('Es','b','context'),('Ew','phrase','a')]]
        index=r.build_index({'nodes':nodes,'edges':edges,'soft_edge_mode':'prose'})
        got=r.query(index,'a','c1')
        self.assertEqual(got,{'Tr':['a','b'],'Ts':['context'],'Tp':['a','b'],'T':['a','b']})
        self.assertEqual(r.query(index,'a','c2')['T'],['a','z'])
        self.assertEqual(index['phrases_by_summary']['a'],['phrase'])
        with self.assertRaises(ValueError):r.query(index,'b','c2')


if __name__=='__main__':unittest.main()
