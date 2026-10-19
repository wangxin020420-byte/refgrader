import tempfile
import unittest
from pathlib import Path
import stage1 as m


class Checks(unittest.TestCase):
    def test_reference_provenance(self):
        rows=[dict(sample_id=s,question_id='1',answer='',references=['same','same','']) for s in ['a','b']]
        units=m.make_units(rows)
        refs=[r for r in units if r['role']=='reference']
        self.assertEqual(len(units),3)
        self.assertEqual(refs[0]['source_sample_ids'],['a','b'])

    def test_grounding_and_coverage(self):
        units=[dict(unit_id='a',text='alpha beta')]
        triple=dict(t='alpha',c='term',w='A')
        self.assertEqual(len(m.validate(units,[dict(unit_id='a',triples=[triple])])),1)
        for records in [[],[dict(unit_id='a',triples=[dict(t='invented',c='term',w='A')])],
                        [dict(unit_id='a',triples=[triple,triple])]]:
            with self.assertRaises(ValueError):m.validate(units,records)

    def test_graph_modes(self):
        records=[dict(unit_id='a',triples=[dict(t='alpha',c='term',w='A'),dict(t='beta',c='term',w='B')]),
                 dict(unit_id='b',triples=[dict(t='gamma',c='term',w='C')])]
        for mode,count in [('prose',2),('equation4',4)]:
            graph=m.graph(records,mode)
            soft=[e for e in graph['edges'] if e['kind']=='Es']
            self.assertEqual(len(soft),count)
            self.assertTrue(all(e['source_unit_ids']==['a'] for e in soft))
            self.assertEqual(len([e for e in graph['edges'] if e['kind']=='Er']),3)
        with self.assertRaises(ValueError):m.graph([],'unknown')

    def test_write_guard(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            m.write_all(p,{'a':b'original'})
            m.write_all(p,{'a':b'original'})
            with self.assertRaises(ValueError):m.write_all(p,{'b':b'new','a':b'changed'})
            self.assertFalse((p/'b').exists())


if __name__=='__main__':unittest.main()
