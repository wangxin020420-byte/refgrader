import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import run_stage2 as a

class Tests(unittest.TestCase):
    def test_weights_and_original_replacement(self):
        r=a.load('retrieval_contract');s=a.load('replacement_contract')
        weights=r.sampling_weights([dict(sample_id=str(i),gold_raw=v) for i,v in enumerate([0,0,10])])
        self.assertEqual([x['probability'] for x in weights],[.25,.25,.5])
        t=dict(t='old',w='summary',c='term')
        x=s.select('old old',t,['w:x'],[1.],{'w:x':['t:x']},{'w:x':{'text':'new summary'},'t:x':{'text':'new'}},5,'42')
        self.assertEqual(x['modified_answer'],'new old')
        self.assertEqual(x['effective_k'],1)

    def test_345_training_only_resume_and_rankings(self):
        with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()):
            root=Path(d);exp=root/'runs/online_experiments/csbench_scale_online_v1';prev=exp/'scale_stage1';prev.mkdir(parents=True)
            quads=[dict(sample_id=str(i),question_id='q',q='question',r=['reference'],a='old',l=5) for i in range(345)]
            units=[dict(unit_id='student:'+str(i),text='old') for i in range(345)]
            extraction=[dict(unit_id=u['unit_id'],triples=[dict(t='old',c='term',w='summary')]) for u in units]
            wn='w:'+a.sha(b'summary');cn='c:'+a.sha(b'term');tn='t:'+a.sha(b'old')
            graph=dict(soft_edge_mode='prose',nodes=[dict(id=wn,kind='w',text='summary'),dict(id=cn,kind='c',text='term'),dict(id=tn,kind='t',text='old')],edges=[dict(kind='Er',source=wn,target=cn),dict(kind='Ew',source=tn,target=wn)])
            jl=lambda x:''.join(json.dumps(r)+'\n' for r in x).encode()
            graph['extractions_sha256']=a.sha(jl(extraction))
            files={'graph.json':a.enc(graph),'training_sources.json':a.enc(quads),'source_units.jsonl':jl(units),'extractions.jsonl':jl(extraction)}
            for n,b in files.items():(prev/n).write_bytes(b)
            (prev/'config.json').write_bytes(a.enc({'fixture':True}))
            (prev/'summary.json').write_bytes(a.enc(dict(status='CSBENCH_TRAIN_EXTRACTION_AND_GRAPH_COMPLETE',remaining=0,config_sha256=a.sha((prev/'config.json').read_bytes()),files_sha256={n:a.sha(b) for n,b in files.items()})))
            original={p.name:p.read_bytes() for p in prev.iterdir()}
            out,layout,art,q,e,index,g=a.prepare(root,'csbench_scale_online_v1',5,42)
            eq=art/'eq10_v1';rank=eq/'rankings';rank.mkdir(parents=True)
            order,scores=a.load('embedding_contract').ranking(np.array([[1.,0.]],dtype='float32'))
            path=rank/'fixture.npz';np.savez_compressed(path,order=order,scores=scores)
            path.with_suffix('.json').write_bytes(a.enc(dict(type_id=cn,summary_ids=[wn])))
            cfg=dict(graph_sha256=a.sha((art/'full_graph_v1/graph.json').read_bytes()),index_sha256=a.sha((art/'retrieval_v1/index.json').read_bytes()))
            (eq/'config.json').write_bytes(a.enc(cfg))
            (eq/'summary.json').write_bytes(a.enc(dict(status='PAPER_EQ10_RANKING_COMPLETE',config_sha256=a.sha(a.enc(cfg)),rankings=[dict(type_id=cn,file='rankings/fixture.npz',sha256=a.sha(path.read_bytes()))])))
            result=a.replace(art,q,e,index,g,5,42)
            self.assertEqual(result['replacement_records'],345)
            self.assertEqual(result['changed_answer_records'],0)
            self.assertTrue(all(x['assigned_label'] is None for x in a.rows(art/'replacement_inputs_v1/replacement_records.jsonl')))
            a.prepare(root,'csbench_scale_online_v1',5,42)
            self.assertEqual(a.replace(art,q,e,index,g,5,42),result)
            self.assertTrue(all((prev/n).read_bytes()==b for n,b in original.items()))
            with self.assertRaises(ValueError):a.prepare(root,'csbench_scale_online_v1',6,42)

if __name__=='__main__':unittest.main()
