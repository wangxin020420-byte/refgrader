import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import initial_filter as f


def unit(x):return x/np.linalg.norm(x,axis=1,keepdims=True)


def reference(o,c,k,b):
    # Independent scalar implementation of Eq12/13 and snapshot-round interpretation.
    anchors=[('o',i) for i in range(len(o))];remaining=list(range(len(c)));selected=[]
    while len(selected)<b:
        for kind,idx in list(anchors):
            a=o[idx] if kind=='o' else c[idx]
            neighbors=sorted([j for j in range(len(o)) if kind!='o' or j!=idx],
                             key=lambda j:(-float(np.dot(a,o[j])),j))[:k]
            scores={j:(1-float(np.clip(np.dot(c[j],a),-1,1)))/max(
                sum(1-float(np.clip(np.dot(c[j],o[t]),-1,1)) for t in neighbors)/k,1e-8)
                for j in remaining}
            winner=max(remaining,key=lambda j:scores[j])
            selected.append(winner);remaining.remove(winner);anchors.append(('c',winner))
            if len(selected)==b:break
    return selected


class Checks(unittest.TestCase):
    def test_matches_scalar_equations_and_resume(self):
        rng=np.random.RandomState(3);o=unit(rng.randn(3,8));c=unit(rng.randn(11,8))
        expected=reference(o,c,2,10)
        state=f.fresh_state(3);list(f.selection_steps(o,c,2,10,state))
        self.assertEqual([x['candidate_index'] for x in state['trace']],expected)
        self.assertTrue(any(x['anchor_code']>=0 for x in state['trace']))
        partial=f.fresh_state(3);iterator=f.selection_steps(o,c,2,10,partial)
        for _ in range(7):next(iterator)
        resumed=json.loads(json.dumps(partial));list(f.selection_steps(o,c,2,10,resumed))
        self.assertEqual(state,resumed)
        self.assertEqual(len(set(expected)),10)

    def test_zero_distance_ties_and_budget(self):
        o=np.ones((3,1));c=np.ones((8,1));state=f.fresh_state(3)
        list(f.selection_steps(o,c,2,5,state))
        self.assertEqual([x['candidate_index'] for x in state['trace']],list(range(5)))
        self.assertTrue(all(x['ratio']==0 and x['denominator_floored'] for x in state['trace']))
        with self.assertRaises(ValueError):list(f.selection_steps(o,c,3,5,f.fresh_state(3)))

    def test_file_pipeline_and_completed_run_no_rerun(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base=root/'paper_method_v1/artifacts';emb=base/'initial_filter_embeddings_v2_max512'
            source=base/'retrieval_v1/training_quadruples.json';cand=base/'ps_generated_v1/candidates.jsonl'
            originals=[{'sample_id':'s%d'%i,'a':'a%d'%i} for i in range(6)]
            candidates=[{'variant_id':'v%02d'%i,'sample_id':'s%d'%(i%6),'answer':'b%d'%i,'assigned_label':None} for i in range(9)]
            f.write(source,f.enc(originals));f.write(cand,''.join(json.dumps(x)+'\n' for x in candidates).encode())
            manifest=cand.parent/'manifest.json';f.write(manifest,f.enc({'candidate_sha256':f.sha(cand)}))
            texts=[x['a'] for x in originals]+[x['answer'] for x in candidates]
            records=[{'kind':'original','id':x['sample_id'],'sample_id':x['sample_id'],'vector_row':i} for i,x in enumerate(originals)]
            records += [{'kind':'candidate','id':x['variant_id'],'sample_id':x['sample_id'],'vector_row':i+6} for i,x in enumerate(candidates)]
            for name,value in [('config.json',{}),('texts.json',texts),('records.json',records),('truncation_audit.json',[])]:f.write(emb/name,f.enc(value))
            vec=unit(np.random.RandomState(5).randn(15,768)).astype('float32');chunk=emb/'chunk.npy'
            np.save(chunk,vec,allow_pickle=False)
            report={'status':'INITIAL_FILTER_EMBEDDINGS_READY','chunks':[{'start':0,'count':15,'file':'chunk.npy','sha256':f.sha(chunk)}],
                    'sources_sha256':f.sha(source),'candidates_sha256':f.sha(cand),'generation_manifest_sha256':f.sha(manifest),
                    'blank_original_answers':0,'truncated_unique_texts':0}
            for name,key in [('config.json','config_sha256'),('texts.json','texts_sha256'),('records.json','records_sha256'),('truncation_audit.json','truncation_audit_sha256')]:report[key]=f.sha(emb/name)
            f.write(emb/'summary.json',f.enc(report))
            before={p:p.read_bytes() for p in base.rglob('*') if p.is_file()}
            # Only the Linux process-lock API is mocked on Windows.
            harness="import sys,types,runpy;sys.modules['fcntl']=types.SimpleNamespace(LOCK_EX=2,LOCK_NB=4,flock=lambda *a:None);sys.argv=sys.argv[1:];runpy.run_path(sys.argv[0],run_name='__main__')"
            script=Path(f.__file__).resolve()
            for command in ['preflight','run','run']:
                result=subprocess.run([sys.executable,'-c',harness,str(script),command,'--root',str(root),'--neighbors','5','--budget','8'],capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('ALREADY_COMPLETE',result.stdout)
            out=base/'initial_filter_k5_b8_v1';summary=f.read(out/'summary.json')
            self.assertEqual(summary['selected_candidates'],8)
            self.assertEqual(summary['remaining_candidates'],1)
            self.assertEqual(summary['assigned_labels'],0)
            self.assertTrue(all(p.read_bytes()==data for p,data in before.items()))
            selected=f.rows(out/'selected_candidates.jsonl')
            lookup={x['variant_id']:x for x in candidates}
            self.assertTrue(all(x==lookup[x['variant_id']] for x in selected))
            with chunk.open('ab') as handle:handle.write(b'corruption')
            with self.assertRaisesRegex(ValueError,'hash'):f.load(root)


if __name__=='__main__':unittest.main()
