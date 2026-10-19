"""Pinned text2vec embeddings and complete within-type cosine rankings (Eq10)."""
import argparse
import hashlib
import json
import os
from pathlib import Path

REPO='shibing624/text2vec-base-chinese'
REV='183bb99aa7af74355fb58d16edf8c13ae7c5433e'
FILES=['config.json','model.safetensors','vocab.txt','tokenizer_config.json','special_tokens_map.json']


def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def dump(p,obj):
    data=(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n').encode('utf8')
    if p.exists():
        if p.read_bytes()!=data:raise ValueError('Different existing artifact: '+str(p))
        return
    p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_name(p.name+'.tmp');temp.write_bytes(data);temp.replace(p)


def normalized(x):
    import numpy as np
    norms=np.linalg.norm(x,axis=1,keepdims=True)
    if not np.isfinite(x).all() or (norms<=0).any():raise ValueError('Invalid embedding')
    return (x/norms).astype('float32')


def ranking(x):
    import numpy as np
    scores=np.clip(x @ x.T,-1,1)
    order=np.argsort(-scores,axis=1,kind='stable').astype('int32')
    return order,np.take_along_axis(scores,order,axis=1)


def download(root):
    from huggingface_hub import hf_hub_download
    folder=root/'models/text2vec-base-chinese'/REV
    folder.mkdir(parents=True,exist_ok=True)
    for name in FILES:
        hf_hub_download(repo_id=REPO,revision=REV,filename=name,local_dir=str(folder))
        print('MODEL_FILE_READY',name,flush=True)
    dump(folder/'offline_manifest.json',{'repo_id':REPO,'revision':REV,
         'files_sha256':{n:digest(folder/n) for n in FILES}})
    print('EMBEDDING_MODEL_READY',folder,flush=True)


def execute(root):
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import numpy as np
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModel
    import fcntl
    modeldir=root/'models/text2vec-base-chinese'/REV
    model_manifest=json.loads((modeldir/'offline_manifest.json').read_text(encoding='utf8'))
    assert model_manifest['repo_id']==REPO and model_manifest['revision']==REV
    for name in FILES:
        if digest(modeldir/name)!=model_manifest['files_sha256'][name]:raise ValueError('Model file changed: '+name)
    base=root/'paper_method_v1';previous=base/'artifacts/retrieval_v1'
    previous_summary=json.loads((previous/'summary.json').read_text(encoding='utf8'))
    assert digest(previous/'index.json')==previous_summary['output_sha256']['index.json']
    graphpath=base/'artifacts/full_graph_v1/graph.json'
    assert digest(graphpath)==previous_summary['graph_sha256']
    graph=json.loads(graphpath.read_text(encoding='utf8'))
    index=json.loads((previous/'index.json').read_text(encoding='utf8'))
    summaries=sorted([n for n in graph['nodes'] if n['kind']=='w'],key=lambda n:n['id'])
    out=base/'artifacts/eq10_v1';out.mkdir(parents=True,exist_ok=True)
    with (out/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cfg={'model':REPO,'revision':REV,'model_manifest_sha256':digest(modeldir/'offline_manifest.json'),
             'graph_sha256':digest(graphpath),'index_sha256':digest(previous/'index.json'),
             'script_sha256':digest(Path(__file__)),'torch':torch.__version__,'transformers':transformers.__version__,
             'numpy':np.__version__,'device':'cpu','threads':4,'batch_size':8,'chunk_size':64,
             'pooling':'attention-mask mean including special tokens, per model card','normalize':'L2',
             'truncation':False,'checkpoint_choice':'text2vec default Chinese checkpoint; not specified in paper',
             'top_k':None,'ranking_scope':'all Tr candidates; equals T under equations 8-9; self retained',
             'tie_break':'ascending node ID','offline':True}
        dump(out/'config.json',cfg)
        dump(out/'summary_nodes.json',summaries)
        torch.set_num_threads(4);torch.manual_seed(42)
        chunks=out/'chunks';chunks.mkdir(exist_ok=True)
        model=None;tokenizer=None;vectors=[]
        for start in range(0,len(summaries),64):
            block=summaries[start:start+64];path=chunks/('%06d.npy'%start);meta=path.with_suffix('.json')
            if path.exists() and meta.exists():
                saved=json.loads(meta.read_text())
                assert saved['sha256']==digest(path) and saved['ids']==[x['id'] for x in block],'Chunk changed'
                vec=np.load(path,allow_pickle=False)
            else:
                if model is None:
                    tokenizer=AutoTokenizer.from_pretrained(str(modeldir),local_files_only=True,trust_remote_code=False)
                    model=AutoModel.from_pretrained(str(modeldir),local_files_only=True,trust_remote_code=False,use_safetensors=True).eval()
                pieces=[]
                for offset in range(0,len(block),8):
                    texts=[x['text'] for x in block[offset:offset+8]]
                    inputs=tokenizer(texts,padding=True,truncation=False,return_tensors='pt')
                    if inputs['input_ids'].shape[1]>model.config.max_position_embeddings:
                        raise ValueError('Summary exceeds model length; refusing silent truncation at chunk '+str(start))
                    with torch.inference_mode():
                        hidden=model(**inputs).last_hidden_state
                        mask=inputs['attention_mask'].unsqueeze(-1).to(hidden.dtype)
                        pooled=(hidden*mask).sum(1)/mask.sum(1).clamp(min=1)
                    pieces.append(normalized(pooled.cpu().numpy()))
                vec=np.concatenate(pieces)
                temp=path.with_suffix('.tmp')
                with temp.open('wb') as f:np.save(f,vec,allow_pickle=False)
                temp.replace(path)
                dump(meta,{'ids':[x['id'] for x in block],'sha256':digest(path)})
            assert vec.shape[0]==len(block) and np.isfinite(vec).all()
            assert np.allclose(np.linalg.norm(vec,axis=1),1,atol=1e-5)
            vectors.append(vec);print('EMBEDDED',start+len(block),'/',len(summaries),flush=True)
        matrix=np.concatenate(vectors);rowmap={n['id']:i for i,n in enumerate(summaries)}
        rankdir=out/'rankings';rankdir.mkdir(exist_ok=True);artifacts=[]
        for c,ids in sorted(index['summaries_by_type'].items()):
            ids=sorted(ids);path=rankdir/(c.split(':',1)[1]+'.npz');meta=path.with_suffix('.json')
            if path.exists() and meta.exists():
                data=json.loads(meta.read_text());assert data['sha256']==digest(path)
                assert data['summary_ids']==ids
            else:
                vec=matrix[[rowmap[w] for w in ids]];order,scores=ranking(vec)
                temp=path.with_suffix('.tmp')
                with temp.open('wb') as f:np.savez_compressed(f,order=order,scores=scores)
                temp.replace(path)
                data={'type_id':c,'summary_ids':ids,'sha256':digest(path),
                      'format':'Row i queries summary_ids[i]; order[i,j] indexes summary_ids for rank j; scores[i,j] is cosine similarity.'}
                dump(meta,data)
            artifacts.append({'type_id':c,'candidates':len(ids),'file':str(path.relative_to(out)),'sha256':digest(path)})
            print('RANKED_TYPE',c,len(ids),flush=True)
        report={'status':'PAPER_EQ10_RANKING_COMPLETE','summary_nodes':len(summaries),'embedding_dimension':int(matrix.shape[1]),
                'entity_types':len(artifacts),'rankings':artifacts,'config_sha256':digest(out/'config.json'),
                'embedding_ranking_complete':True,'top_k_selected':False,'replacement_applied':False,
                'api_calls':0,'training_started':False,'test_accessed':False,'offline':True}
        dump(out/'summary.json',report)
        print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['download','run'])
    p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    if args.command=='download':download(args.root.resolve())
    else:execute(args.root.resolve())


if __name__=='__main__':main()
