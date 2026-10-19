"""Apply six reviewed, hash-bound source-span alignments without network access."""
import argparse
import copy
import json
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--execute',action='store_true')
    args=p.parse_args();root=args.root.resolve()
    sys.path.insert(0,str(root/'control/scale-paper-stage1'))
    import stage1 as m
    import fcntl
    run=root/'paper_method_v1/runs/pe_glm_full_v1'
    audit=run/'source_alignment_v1'
    plan_path=Path(__file__).with_name('plan.json')
    plan=json.loads(plan_path.read_text(encoding='utf8'))
    with (run/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cfg=json.loads((run/'config.json').read_text(encoding='utf8'))
        source=root/'paper_method_v1/artifacts/triplets/source_units.jsonl'
        assert m.hashed(source.read_bytes())==cfg['source_sha256'],'Training source changed'
        assert m.hashed(Path(m.__file__).read_bytes())==cfg['stage1_sha256'],'Validator changed'
        units={u['unit_id']:u for u in m.read(source)}
        staged=[]
        for item in plan['items']:
            uid=item['unit_id'];unit=units[uid];stem=m.hashed(uid.encode('utf8'))
            assert m.hashed(unit['text'].encode('utf8'))==item['source_sha256'],'Source text changed'
            target=run/'cache'/(stem+'.json');slot=audit/stem
            current=target.read_bytes()
            if (slot/'accepted_cache.json').exists() and current==(slot/'accepted_cache.json').read_bytes():
                print(uid,'ALREADY_APPLIED');continue
            wrapper=json.loads(current);assert wrapper['unit_id']==uid
            try:m.validate([unit],[{'unit_id':uid,'triples':json.loads(wrapper['response']['choices'][0]['message']['content'])['triples']}])
            except (ValueError,KeyError,TypeError,IndexError):pass
            else:
                if wrapper['response']['choices'][0].get('finish_reason')=='stop':
                    raise ValueError('Target already valid; refusing to change '+uid)
            chosen=(root/item['response_file']).resolve()
            assert run.resolve() in chosen.parents,'Response outside run'
            chosen_bytes=chosen.read_bytes();saved=json.loads(chosen_bytes)
            raw=copy.deepcopy(saved['response'] if 'response' in saved else saved)
            choice=raw['choices'][0];content=choice['message']['content']
            assert choice['finish_reason']=='stop'
            assert m.hashed(content.encode('utf8'))==item['content_sha256'],'Response changed'
            obj=json.loads(content)
            for edit in item['edits']:
                triple=obj['triples'][edit['index']]
                assert triple['t']==edit['before']
                assert edit['after'] in unit['text']
                triple['t']=edit['after']
            m.validate([unit],[{'unit_id':uid,'triples':obj['triples']}])
            choice['message']['content']=json.dumps(obj,ensure_ascii=False)
            result={'unit_id':uid,'origin':'source_alignment_v1/'+stem,'response':raw,
                    'alignment':item,'note':'Reviewed extraction postprocessing; original API response retained in audit.'}
            staged.append((uid,target,slot,current,m.encoded(result),chosen_bytes))
        print('ALIGNMENT_PREFLIGHT_OK; pending='+str(len(staged)))
        if not args.execute:return
        m.write_all(audit,{'plan.json':plan_path.read_bytes()})
        for uid,target,slot,current,result,chosen_bytes in staged:
            m.write_all(slot,{'original_cache.json':current,'selected_response.json':chosen_bytes,'accepted_cache.json':result})
            assert target.read_bytes()==current,'Cache changed during installation'
            tmp=slot/'install.tmp';tmp.write_bytes(result);tmp.replace(target)
            print(uid,'ALIGNED')
        print('ALIGNMENT_COMPLETE; run full extractor --mode offline to build the graph.')


if __name__=='__main__':main()
