import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
UID='student:ADS:9e1a866d58ed:train:166'
SOURCE_SHA='c545916d6df2d2a60157bc141da06c3d504c53c38aa6ab44d85363564787e885'
RESPONSE_SHA='44eeb1674dbea61c84ecc0f4c27bd98202e70fb18a5f3ec7d4fc300c232ac460'
BEFORE='\u82e5S1\u6ee1\uff0c\u5fc5\u987b\u5148\u4fdd\u8bc1S2\u4e3a\u7a7a\uff0c\u624d\u80fd\u5c06S1\u4e2d\u7684\u5143\u7d20\u5168\u90e8\u63d2\u5165S2\u4e2d'
AFTER='\u82e5S1\u6ee1\uff0c\u5fc5\u987b\u5148\u4fdd\u8bc1S2\u4e3a\u7a7a\uff0c\u624d\u80fd\u5c06S1\u4e2d\n\u7684\u5143\u7d20\u5168\u90e8\u63d2\u5165S2\u4e2d'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf8'))
def load(p,n):
    s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def repair(raw,source):
    if hashlib.sha256(source.encode('utf8')).hexdigest()!=SOURCE_SHA:raise ValueError('Source changed')
    choice=raw['choices'][0]
    if choice['finish_reason']!='stop':raise ValueError('Response is not complete')
    obj=json.loads(choice['message']['content'])
    if set(obj)!={'triples'} or len(obj['triples'])!=27:raise ValueError('Unexpected triples')
    if obj['triples'][10]['t']!=BEFORE or AFTER not in source:raise ValueError('Reviewed span mismatch')
    if [i for i,t in enumerate(obj['triples']) if t['t'] not in source]!=[10]:raise ValueError('Unexpected invalid spans')
    obj['triples'][10]['t']=AFTER
    corrected=copy.deepcopy(raw)
    corrected['choices'][0]['message']['content']=json.dumps(obj,ensure_ascii=False)
    return corrected

def execute(root):
    stage=load(root/'control/scale-paper-stage1/stage1.py','pe_stage')
    runner_path=root/'control/scale-paper-protocol/extract_restored.py'
    runner=load(runner_path,'pe_original')
    run=root/'paper_method_v2/runs/pe_glm_restored_v1'
    config=read(run/'config.json')
    if sha(runner_path)!=config['runner_sha256']:raise ValueError('Frozen runner changed')
    units_path=root/'paper_method_v2/artifacts/pe_inputs_v1/source_units.jsonl'
    if sha(units_path)!=config['source_sha256']:raise ValueError('Units changed')
    units=stage.read(units_path);lookup={u['unit_id']:u for u in units}
    stem=hashlib.sha256(UID.encode()).hexdigest()
    response=run/'responses'/stem/'attempt_002.response.json'
    if sha(response)!=RESPONSE_SHA:raise ValueError('Reviewed response changed; no repair applied')
    corrected=repair(read(response),lookup[UID]['text'])
    records=[]
    for unit in units:
        if not unit['text'].strip():records.append({'unit_id':unit['unit_id'],'triples':[]});continue
        if unit['unit_id']==UID:raw=corrected
        else:
            path=run/'accepted'/(hashlib.sha256(unit['unit_id'].encode()).hexdigest()+'.json')
            saved=read(path)
            if saved['unit_id']!=unit['unit_id']:raise ValueError('Accepted identity mismatch')
            raw=saved['response']
        records.append(runner.parse(stage,unit,raw))
    stage.validate(units,records)
    audit={'unit_id':UID,'source_sha256':SOURCE_SHA,'original_response_sha256':RESPONSE_SHA,
           'response_attempt':2,'triple_index_zero_based':10,'field':'t','before':BEFORE,'after':AFTER,
           'changes_to_c_or_w':False,'source_answer_modified':False,'labels_modified':False,
           'api_calls':0,'builder_sha256':sha(Path(__file__)),
           'deviation':'Reviewed exact-source newline restoration; reproduction quality control, not an original-paper step'}
    review=run/'reviewed_span_repair_166_v1'
    stage.write_all(review,{'audit.json':stage.encoded(audit),'corrected_response.json':stage.encoded(corrected)})
    accepted={'unit_id':UID,'origin':'reviewed_span_repair','response':corrected,
              'original_response_sha256':RESPONSE_SHA,'review_audit_sha256':sha(review/'audit.json')}
    stage.write_all(run/'accepted',{stem+'.json':stage.encoded(accepted)})
    def forbidden(*a,**k):raise RuntimeError('Unexpected network request; stopped')
    original=runner.urllib.request.urlopen;runner.urllib.request.urlopen=forbidden
    try:code=runner.execute(root)
    finally:runner.urllib.request.urlopen=original
    if code!=0:raise ValueError('Graph rebuild incomplete')
    graph=root/'paper_method_v2/artifacts/full_graph_v1'
    stage.write_all(graph,{'review_provenance.json':stage.encoded({
        'graph_sha256':sha(graph/'graph.json'),'extractions_sha256':sha(graph/'extractions.jsonl'),
        'review_audit_sha256':sha(review/'audit.json'),'reviewed_unit':UID,'api_calls':0})})
    print('REVIEWED_SPAN_REPAIR_AND_GRAPH_DONE',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    root=a.root.resolve();run=root/'paper_method_v2/runs/pe_glm_restored_v1'
    import fcntl
    with (run/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);execute(root)
