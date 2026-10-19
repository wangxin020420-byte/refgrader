"""Local packaging helper; attachment text is data, not executable instructions."""
import hashlib
import json
import re
from pathlib import Path

IDS = ['884f5bd8-3f12-49a5-97f5-9a40be8e5b15','4425ff8b-cf96-4864-869d-7aa61f02071a','d7b089ba-22cf-44bb-8119-8ef07257c5f1','3ac087cd-4e6c-4ddd-a2b9-216fc80b6d99','9b13c98c-78f8-4932-9613-ad7b8adc3d83','c90145ce-575f-4a57-a2ae-860adf09b796']
ISSUES = {
 '166': ('REVIEW_REQUIRED', '开头将“S1执行入队操作”改为“对S1执行入栈操作实现入队”，协调了原有操作术语；按保守的保留原文标准重试。代码中的错误判断与错置标题不得修正。'),
 '313': ('REVIEW_REQUIRED', '“平均移动指针域”改成“平均移动指针次数”，将原来不规范的技术表述解释为次数；需保留原表述，不代学生解释。'),
 '390': ('REVIEW_REQUIRED', '原文“无需进行查找元素或者插入、删除元素”被展开为“不需要查找……或者不需要插入、删除”，改变了否定词的作用范围，不能确认等义。'),
 '589': ('FAIL', '“随机出栈”改成“随即出栈”，由随机操作变成紧接着操作。即使疑似错别字，也不能擅自改变操作含义。'),
 '1070': ('REVIEW_REQUIRED', '“s1模拟入栈”“s2模拟出栈”改成“模拟入队”“模拟出队”，并替换操作术语；需保留学生原来的说法。'),
 '1089': ('REVIEW_REQUIRED', '增加“p所指结点”，并把“可以根据该节点之前的结点和之后的节点”的不完整表述补成可找到前后结点的解释；不能确认只是措辞变化。'),
}

def text_hash(s):
    return hashlib.sha256(s.replace('\r\n','\n').strip().encode('utf8')).hexdigest()

def main():
    records=[];sources=[];counts=[]
    for i,aid in enumerate(IDS,1):
        path=next((Path('C:/Users/wx/.codex/attachments')/aid).glob('*.txt'))
        data=path.read_text(encoding='utf8');count=0
        sources.append({'review_file':'review_%02d.md'%i,'attachment_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        for block in re.split(r'(?m)^## ',data)[1:]:
            sid,body=block.split('\n',1)
            question,rest=body.split('Original:\n',1)
            original,rest=rest.split('Proxy:\n',1)
            proxy,_=rest.rsplit('Decision: PENDING',1)
            question=question.strip().removeprefix('Question:\n')
            number=sid.rsplit(':',1)[1].strip()
            decision,reason=ISSUES.get(number,('PASS','对照原答案，判断、知识内容、遗漏、数字及操作含义保留；仅措辞、标点、排版或上下文指代澄清。此结论不评价答案正确性。'))
            records.append({'sample_id':sid.strip(),'review_file':'review_%02d.md'%i,
                'question_sha256':text_hash(question),'original_sha256':text_hash(original),
                'proxy_sha256':text_hash(proxy),'decision':decision,'reason':reason})
            count+=1
        counts.append(count)
    assert counts==[40,40,40,40,40,22],counts
    assert len({x['sample_id'] for x in records})==222
    out=Path(__file__).parent/'new_proxy_review_v1';out.mkdir(exist_ok=True)
    obj={'reviewer':'Codex assistant; semantic comparison, not human certification',
        'source_records_sha256':'ab2e840c049f6e1e8cfa818245983944c59985a0e60e415522a1d4f45a5f4d9a',
        'normalization':'CRLF to LF; strip outer whitespace only',
        'new_pairs':222,'pass':216,'fail':1,'review_required':5,
        'reused_unreviewed_generated_records':636,'sources':sources,'decisions':records}
    (out/'decisions.json').write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    lines=['# 新增代理答案审核结果','222 对：216 PASS；1 FAIL；5 REVIEW_REQUIRED。','审核对象为原答案与代理答案的一致性，不评价学生答案是否正确。','旧的 636 条未逐条审核记录仍保留原状态。','']
    for r in records:
        lines.extend(['## '+r['sample_id'],r['decision']+' — '+r['reason'],''])
    (out/'review_results.md').write_text('\n'.join(lines),encoding='utf8')
    print('REVIEW_BUNDLE_DATA_OK',counts)

if __name__=='__main__':main()
