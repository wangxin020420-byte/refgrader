"""Create offline image/text review pages; never approves or changes source data."""
import argparse
import base64
import hashlib
import html
import json
from pathlib import Path


def sha(b):
    return hashlib.sha256(b).hexdigest()


def inventory(report):
    result = []
    for r in report['comparisons']:
        result.append(dict(unit_id=r['unit_id'], image=r['image'], group='supplement',
            original=r['original_text'], previous=r.get('previous_transcription') or '',
            proposed=r['supplement_transcription'], issues=r['supplement_issues'],
            earlier_issues=r.get('previous_issues', [])))
    for r in report['remaining_old_issues']:
        result.append(dict(unit_id=r['unit_id'], image=r['image'], group='old_issue',
            original=r['original_text'], previous=r['transcription'], proposed=r['transcription'],
            issues=r['issues'], earlier_issues=[]))
    for r in report['manual_1301']:
        result.append(dict(unit_id=r['unit_id'], image=r['image'], group='manual_1301',
            original=r['original_text'], previous='', proposed=r.get('manual_transcription', ''),
            issues=['人工核对原图并完整转写；不调用服务端重试。'], earlier_issues=[]))
    if len({r['unit_id'] for r in result}) != len(result):
        raise ValueError('Duplicate review IDs')
    return result


TEMPLATE = r'''<!doctype html><html lang="zh"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CSBench 原图对照审核</title>
<style>body{font:16px sans-serif;margin:24px;background:#f5f7fa;color:#172333}h2{font-size:20px}
header{position:sticky;top:0;background:#fff;padding:14px;border-bottom:2px solid #2454a6;z-index:1}
article{background:white;margin:24px 0;padding:20px;border:1px solid #bbb;border-radius:8px}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:20px}.picture{max-height:760px;overflow:auto}
img{max-width:100%;cursor:zoom-in}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f5f8;padding:12px}
textarea{box-sizing:border-box;width:100%;min-height:280px;font:16px monospace}.notes{min-height:65px}
button,select,input{font-size:16px;padding:8px;margin:4px}button{cursor:pointer}label{display:block;margin-top:12px}
.warn{color:#9b3419}.status{font-weight:bold}@media(max-width:900px){.cols{grid-template-columns:1fr}}</style>
<header><b>CSBench 原图对照审核 · PAGE_NAME</b><br>
审核者：<input id="reviewer" placeholder="填写实际审核者">
<button id="save">下载本页审核 JSON</button>
<label>继续上次审核：<input id="restore" type="file" accept=".json"></label>
<span id="count"></span><div class="warn">不会自动保存。关闭页面前请下载；下载后可重新导入继续。没有任何联网请求。</div></header>
<p>原样保留学生错误、遗漏与矛盾；只转写可见内容，不根据题意补答案。图中未画箭头时不要推断方向。
省略号若确实是学生所写则保留，否则不能代替未转录内容。教师批改与学生涂改须区分，不确定则留待处理。</p>
<p>选择“已核对”之前请查看原图并确认右侧最终文本。无法确认时选择“待处理”，不要猜测。
最终文本只包含学生作答及中性的图表说明；审核解释填写在备注栏。</p>
<main id="rows"></main><script>
const DATA=PAYLOAD;
function el(tag,text,parent){const e=document.createElement(tag);if(text!==null)e.textContent=text;if(parent)parent.append(e);return e;}
const controls=[];
for(const r of DATA.records){
 const a=el('article',null,document.getElementById('rows'));el('h2',r.unit_id+' · '+r.group,a);
 const cols=el('div',null,a);cols.className='cols';const left=el('div',null,cols);const box=el('div',null,left);box.className='picture';
 const img=el('img',null,box);img.src=r.image_data;img.alt=r.unit_id;
 img.onclick=()=>{img.style.maxWidth=img.style.maxWidth==='none'?'100%':'none';};
 el('p','点击图片切换原尺寸；可在图片区域滚动查看。',left);
 el('p',r.image,left);const right=el('div',null,cols);
 for(const [title,key] of [['原有文字','original'],['先前转写','previous']]){const d=el('details',null,right);el('summary',title,d);el('pre',r[key],d);}
 el('p','本轮疑问：'+JSON.stringify(r.issues),right);el('p','此前疑问：'+JSON.stringify(r.earlier_issues),right);
 el('label','最终完整转写（可编辑）',right);const text=el('textarea',null,right);text.value=r.proposed;
 const b=el('button','恢复本页候选文字',right);b.onclick=()=>{if(confirm('替换当前编辑内容？'))text.value=r.proposed;};
 el('label','审核状态',right);const state=el('select',null,right);
 for(const [value,label] of [['PENDING','待核对'],['REVIEWED','已对照原图核对'],['UNRESOLVED','待处理：仍无法确认']]){const o=el('option',label,state);o.value=value;}
 el('label','审核备注（疑点、修改位置或无法辨认处）',right);const notes=el('textarea',null,right);notes.className='notes';
 controls.push({r,text,state,notes});state.onchange=update;
}
function update(){document.getElementById('count').textContent='已核对 '+controls.filter(c=>c.state.value==='REVIEWED').length+' / '+controls.length;}
update();
document.getElementById('save').onclick=()=>{
 const reviewer=document.getElementById('reviewer').value.trim();
 for(const c of controls){if(c.state.value==='REVIEWED'&&(!reviewer||!c.text.value.trim())){alert('已核对条目必须有审核者和完整文本：'+c.r.unit_id);return;}}
 const output={schema:'csbench_visual_human_review_v1',batch_sha256:DATA.batch_sha256,page:DATA.page,reviewer,
  reviewed_at:new Date().toISOString(),decisions:controls.map(c=>({unit_id:c.r.unit_id,image_sha256:c.r.image_sha256,
  candidate_sha256:c.r.candidate_sha256,status:c.state.value,final_text:c.text.value,notes:c.notes.value}))};
 const url=URL.createObjectURL(new Blob([JSON.stringify(output,null,2)],{type:'application/json'}));
 const a=document.createElement('a');a.href=url;a.download=DATA.page+'.decisions.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
};
document.getElementById('restore').onchange=async e=>{try{
 const s=JSON.parse(await e.target.files[0].text());
 if(s.schema!=='csbench_visual_human_review_v1'||s.batch_sha256!==DATA.batch_sha256||s.page!==DATA.page)throw Error('文件不属于本页');
 const map=new Map(s.decisions.map(x=>[x.unit_id,x]));if(map.size!==controls.length||s.decisions.length!==controls.length)throw Error('条目数不匹配');
 for(const c of controls){const d=map.get(c.r.unit_id);if(!d||d.image_sha256!==c.r.image_sha256||d.candidate_sha256!==c.r.candidate_sha256||!['PENDING','REVIEWED','UNRESOLVED'].includes(d.status)||typeof d.final_text!=='string'||typeof d.notes!=='string')throw Error('条目校验失败');}
 for(const c of controls){const d=map.get(c.r.unit_id);c.text.value=d.final_text;c.state.value=d.status;c.notes.value=d.notes;}
 document.getElementById('reviewer').value=s.reviewer||'';update();
 }catch(err){alert('导入失败：'+err.message);}};
</script></html>'''


def build(report_path, out, per_page=10):
    raw = report_path.read_bytes()
    report = json.loads(raw.decode('utf-8'))
    records = inventory(report)
    if out.exists():
        raise ValueError('Output exists; keep existing review work')
    images = {}
    for r in records:
        b = Path(r['image']).read_bytes()
        mime = 'image/jpeg' if b.startswith(b'\xff\xd8\xff') else 'image/png' if b.startswith(b'\x89PNG\r\n\x1a\n') else None
        if not mime:
            raise ValueError('Unsupported image: ' + r['image'])
        r['image_sha256'] = sha(b)
        r['candidate_sha256'] = sha(r['proposed'].encode('utf-8'))
        images[r['unit_id']] = 'data:'+mime+';base64,'+base64.b64encode(b).decode('ascii')
    out.mkdir(parents=True)
    manifest = dict(schema='csbench_visual_review_pages_v1',report_sha256=sha(raw),records=records,
                    api_calls=0,labels_read=False,source_files_modified=False,ready_for_training=False)
    batch = sha(json.dumps(manifest,sort_keys=True,ensure_ascii=False).encode())
    manifest['batch_sha256'] = batch
    pages=[]
    for i in range(0,len(records),per_page):
        name=f'review_{i//per_page+1:02d}'
        payload=dict(batch_sha256=batch,page=name,records=[dict(r,image_data=images[r['unit_id']]) for r in records[i:i+per_page]])
        js=json.dumps(payload,ensure_ascii=False).replace('<','\\u003c').replace('\u2028','\\u2028').replace('\u2029','\\u2029')
        page=TEMPLATE.replace('PAGE_NAME',name).replace('PAYLOAD',js)
        (out/(name+'.html')).write_text(page,encoding='utf-8')
        pages.append(name+'.html')
    manifest['pages']=pages
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>CSBench 原图审核</h1><p>每页10条；按页下载审核结果。图片已嵌入，无需连接服务器。</p>'+''.join('<p><a href="'+p+'">'+html.escape(p)+'</a></p>' for p in pages),encoding='utf-8')
    print(json.dumps(dict(review_records=len(records),pages=len(pages),output=str(out),api_calls=0,ready_for_training=False),ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    build(args.report,args.out)
