"""Targeted image recheck proposals, isolated from original accepted results."""
import argparse
import importlib.util
import json
from pathlib import Path

BASE_HASH = '17da94d1984acc0829a09de77939c8132254f4a834eb711cfaab67d56559ce8d'
PROMPT = '''你是忠实图像转写员，不是解题或评分助手。仅转写本次给出的图片。
已有文字可能用“如图/如表所示”代替图形内容，不能仅复述已有文字。
逐个检查各小题对应的图、表、竖式、程序、时序线或树。输出完整作答文字，在原位置插入【图表转录：位置】描述。
表格保留表头、行列与单元格；时序图保留可见时间点、线段和切换顺序；图与树保留节点、边、方向、标签和层次；竖式保留行次和对齐含义。
只能描述图片实际可见的内容。不能依据题意或常识补出正确图形，不能纠正原有算式、数字或结论。
若某处只有原作者的“如图所示”文字，实际未见图，写【未见对应图表：位置】，并在issues中说明。
模糊部分写【无法辨认：位置】并在issues说明；不能猜测。明确的教师分数和批改标记不作为学生答案；无法区分则记录疑问。
已有文字只用于定位，以图片为依据；不得因文字中的“表中”等措辞虚构学生画了表。
仅返回JSON：{"text":"完整转写及就地展开的图表内容", "issues":[]}。'''


def select(details):
    excluded = {r['unit_id'] for r in details['manual_1301']}
    selected = {}
    for category in ('uncovered_visual_cues', 'unchanged_visual_placeholder_records', 'unreadable_records'):
        for r in details[category]:
            uid = r['unit_id']
            if uid in excluded:
                raise ValueError('Blocked 1301 unit must not be requested')
            if uid in selected and any(selected[uid][k] != r[k] for k in ('image', 'original_text')):
                raise ValueError('Conflicting request sources')
            selected[uid] = r
    return selected


def main():
    a = argparse.ArgumentParser()
    a.add_argument('command', choices=['plan', 'run'])
    a.add_argument('--base-runner', type=Path, required=True)
    a.add_argument('--details', type=Path, required=True)
    a.add_argument('--inputs', type=Path, required=True)
    a.add_argument('--original-run', type=Path, required=True)
    a.add_argument('--out', type=Path, required=True)
    a.add_argument('--attempts', type=int, default=8)
    args = a.parse_args()
    if args.attempts < 1:
        a.error('attempts must be positive')
    import hashlib
    if hashlib.sha256(args.base_runner.read_bytes()).hexdigest() != BASE_HASH:
        raise ValueError('Base runner hash mismatch')
    spec = importlib.util.spec_from_file_location('verified_transcriber', args.base_runner)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    for path in (args.inputs, args.original_run):
        if args.out.resolve() == path.resolve() or path.resolve() in args.out.resolve().parents:
            raise ValueError('Supplement output must be isolated')
    p = m.plan(args.inputs, args.original_run)
    details = m.read(args.details)
    chosen = select(details)
    originals = {u['unit_id']:u for u in p['units']}
    source = {}
    for split in ('train', 'validation', 'test'):
        for line in (args.inputs / (split + '.inputs.jsonl')).read_text(encoding='utf-8').splitlines():
            r = json.loads(line)
            source['answer:' + r['sample_id']] = r
    new_units = []
    for uid, r in sorted(chosen.items()):
        old = originals.get(uid)
        if old:
            if old['image'] != r['image'] or old['original_text'] != r['original_text']:
                raise ValueError('Detail/original plan mismatch: ' + uid)
            folder = args.original_run / 'responses' / m.sha(uid.encode())
            if any(str(m.read(f).get('service_code')) == '1301' for f in folder.glob('*.error.json')):
                raise ValueError('Historical service rejection requires manual review: ' + uid)
        else:
            s = source[uid]
            if s['blank_passthrough'] or s['answer'] != r['original_text'] or s['student_image'] != r['image']:
                raise ValueError('Uncovered unit/source mismatch')
        raw, mime = m.image_data(r['image'])
        if old and m.sha(raw) != old['image_sha256']:
            raise ValueError('Original image changed')
        new_units.append(dict(unit_id=uid, kind=old['kind'] if old else 'student_answer',
                              image=r['image'], original_text=r['original_text'], mime=mime, image_sha256=m.sha(raw)))
    supplement = dict(p)
    supplement.update(units=new_units, prompt=PROMPT, missing_visual_context=[],
        parent_plan_sha256=m.sha(m.encoded(p)), details_sha256=m.sha(args.details.read_bytes()),
        policy='Targeted recheck proposals only; no automatic replacement of prior transcriptions')
    args.out.mkdir(parents=True, exist_ok=True)
    plan_path = args.out / 'plan.json'
    if plan_path.exists() and plan_path.read_bytes() != m.encoded(supplement):
        raise ValueError('Supplement plan changed; do not mix caches')
    if not plan_path.exists():
        m.write(plan_path, supplement)
    # No need to invent missing images where a source field was never configured.
    context = [dict(r, assessment='TEXT_SOURCE_PRESENT_NO_IMAGE_CONFIGURED' if r.get('configured_path') is None else 'PATH_REVIEW_REQUIRED',
                    note='Administrative source assessment only; does not certify answer correctness or alter original data')
               for r in details['missing_visual_context']]
    m.write(args.out / 'context_assessment.json', context)
    print('SUPPLEMENT_PLAN_READY', len(new_units), flush=True)
    print('OUTPUT', args.out, flush=True)
    if args.command == 'plan':
        return 0
    m.plan = lambda inputs, out: supplement
    rc = m.run(args.inputs, args.out, 0, args.attempts)
    s = m.read(args.out / 'summary.json')
    s.update(supplement_runner_sha256=m.sha(Path(__file__).read_bytes()),
             original_run_modified=False, proposals_only=True)
    m.write(args.out / 'summary.json', s)
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
