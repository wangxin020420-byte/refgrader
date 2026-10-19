import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import time

from PIL import Image

BASE = 'https://open.bigmodel.cn/api/coding/paas/v4/'
PROMPT = '''忠实转录图片中学生已写出的作答，不解题、不评分、不纠错、不根据规律补全。输出严格JSON：{"text":"...","issues":[]}。真值表按图片顺序逐行转录A、B、C、D、F全部16行，保留学生实际写出的值；不清楚的值写[无法辨认]并在issues说明行号。保留可见涂改及不确定性。若有表格外的学生文字说明、卡诺图或公式，也应完整转录其位置、内容与结构。空白遮蔽区域无需描述。仅输出JSON，不要Markdown围栏。'''


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def masked(image, boxes):
    image = image.convert('RGB')
    assert image.size == (966, 614), image.size
    result = Image.new('RGB', image.size, 'white')
    for box in boxes:
        result.paste(image.crop(box), box[:2])
    return result


def parse(content):
    obj = json.loads(content)
    assert set(obj) == {'text', 'issues'}
    assert isinstance(obj['text'], str) and obj['text'].strip()
    assert isinstance(obj['issues'], list)
    assert all(isinstance(x, str) for x in obj['issues'])
    assert not any(x in obj['text'] for x in ('评卷员', '评分：', '评分:'))
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', type=Path, required=True)
    ap.add_argument('--run', action='store_true')
    args = ap.parse_args()
    policy = json.loads(Path(__file__).with_name('policy.json').read_text())
    assert len(policy) == len({r['sample_id'] for r in policy}) == 56
    out = args.project / 'runs/online_experiments/csbench_scale_online_clean_v2/ml1_clean_transcription'
    for sub in ('images', 'responses', 'accepted'):
        (out / sub).mkdir(parents=True, exist_ok=True)
    plan = []
    for row in policy:
        src = Path(row['path'])
        assert digest(src.read_bytes()) == row['image_sha256'], str(src)
        with Image.open(src) as im:
            clean = masked(im, row['keep_rectangles'])
        target = out / 'images' / (row['sample_id'] + '.png')
        if target.exists():
            with Image.open(target) as old:
                assert old.convert('RGB').tobytes() == clean.tobytes(), str(target)
        else:
            clean.save(target)
        plan.append(dict(row, clean_path=str(target), clean_sha256=digest(target.read_bytes())))
    config = {'model': 'glm-4.6v', 'base_url': BASE, 'prompt': PROMPT, 'max_tokens': 4096,
              'requests': plan, 'policy_sha256': digest(Path(__file__).with_name('policy.json').read_bytes())}
    config_hash = digest(json.dumps(config, sort_keys=True).encode())
    planfile = out / 'plan.json'
    if planfile.exists():
        assert json.loads(planfile.read_text(encoding='utf-8')) == config, 'Plan changed'
    else:
        save(planfile, config)
    print('MASKED_IMAGES_READY 56; original images unchanged', flush=True)
    client = None
    if args.run:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ['ZHIPUAI_API_KEY'], base_url=BASE, timeout=180, max_retries=0)
    calls = 0
    records, errors = [], []
    for row in plan:
        uid = row['sample_id']
        accepted = out / 'accepted' / (uid + '.json')
        if accepted.exists():
            envelope = json.loads(accepted.read_text(encoding='utf-8'))
            assert envelope['config_sha256'] == config_hash
            response_file = out / 'responses' / envelope['response_file']
            assert digest(response_file.read_bytes()) == envelope['response_sha256']
            raw = json.loads(response_file.read_text(encoding='utf-8'))
            assert raw['config_sha256'] == config_hash
            assert parse(raw['response']['choices'][0]['message']['content']) == envelope['result']
        elif client:
            for attempt in range(1, 4):
                try:
                    image_bytes = Path(row['clean_path']).read_bytes()
                    assert digest(image_bytes) == row['clean_sha256']
                    calls += 1
                    response = client.chat.completions.create(model=config['model'], max_tokens=4096,
                        messages=[{'role': 'user', 'content': [
                            {'type': 'text', 'text': PROMPT},
                            {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(image_bytes).decode()}}
                        ]}])
                    name = uid + '_' + str(time.time_ns()) + '.json'
                    response_file = out / 'responses' / name
                    save(response_file, {'config_sha256': config_hash, 'sample_id': uid, 'response': response.model_dump()})
                    assert response.choices[0].finish_reason == 'stop', 'Incomplete response'
                    result = parse(response.choices[0].message.content)
                    envelope = {'sample_id': uid, 'config_sha256': config_hash, 'result': result,
                                'response_file': name, 'response_sha256': digest(response_file.read_bytes())}
                    save(accepted, envelope)
                    break
                except Exception as exc:
                    detail = {'sample_id': uid, 'attempt': attempt, 'exception_type': type(exc).__name__,
                              'http_status': getattr(exc, 'status_code', None)}
                    save(out / 'responses' / (uid + '_' + str(time.time_ns()) + '.error.json'), detail)
                    print(json.dumps(detail), flush=True)
                    if detail['http_status'] in (400, 401, 403) or attempt == 3:
                        errors.append(detail)
                        break
                    time.sleep(2)
        if accepted.exists():
            envelope = json.loads(accepted.read_text(encoding='utf-8'))
            records.append({'sample_id': uid, 'split': row['split'], **envelope['result']})
            print(f'ML1 {len(records)} / 56 CALLS {calls}', flush=True)
    (out / 'proposals.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records), encoding='utf-8')
    summary = {'status': 'RETRANSCRIPTION_REVIEW_REQUIRED' if len(records) == 56 else 'RETRANSCRIPTION_PENDING',
               'planned_requests': 56, 'accepted': len(records), 'remaining': 56-len(records),
               'new_api_calls': calls, 'errors': errors, 'records_with_issues': sum(bool(r['issues']) for r in records),
               'config_sha256': config_hash, 'original_files_modified': False, 'labels_read': False,
               'training_started': False, 'ready_for_training': False, 'proposals_only': True}
    save(out / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if args.run and len(records) != 56:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
