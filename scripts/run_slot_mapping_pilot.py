"""One-sample, label-free slot-and-mapping pilot. No grading or checkpoint write."""
import argparse
import getpass
import json
import os
import sys
from pathlib import Path
import httpx

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / 'label_free_model.py').is_file():
    sys.path.insert(0, str(ROOT))

import label_free_model as model
from label_free_checkpoint import Checkpoint, _digest
from label_free_inputs import load_unlabeled_answers
from label_free_question_snapshot import load_candidate_question
from label_free_run import _image
from label_free_slot_extract import slot_instruction, parse_slots
from label_free_slot_mapping_model import propose_mapping
from scripts.run_label_free_development import ROOT, _inputs, _single_writer

RUN_ID = 'lf45_co3ind_20261029_v1_calibration_mixed_v1'
QID, AID = 'CPL_5', 'ANS_CPL_254'
SNAPSHOT = ROOT / 'data/label_free_snapshots/csbench_3531732_co3_independent_v1'


def mocked_client():
    def handler(request):
        body = json.loads(request.content)
        if body['model'] == model.VISION_MODEL:
            text = '\n'.join(line for i in range(1, 6) for line in
                (f'<<<BEGIN_SLOT_{i}>>>', 'printf("%d", x);', f'<<<END_SLOT_{i}>>>'))
        else:
            text = json.dumps({'items': [{'item_id': f'step_{i}', 'slot_ids': ['slot_1']}
                                         for i in range(1, 6)]})
        return httpx.Response(200, json={'id': 'mock', 'object': 'chat.completion', 'created': 0,
            'model': body['model'], 'choices': [{'index': 0, 'finish_reason': 'stop',
            'message': {'role': 'assistant', 'content': text}}]})
    return model.create_client('fake', http_client=httpx.Client(transport=httpx.MockTransport(handler), trust_env=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mock', action='store_true')
    args = parser.parse_args(argv)
    try:
        run = ROOT / 'results_runs/label_free_development' / RUN_ID
        envelope = json.loads((run/'manifest.json').read_text(encoding='utf-8'))
        manifest = envelope['manifest']; cfg = manifest['run_config']
        if envelope['sha256'] != _digest(manifest) or _inputs(cfg['question_ids'], 'calibration', SNAPSHOT, 'candidate_max_v1') != manifest:
            raise ValueError('Current source or inputs differ from the run manifest')
        lineage = json.loads((run/'lineage.json').read_text(encoding='utf-8'))
        if lineage['sha256'] != _digest(lineage['lineage']) or lineage['lineage']['target'] != manifest['source']:
            raise ValueError('Invalid lineage')
        with _single_writer(run):
            ck = Checkpoint(run, manifest)
            for source in lineage['lineage']['source_answers']:
                result = ck.load(source['checkpoint_id'])
                if result is None or _digest(result) != source['result_sha256']:
                    raise ValueError('Imported answer does not match lineage')
            next_pair = None
            for q in cfg['question_ids']:
                for a in load_unlabeled_answers(ROOT/'data/csbench', q, 'calibration'):
                    if ck.load(q + '_' + a['answer_id']) is None:
                        next_pair = (q, a['answer_id']); break
                if next_pair: break
            if next_pair != (QID, AID):
                raise ValueError('Next uncompleted answer changed; do not use this pilot')
            cand = load_candidate_question(SNAPSHOT, QID)
            if cand['reference_images'] or cand['question_image']:
                raise ValueError('Pilot does not support extra images')
            context = cand['context']; rubric = context['initial_rubric']
            if len(rubric) != 5: raise ValueError('Unexpected criterion count')
            image = _image(ROOT/'data/csbench', 'student_images/' + QID, QID, AID, required=True)
            if model.BASE_URL != 'https://open.bigmodel.cn/api/coding/paas/v4':
                raise ValueError('Incorrect model endpoint')
            if args.mock:
                client = mocked_client()
            else:
                key = os.getenv('ZHIPUAI_API_KEY') or os.getenv('ZHIPU_API_KEY') or getpass.getpass('Coding Plan API Key: ')
                if not key: raise ValueError('Coding Plan key missing')
                client = model.create_client(key)
            with client:
                client.max_retries = 0
                response = client.chat.completions.create(model=model.VISION_MODEL, temperature=0,
                    messages=[{'role': 'user', 'content': [
                        {'type': 'text', 'text': slot_instruction(context['question_text'], 5)},
                        {'type': 'text', 'text': 'Student answer:'}, model._image(image)]}])
                if response.choices[0].finish_reason != 'stop': raise ValueError('Unexpected extraction finish reason')
                slots = parse_slots(response.choices[0].message.content, 5)
                mapping = propose_mapping(client, context['question_text'], rubric, slots)
            print(json.dumps({'scope': 'manual_review_pilot_only', 'question_id': QID, 'answer_id': AID,
                              'mock': args.mock, 'checkpoint_written': False, 'score_generated': False,
                              'slots': slots, 'candidate_mapping': mapping}, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print('Pilot failed: ' + str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
