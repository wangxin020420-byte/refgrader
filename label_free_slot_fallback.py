"""Conservative exploratory fallback for malformed visual JSON.

Transcription and mapping are unverified; the caller MUST retain review status.
Other layouts are transcribed as a single whole-image region and must be reviewed.
"""
from label_free_slot_extract import slot_instruction, parse_slots
from label_free_slot_mapping_model import propose_mapping


def extract_fallback(client, *, context, rubric, images, vision_model):
    if not isinstance(images, list) or len(images) < 2:
        raise ValueError('Missing student image')
    question = str(context.get('question_text', ''))
    if context.get('question_id') == 'CPL_5' and len(rubric) == 5:
        count = 5
        prompt = slot_instruction(question, count)
        protocol = 'cpl5_visible_slots_candidate_v1'
    else:
        count = 1
        prompt = ('Transcribe all visible student writing across the entire answer image in reading order. '
                  'Do not grade or infer missing content. Preserve code, quotes, punctuation, '
                  'and explicitly identify crossed-out text. If no writing is readable leave the '
                  'section empty; that does not prove a blank answer. Return only one section: '
                  '<<<BEGIN_SLOT_1>>> on its own line, the transcription, then '
                  '<<<END_SLOT_1>>> on its own line.\nQuestion: ' + question)
        protocol = 'whole_image_candidate_v1'
    content = [{'type': 'text', 'text': prompt}] + images[1:]
    response = client.chat.completions.create(model=vision_model, messages=[{'role': 'user', 'content': content}], temperature=0)
    if not response.choices or response.choices[0].finish_reason != 'stop':
        raise ValueError('slot_extraction: incomplete response')
    slots = parse_slots(response.choices[0].message.content, count)
    mapping = propose_mapping(client, str(context.get('question_text', '')), rubric, slots)
    facts = {}
    for item in mapping['criteria']:
        # Whole-slot citations are not verified item-level evidence. Mark every
        # criterion unclear, even when an excerpt exists, to avoid overclaiming.
        facts[item['item_id']] = {'status': 'unclear', 'evidence': '\n'.join(r['excerpt'] for r in item['citations'])}
    return facts, {'protocol': protocol, 'slots': slots, 'mapping': mapping,
                   'requires_human_review': True, 'score_change_authorized': False}
