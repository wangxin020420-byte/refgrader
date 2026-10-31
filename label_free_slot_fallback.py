"""Conservative exploratory fallback for the verified five-slot CPL_5 layout.

Transcription and mapping are unverified; the caller MUST retain review status.
The layout restriction is intentional until other images are audited.
"""
from label_free_slot_extract import slot_instruction, parse_slots
from label_free_slot_mapping_model import propose_mapping


def extract_fallback(client, *, context, rubric, images, vision_model):
    if context.get('question_id') != 'CPL_5' or len(rubric) != 5:
        raise ValueError('Slot fallback is only supported for CPL_5')
    if not isinstance(images, list) or len(images) < 2:
        raise ValueError('Missing student image')
    content = [{'type': 'text', 'text': slot_instruction(str(context.get('question_text', '')), 5)}] + images[1:]
    response = client.chat.completions.create(model=vision_model, messages=[{'role': 'user', 'content': content}], temperature=0)
    if not response.choices or response.choices[0].finish_reason != 'stop':
        raise ValueError('slot_extraction: incomplete response')
    slots = parse_slots(response.choices[0].message.content, 5)
    mapping = propose_mapping(client, str(context.get('question_text', '')), rubric, slots)
    facts = {}
    for item in mapping['criteria']:
        # Whole-slot citations are not verified item-level evidence. Mark every
        # criterion unclear, even when an excerpt exists, to avoid overclaiming.
        facts[item['item_id']] = {'status': 'unclear', 'evidence': '\n'.join(r['excerpt'] for r in item['citations'])}
    return facts, {'protocol': 'cpl5_visible_slots_candidate_v1', 'slots': slots, 'mapping': mapping,
                   'requires_human_review': True, 'score_change_authorized': False}
