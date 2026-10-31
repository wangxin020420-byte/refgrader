"""Trace candidate criterion evidence to exact slot offsets, without grading.

Offset checks prove only that text exists in a transcription. They do not
prove image fidelity, criterion relevance, or the meaning of a crossed-out span.
"""


def validate_candidate_mapping(slot_document, rubric_ids, proposals):
    if not isinstance(slot_document, dict) or slot_document.get('schema') != 'label_free_visible_slots_v1':
        raise ValueError('Invalid slot document')
    slots = slot_document.get('slots')
    if not isinstance(slots, list) or not slots:
        raise ValueError('Missing slots')
    slot_map = {}
    for slot in slots:
        if not isinstance(slot, dict) or set(slot) != {'id', 'status', 'evidence'}:
            raise ValueError('Invalid slot')
        ident, status, evidence = slot['id'], slot['status'], slot['evidence']
        if not isinstance(ident, str) or not ident or ident in slot_map or status not in {'present', 'unclear'} or not isinstance(evidence, str):
            raise ValueError('Invalid slot values')
        if (status == 'present') != bool(evidence.strip()):
            raise ValueError('Slot status and evidence disagree')
        slot_map[ident] = slot
    if (not isinstance(rubric_ids, list) or not rubric_ids or
        any(not isinstance(item, str) or not item for item in rubric_ids) or len(set(rubric_ids)) != len(rubric_ids)):
        raise ValueError('Invalid rubric IDs')
    if not isinstance(proposals, list) or len(proposals) != len(rubric_ids):
        raise ValueError('Expected one proposal per criterion')
    output, seen = [], set()
    for proposal in proposals:
        if not isinstance(proposal, dict) or set(proposal) != {'item_id', 'citations'}:
            raise ValueError('Unexpected proposal fields')
        item_id, refs = proposal['item_id'], proposal['citations']
        if item_id not in rubric_ids or item_id in seen or not isinstance(refs, list):
            raise ValueError('Unknown or duplicate criterion')
        seen.add(item_id)
        citations, unique = [], set()
        for ref in refs:
            if not isinstance(ref, dict) or set(ref) != {'slot_id', 'start', 'end'}:
                raise ValueError('Invalid citation fields')
            ident, start, end = ref['slot_id'], ref['start'], ref['end']
            if ident not in slot_map or slot_map[ident]['status'] != 'present':
                raise ValueError('Citation requires an existing readable slot')
            if any(isinstance(x, bool) or not isinstance(x, int) for x in (start, end)):
                raise ValueError('Offsets must be integers')
            evidence = slot_map[ident]['evidence']
            if not 0 <= start < end <= len(evidence) or not evidence[start:end].strip():
                raise ValueError('Citation offset out of range')
            key = (ident, start, end)
            if key in unique:
                raise ValueError('Duplicate citation in criterion')
            unique.add(key)
            citations.append({'slot_id': ident, 'start': start, 'end': end, 'excerpt': evidence[start:end]})
        output.append({'item_id': item_id, 'citations': citations,
                       'mapping_status': 'candidate' if citations else 'unresolved'})
    return {'schema': 'label_free_slot_mapping_candidate_v1', 'scope': 'diagnostic_only',
            'criteria': output, 'offsets_verified': True, 'image_fidelity_verified': False,
            'criterion_relevance_verified': False, 'crossout_status_verified': False,
            'teacher_score_independence_verified': False, 'score_change_authorized': False,
            'requires_human_review': True}
