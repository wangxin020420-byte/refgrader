"""Annotate diagnostic slot mappings; do not decide a grade or route.

An observed inconsistency is a reason to review, not evidence of an incorrect
student answer. Visual annotations and mappings remain externally sourced.
"""


def assess_slot_mapping(mapping, slot_signals):
    if not isinstance(mapping, dict) or mapping.get('schema') != 'label_free_slot_mapping_candidate_v1':
        raise ValueError('Candidate slot mapping required')
    if not isinstance(slot_signals, dict):
        raise ValueError('Slot signals must be a dictionary')
    allowed = {'not_checked', 'suspected', 'confirmed_conflict'}
    if any(not isinstance(k, str) or not k or v not in allowed for k, v in slot_signals.items()):
        raise ValueError('Invalid slot signal')
    criteria = mapping.get('criteria')
    if not isinstance(criteria, list) or not criteria:
        raise ValueError('Criteria required')
    output = []
    seen = set()
    for item in criteria:
        if not isinstance(item, dict) or set(item) != {'item_id', 'citations', 'mapping_status'}:
            raise ValueError('Invalid candidate criterion')
        ident = item['item_id']
        if not isinstance(ident, str) or not ident or ident in seen:
            raise ValueError('Duplicate criterion')
        seen.add(ident)
        refs = item['citations']
        if not isinstance(refs, list) or item['mapping_status'] != ('candidate' if refs else 'unresolved'):
            raise ValueError('Invalid mapping status')
        slots = []
        for ref in refs:
            if not isinstance(ref, dict) or set(ref) != {'slot_id', 'start', 'end', 'excerpt'}:
                raise ValueError('Invalid citation')
            slot = ref['slot_id']
            if not isinstance(slot, str) or not slot or not isinstance(ref['excerpt'], str) or not ref['excerpt'].strip():
                raise ValueError('Invalid citation value')
            slots.append(slot)
        states = [slot_signals.get(slot, 'not_checked') for slot in slots]
        reasons = []
        if not refs: reasons.append('mapping_unresolved')
        if 'confirmed_conflict' in states: reasons.append('observed_transcription_conflict')
        if 'suspected' in states: reasons.append('suspected_transcription_conflict')
        if 'not_checked' in states: reasons.append('image_fidelity_not_checked')
        reasons.append('criterion_mapping_unverified')
        output.append({'item_id': ident, 'slot_ids': slots, 'review_reasons': reasons})
    return {'schema': 'label_free_slot_review_v1', 'scope': 'diagnostic_only',
            'criteria': output, 'automatic_score': None, 'score_change_authorized': False,
            'requires_human_review': True,
            'note': 'Signals require provenance; this report does not authorize grading or assert correctness.'}
