"""Minimal independent-call adapter; not an experiment runner or provenance audit.

Caller must verify immutable source checkpoints, snapshot and image content.
No API calls are made here unless a caller injects a live grader.
"""
from label_free_pos_correction import _id, _number


def blind_pos_record(*, question_id, answer_id, decision, observation, context,
                     student_image, question_image, reference_images, grader):
    """Create a numeric POS record using a grader blind to old score/route.

    The grader gets only an explicit question context and image arguments.
    It may share model biases with original scoring; the caller must verify
    inputs and record model/prompt provenance separately.
    """
    qid, aid = _id(question_id), _id(answer_id)
    if not isinstance(decision, dict) or decision.get('route') != 'POS':
        raise ValueError('Original decision must be POS')
    if not isinstance(context, dict) or set(context) != {
        'question_id', 'question_text', 'reference_text', 'total_score', 'initial_rubric'
    } or context['question_id'] != qid:
        raise ValueError('Invalid whitelisted question context')
    rubric = context['initial_rubric']
    if not isinstance(rubric, list) or not rubric:
        raise ValueError('Invalid rubric')
    limits = {}
    for item in rubric:
        if not isinstance(item, dict) or set(item) != {'id', 'item', 'points', 'standard_answer_text'}:
            raise ValueError('Unexpected rubric fields')
        item_id = _id(item['id'])
        if item_id in limits or not isinstance(item['item'], str) or not isinstance(item['standard_answer_text'], str):
            raise ValueError('Invalid rubric item')
        limits[item_id] = _number(item['points'], 'points', minimum=0)
    total = _number(context['total_score'], 'total_score', minimum=0)
    if total <= 0 or abs(sum(limits.values()) - total) > 1e-6:
        raise ValueError('Invalid rubric total')
    if not isinstance(context['question_text'], str) or not isinstance(context['reference_text'], str):
        raise ValueError('Invalid question text')

    def means(value):
        if not isinstance(value, dict) or value.get('valid_probe_count') != 3:
            raise ValueError('Three valid probes required')
        probes = value.get('probes')
        if not isinstance(probes, list) or len(probes) != 3:
            raise ValueError('Three probes required')
        collected = {item_id: [] for item_id in limits}
        for probe in probes:
            if not isinstance(probe, dict) or set(probe) != set(limits):
                raise ValueError('Probe items differ from rubric')
            for item_id, entry in probe.items():
                if not isinstance(entry, dict) or set(entry) != {'score', 'category'}:
                    raise ValueError('Invalid probe item')
                collected[item_id].append(_number(entry['score'], 'score', minimum=0, maximum=limits[item_id]))
        return {item_id: sum(scores) / 3 for item_id, scores in collected.items()}

    base = means(observation)  # validate before the separate call
    blind = means(grader(context, student_image, question_image=question_image,
                         reference_images=reference_images))
    return {'question_id': qid, 'answer_id': aid, 'route': 'POS', 'items': [
        {'item_id': item_id, 'max_score': limits[item_id],
         'base_score': base[item_id], 'blind_score': blind[item_id]}
        for item_id in limits
    ]}
