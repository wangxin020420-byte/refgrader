"""Offline, teacher-score-free POS pseudo-difference prototype.

Only caller-supplied, explicitly projected numeric records are accepted. This
module does not generate blind reassessments, prove their provenance, authorize
BND arbitration, or release corrected grades. Inputs and fitted artifacts must
be frozen by the caller; no existing scoring run is read or modified.
"""
import hashlib
import json
import math


def _digest(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()


def _number(value, name, *, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    value = float(value)
    if minimum is not None and value < minimum or maximum is not None and value > maximum:
        raise ValueError(f'{name} out of bounds')
    return value


def _id(value):
    if not isinstance(value, str) or not value or len(value) > 128:
        raise ValueError('Invalid identifier')
    return value


def _items(items, *, reference):
    if not isinstance(items, list) or not items:
        raise ValueError('Nonempty item list required')
    result = {}
    expected = {'item_id', 'max_score', 'base_score'} | ({'blind_score'} if reference else set())
    for item in items:
        if not isinstance(item, dict) or set(item) != expected:
            raise ValueError('Unexpected item fields')
        item_id = _id(item['item_id'])
        if item_id in result:
            raise ValueError('Duplicate item')
        cap = _number(item['max_score'], 'max_score', minimum=0)
        base = _number(item['base_score'], 'base_score', minimum=0, maximum=cap)
        blind = _number(item['blind_score'], 'blind_score', minimum=0, maximum=cap) if reference else None
        result[item_id] = (cap, base, blind)
    return result


def fit_pos(records, *, min_count, shrinkage, cap):
    """Fit per-question/item mean blind-minus-base differences, without labels.

    records: POS-only projections with question_id, answer_id, route, items.
    Non-POS, extra fields, and duplicate identities fail closed.
    """
    if not isinstance(records, list) or not records:
        raise ValueError('Nonempty POS records required')
    if isinstance(min_count, bool) or not isinstance(min_count, int) or min_count < 1:
        raise ValueError('min_count must be a positive integer')
    shrinkage = _number(shrinkage, 'shrinkage', minimum=0)
    cap = _number(cap, 'cap', minimum=0, maximum=1)
    groups, identities, question_rubrics = {}, set(), {}
    for record in records:
        if not isinstance(record, dict) or set(record) != {'question_id', 'answer_id', 'route', 'items'}:
            raise ValueError('Unexpected POS record fields')
        qid, aid = _id(record['question_id']), _id(record['answer_id'])
        if record['route'] != 'POS' or (qid, aid) in identities:
            raise ValueError('Only unique POS records may fit')
        identities.add((qid, aid))
        items = _items(record['items'], reference=True)
        rubric = {item_id: item[0] for item_id, item in items.items()}
        if qid in question_rubrics and question_rubrics[qid] != rubric:
            raise ValueError('Inconsistent question rubric across POS records')
        question_rubrics[qid] = rubric
        for item_id, (maximum, base, blind) in items.items():
            key = (qid, item_id)
            group = groups.setdefault(key, {'max_score': maximum, 'differences': []})
            if group['max_score'] != maximum:
                raise ValueError('Inconsistent item maximum')
            # Zero-point items cannot be normalized or corrected.
            if maximum > 0:
                group['differences'].append((blind - base) / maximum)
    result = []
    for (qid, item_id), group in sorted(groups.items()):
        differences = group['differences']
        n = len(differences)
        usable = n >= min_count and group['max_score'] > 0
        mean = sum(differences) / n if n else None
        delta = max(-cap, min(cap, sum(differences) / (n + shrinkage))) if usable else 0.0
        result.append({'question_id': qid, 'item_id': item_id, 'max_score': group['max_score'],
                       'count': n, 'mean_pseudo_difference': mean, 'delta': delta,
                       'usable': usable})
    return {'schema': 'label_free_pos_correction_fit_v1',
            'config': {'min_count': min_count, 'shrinkage': shrinkage, 'cap': cap},
            'input_sha256': _digest(records), 'source_identities': [list(x) for x in sorted(identities)],
            'groups': result, 'scope': 'offline_exploratory_only',
            'reference_provenance_verified': False, 'teacher_independence_verified': False}


def apply_candidate(fit, record):
    """Compute a candidate, preserving original route and never releasing a grade.

    No same-sample reuse: this inductive prototype refuses any fit identity.
    BND baseline means original probe mean, not an unimplemented arbitration.
    """
    if not isinstance(fit, dict) or fit.get('schema') != 'label_free_pos_correction_fit_v1':
        raise ValueError('Invalid correction fit')
    if not isinstance(record, dict) or set(record) != {'question_id', 'answer_id', 'route', 'items'}:
        raise ValueError('Unexpected target record fields')
    qid, aid = _id(record['question_id']), _id(record['answer_id'])
    route = record['route']
    if route not in {'POS', 'BND', 'NEG'}:
        raise ValueError('Invalid route')
    if [qid, aid] in fit['source_identities']:
        raise ValueError('Target is a fit sample; use separate data or cross-fitting')
    groups = {(g['question_id'], g['item_id']): g for g in fit['groups']}
    target_items = _items(record['items'], reference=False)
    known_items = {item_id for question_id, item_id in groups if question_id == qid}
    if known_items and set(target_items) != known_items:
        raise ValueError('Target rubric items differ from fit')
    output = []
    for item_id, (maximum, base, _) in target_items.items():
        group = groups.get((qid, item_id))
        if group and group['max_score'] != maximum:
            raise ValueError('Target item maximum differs from fit')
        usable = bool(group and group['usable'])
        delta = group['delta'] if usable else 0.0
        candidate = max(0.0, min(maximum, base + maximum * delta))
        output.append({'item_id': item_id, 'base_score': base, 'max_score': maximum,
                       'delta': delta, 'candidate_score': candidate,
                       'correction_available': usable})
    return {'question_id': qid, 'answer_id': aid, 'original_route': route,
            'items': output, 'base_total': sum(x['base_score'] for x in output),
            'candidate_total': sum(x['candidate_score'] for x in output),
            'automatic_score': None, 'requires_human_review': route != 'POS',
            'candidate_only': True, 'score_change_authorized': False,
            'bnd_arbitration_applied': False, 'fit_sha256': _digest(fit)}
