import unittest
from label_free_slot_extract import parse_slots
from label_free_slot_mapping import validate_candidate_mapping


def ref(slot, start, end):
    return {'slot_id': slot, 'start': start, 'end': end}


class SlotMappingTests(unittest.TestCase):
    def setUp(self):
        self.slots = parse_slots('<<<BEGIN_SLOT_1>>>\nif (x) {fopen("r");}\n<<<END_SLOT_1>>>\n'
                                 '<<<BEGIN_SLOT_2>>>\n\n<<<END_SLOT_2>>>', 2)
        self.refs = [{'item_id': 'step_1', 'citations': [ref('slot_1', 0, 6)]},
                     {'item_id': 'step_2', 'citations': [ref('slot_1', 0, 6)]},
                     {'item_id': 'step_3', 'citations': []}]

    def test_one_slot_may_support_multiple_criteria_without_authorization(self):
        out = validate_candidate_mapping(self.slots, ['step_1', 'step_2', 'step_3'], self.refs)
        self.assertEqual(out['criteria'][0]['citations'][0]['excerpt'], 'if (x)')
        self.assertEqual(out['criteria'][1]['citations'][0]['excerpt'], 'if (x)')
        self.assertEqual(out['criteria'][2]['mapping_status'], 'unresolved')
        self.assertFalse(out['criterion_relevance_verified'])
        self.assertFalse(out['teacher_score_independence_verified'])
        self.assertFalse(out['score_change_authorized'])

    def test_no_fabricated_or_out_of_bounds_evidence(self):
        for bad in (ref('slot_1', -1, 2), ref('slot_1', 0, 999), ref('slot_2', 0, 1), ref('slot_1', True, 3)):
            refs = [dict(self.refs[0], citations=[bad]), *self.refs[1:]]
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_candidate_mapping(self.slots, ['step_1', 'step_2', 'step_3'], refs)

    def test_missing_and_extra_score_fields_rejected(self):
        with self.assertRaises(ValueError):
            validate_candidate_mapping(self.slots, ['step_1', 'step_2', 'step_3'], self.refs[:2])
        with self.assertRaises(ValueError):
            validate_candidate_mapping(self.slots, ['step_1', 'step_2', 'step_3'],
                                       [dict(self.refs[0], score=2), *self.refs[1:]])


if __name__ == '__main__':
    unittest.main()
