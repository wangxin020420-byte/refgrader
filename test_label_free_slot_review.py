import unittest
from label_free_slot_extract import parse_slots
from label_free_slot_mapping import validate_candidate_mapping
from label_free_slot_review import assess_slot_mapping


class ReviewGateTests(unittest.TestCase):
    def test_conflict_goes_to_referencing_criteria_only(self):
        slots = parse_slots('<<<BEGIN_SLOT_1>>>\nx\n<<<END_SLOT_1>>>\n'
                            '<<<BEGIN_SLOT_2>>>\ny\n<<<END_SLOT_2>>>', 2)
        proposals = [{'item_id': 's1', 'citations': [{'slot_id': 'slot_1','start': 0,'end': 1}]},
                     {'item_id': 's2', 'citations': [{'slot_id': 'slot_2','start': 0,'end': 1}]},
                     {'item_id': 's3', 'citations': []}]
        mapping = validate_candidate_mapping(slots, ['s1', 's2', 's3'], proposals)
        out = assess_slot_mapping(mapping, {'slot_1': 'confirmed_conflict'})
        self.assertIn('observed_transcription_conflict', out['criteria'][0]['review_reasons'])
        self.assertNotIn('observed_transcription_conflict', out['criteria'][1]['review_reasons'])
        self.assertIn('image_fidelity_not_checked', out['criteria'][1]['review_reasons'])
        self.assertIn('mapping_unresolved', out['criteria'][2]['review_reasons'])
        self.assertIsNone(out['automatic_score'])
        self.assertFalse(out['score_change_authorized'])

    def test_invalid_signal_fails_closed(self):
        with self.assertRaises(ValueError):
            assess_slot_mapping({'schema': 'label_free_slot_mapping_candidate_v1'}, {'slot_1': 'verified_correct'})


if __name__ == '__main__': unittest.main()
