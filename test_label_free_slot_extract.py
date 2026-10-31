import unittest
from label_free_slot_extract import parse_slots, slot_ids, slot_instruction


class SlotExtractionTests(unittest.TestCase):
    def test_slots_are_not_criteria_and_preserve_quotes(self):
        text = ('<<<BEGIN_SLOT_1>>>\nprintf("%d", x);\n<<<END_SLOT_1>>>\n'
                '<<<BEGIN_SLOT_2>>>\n\n<<<END_SLOT_2>>>')
        out = parse_slots(text, 2)
        self.assertEqual(out['slots'][0]['id'], 'slot_1')
        self.assertEqual(out['slots'][0]['evidence'], 'printf("%d", x);')
        self.assertEqual(out['slots'][1]['status'], 'unclear')
        self.assertFalse(out['criterion_alignment_verified'])
        self.assertFalse(out['score_change_authorized'])

    def test_missing_marker_and_wrong_count_fail(self):
        with self.assertRaises(ValueError):
            parse_slots('<<<BEGIN_SLOT_1>>>\nx\n<<<END_SLOT_1>>>', 2)
        for count in (0, True, 101):
            with self.assertRaises(ValueError):
                slot_ids(count)

    def test_prompt_requires_crossout_visibility_without_criteria(self):
        prompt = slot_instruction('Question', 5)
        self.assertIn('crossed-out', prompt)
        self.assertIn('not by rubric item', prompt)
        self.assertIn('<<<BEGIN_SLOT_5>>>', prompt)
        self.assertNotIn('<<<BEGIN_STEP_5>>>', prompt)


if __name__ == '__main__':
    unittest.main()
