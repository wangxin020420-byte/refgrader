import unittest
from label_free_marker_extract import marker_instruction, parse_markers
from label_free_model import _validate_facts


class MarkerExtractTests(unittest.TestCase):
    def test_preserves_code_quotes_and_empty_is_unclear(self):
        text = ('<<<BEGIN_STEP_1>>>\nprintf("%d", x);\n<<<END_STEP_1>>>\n'
                '<<<BEGIN_STEP_2>>>\n\n<<<END_STEP_2>>>')
        facts = _validate_facts(parse_markers(text, ['step_1', 'step_2']), ['step_1', 'step_2'])
        self.assertEqual(facts['step_1']['evidence'], 'printf("%d", x);')
        self.assertEqual(facts['step_2']['status'], 'unclear')

    def test_missing_duplicate_or_reordered_markers_fail(self):
        bad = ['<<<BEGIN_STEP_1>>>\nx',
               '<<<BEGIN_STEP_1>>>\nx\n<<<END_STEP_1>>>\n<<<END_STEP_1>>>',
               '<<<END_STEP_1>>>\nx\n<<<BEGIN_STEP_1>>>',
               'prefix\n<<<BEGIN_STEP_1>>>\nx\n<<<END_STEP_1>>>']
        for text in bad:
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_markers(text, ['step_1'])

    def test_rejects_embedded_marker_and_case_collisions(self):
        with self.assertRaises(ValueError):
            parse_markers('<<<BEGIN_X>>>\na <<<END_X>>> b\n<<<END_X>>>', ['x'])
        with self.assertRaises(ValueError):
            parse_markers('x', ['a', 'A'])

    def test_prompt_uses_only_passed_rubric_fields(self):
        prompt = marker_instruction('Q', [{'id': 'step_1', 'item': 'I', 'teacher_score': 7}])
        self.assertNotIn('teacher_score', prompt)
        self.assertIn('<<<BEGIN_STEP_1>>>', prompt)


if __name__ == '__main__':
    unittest.main()
