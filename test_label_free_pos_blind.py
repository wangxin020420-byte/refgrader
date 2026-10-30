import unittest
from label_free_pos_blind import blind_pos_record


class BlindAdapterTests(unittest.TestCase):
    def setUp(self):
        self.context = {'question_id': 'Q1', 'question_text': 'Question', 'reference_text': 'Reference',
                        'total_score': 5, 'initial_rubric': [
                            {'id': 's1', 'item': 'Criterion', 'points': 5, 'standard_answer_text': 'Reference'}]}
        self.observation = {'valid_probe_count': 3, 'probes': [
            {'s1': {'score': x, 'category': 'partial'}} for x in (2, 3, 4)]}

    def test_grader_receives_no_original_observation_or_route(self):
        seen = []
        def grader(context, image, **kwargs):
            seen.append((context, image, kwargs))
            return {'valid_probe_count': 3, 'probes': [
                {'s1': {'score': 4, 'category': 'match'}} for _ in range(3)]}
        result = blind_pos_record(question_id='Q1', answer_id='A1', decision={'route': 'POS'},
                                  observation=self.observation, context=self.context, student_image='img',
                                  question_image=None, reference_images=[], grader=grader)
        self.assertEqual(result['items'][0]['base_score'], 3)
        self.assertEqual(result['items'][0]['blind_score'], 4)
        self.assertEqual(seen[0][0], self.context)
        self.assertNotIn('route', seen[0][0])
        self.assertNotIn('observation', seen[0][2])

    def test_nonpos_or_extra_fields_rejected_before_grading(self):
        def forbidden(*args, **kwargs):
            self.fail('grader must not run')
        common = dict(question_id='Q1', answer_id='A1', observation=self.observation,
                      student_image='img', question_image=None, reference_images=[], grader=forbidden)
        with self.assertRaises(ValueError):
            blind_pos_record(decision={'route': 'NEG'}, context=self.context, **common)
        with self.assertRaises(ValueError):
            blind_pos_record(decision={'route': 'POS'}, context=dict(self.context, teacher_score=5), **common)


if __name__ == '__main__':
    unittest.main()
