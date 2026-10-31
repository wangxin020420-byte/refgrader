import json
import tempfile
import unittest
from pathlib import Path
import httpx
from label_free_model import BASE_URL, create_client, grade_one


class SlotFallbackTests(unittest.TestCase):
    def fixture(self, *, marker_valid=True, question_id='CPL_5'):
        counts = {'vision': 0, 'text': 0}
        requests = []
        def reply(request):
            requests.append(request)
            body = json.loads(request.content)
            if body['model'] == 'glm-4.6v':
                counts['vision'] += 1
                if counts['vision'] <= 3:
                    content = 'invalid JSON {'
                else:
                    content = '\n'.join(line for i in range(1, 6) for line in
                        (f'<<<BEGIN_SLOT_{i}>>>', 'fopen("w")' if i == 1 else '', f'<<<END_SLOT_{i}>>>'))
                    if not marker_valid: content = content.replace('<<<END_SLOT_5>>>', '')
            else:
                counts['text'] += 1
                if counts['text'] == 1:
                    content = json.dumps({'items': [{'item_id': f'step_{i}', 'slot_ids': ['slot_1'] if i == 1 else []} for i in range(1, 6)]})
                else:
                    content = json.dumps({'items': [{'id': f'step_{i}', 'score': 0, 'category': 'unclear'} for i in range(1, 6)]})
            return httpx.Response(200, json={'id': 'mock', 'object': 'chat.completion', 'created': 0,
                'model': body['model'], 'choices': [{'index': 0, 'finish_reason': 'stop',
                'message': {'role': 'assistant', 'content': content}}]})
        context = {'question_id': question_id, 'question_text': 'q', 'reference_text': 'r', 'total_score': 5,
                   'initial_rubric': [{'id': f'step_{i}', 'item': 'x', 'points': 1, 'standard_answer_text': 'r'} for i in range(1,6)]}
        client = create_client('fake', http_client=httpx.Client(transport=httpx.MockTransport(reply), trust_env=False))
        return client, context, counts, requests

    def test_fallback_preserves_quotes_and_treats_all_facts_as_uncertain(self):
        client, context, counts, requests = self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp)/'answer.png'; image.write_bytes(b'x')
            with client: result = grade_one(client, context, image)
        self.assertEqual(counts, {'vision': 4, 'text': 4})
        self.assertEqual(result['facts']['step_1']['evidence'], 'fopen("w")')
        self.assertTrue(all(f['status'] == 'unclear' for f in result['facts'].values()))
        self.assertTrue(result['slot_fallback']['requires_human_review'])
        self.assertTrue(all(str(r.url) == BASE_URL + '/chat/completions' for r in requests))
        self.assertTrue(all(b'teacher_score' not in r.content for r in requests))

    def test_bad_markers_and_other_questions_fail_closed(self):
        for kwargs in ({'marker_valid': False}, {'question_id': 'OTHER'}):
            client, context, _, _ = self.fixture(**kwargs)
            with tempfile.TemporaryDirectory() as tmp:
                image = Path(tmp)/'answer.png'; image.write_bytes(b'x')
                with client, self.assertRaises(ValueError): grade_one(client, context, image)


if __name__ == '__main__': unittest.main()
