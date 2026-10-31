import json
import unittest
import httpx
from label_free_model import BASE_URL, create_client
from label_free_slot_extract import parse_slots
from label_free_slot_mapping_model import propose_mapping


class MappingModelTests(unittest.TestCase):
    def setUp(self):
        self.rubric = [{'id': 's1', 'item': 'first', 'points': 1, 'standard_answer_text': 'answer'},
                       {'id': 's2', 'item': 'second', 'points': 1, 'standard_answer_text': 'answer'}]
        self.slots = parse_slots('<<<BEGIN_SLOT_1>>>\nfopen("w")\n<<<END_SLOT_1>>>\n'
                                 '<<<BEGIN_SLOT_2>>>\n\n<<<END_SLOT_2>>>', 2)

    def client(self, items):
        requests = []
        def handler(request):
            requests.append(request)
            return httpx.Response(200, json={'id': 'mock', 'object': 'chat.completion', 'created': 0,
                'model': 'glm-5.2', 'choices': [{'index': 0, 'finish_reason': 'stop',
                'message': {'role': 'assistant', 'content': json.dumps({'items': items})}}]})
        return create_client('fake', http_client=httpx.Client(transport=httpx.MockTransport(handler), trust_env=False)), requests

    def test_overlapping_slots_are_traceable_and_never_scored(self):
        items = [{'item_id': 's1', 'slot_ids': ['slot_1']}, {'item_id': 's2', 'slot_ids': ['slot_1']}]
        client, requests = self.client(items)
        with client:
            result = propose_mapping(client, 'Q', self.rubric, self.slots)
        self.assertEqual(result['criteria'][1]['citations'][0]['excerpt'], 'fopen("w")')
        self.assertFalse(result['score_change_authorized'])
        self.assertFalse(result['criterion_relevance_verified'])
        self.assertEqual(str(requests[0].url), BASE_URL + '/chat/completions')
        self.assertNotIn('teacher_score', requests[0].content.decode())

    def test_rejects_unclear_slot_or_model_score(self):
        for items in ([{'item_id': 's1', 'slot_ids': ['slot_2']}, {'item_id': 's2', 'slot_ids': []}],
                      [{'item_id': 's1', 'slot_ids': [], 'score': 1}, {'item_id': 's2', 'slot_ids': []}]):
            client, _ = self.client(items)
            with client, self.assertRaises(ValueError):
                propose_mapping(client, 'Q', self.rubric, self.slots)


if __name__ == '__main__': unittest.main()
