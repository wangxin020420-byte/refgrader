import json
import unittest
import httpx
from label_free_model import BASE_URL, _reference_transcription, create_client


class ReferenceFallbackTests(unittest.TestCase):
    def client(self, text):
        seen = []
        def reply(request):
            seen.append(request)
            content = 'not valid JSON {' if len(seen) <= 3 else text
            return httpx.Response(200, json={'id':'mock','object':'chat.completion','created':0,
                'model':'glm-4.6v','choices':[{'index':0,'finish_reason':'stop',
                'message':{'role':'assistant','content':content}}]})
        return create_client('fake', http_client=httpx.Client(transport=httpx.MockTransport(reply),trust_env=False)), seen

    def test_quotes_preserved_and_fallback_recorded(self):
        client, seen = self.client('<<<BEGIN_REFERENCE>>>\nprintf("%d", x);\n<<<END_REFERENCE>>>')
        with client:
            value, used = _reference_transcription(client, {'type':'image_url','image_url':{'url':'data:image/png;base64,eA=='}}, ['s1'])
        self.assertTrue(used)
        self.assertEqual(value, 'printf("%d", x);')
        self.assertEqual(len(seen),4)
        self.assertTrue(all(str(x.url)==BASE_URL+'/chat/completions' for x in seen))
        self.assertNotIn('response_format', json.loads(seen[-1].content))

    def test_ambiguous_or_empty_marker_fails_closed(self):
        for text in ('no markers', '<<<BEGIN_REFERENCE>>>\n\n<<<END_REFERENCE>>>',
                     '<<<BEGIN_REFERENCE>>>\nx\n<<<END_REFERENCE>>>\nextra'):
            client, _ = self.client(text)
            with client, self.assertRaises(ValueError):
                _reference_transcription(client, {'type':'image_url','image_url':{'url':'data:image/png;base64,eA=='}})


if __name__=='__main__': unittest.main()
