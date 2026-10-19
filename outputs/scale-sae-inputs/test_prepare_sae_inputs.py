import unittest

import prepare_sae_inputs as p


class Batch(dict):
    def __init__(self, a, b):
        ids = [101] + [ord(t[0]) + 1000 for t in a] + [102] + [ord(t[0]) + 1000 for t in b] + [102]
        super().__init__(input_ids=ids, attention_mask=[1] * len(ids),
                         token_type_ids=[0] * (len(a) + 2) + [1] * (len(b) + 1),
                         offset_mapping=[(0, 0)] + [t[1] for t in a] + [(0, 0)] + [t[1] for t in b] + [(0, 0)])
        self.sides = [None] + [0] * len(a) + [None] + [1] * len(b) + [None]

    def sequence_ids(self):
        return self.sides


class CharacterTokenizer:
    """Offset contract fixture, not a substitute for remote real-tokenizer preflight."""
    def __call__(self, a, b, truncation=False, max_length=512, **kwargs):
        a = [(c, (i, i+1)) for i, c in enumerate(a) if not c.isspace()]
        b = [(c, (i, i+1)) for i, c in enumerate(b) if not c.isspace()]
        if truncation:
            while len(a) + len(b) + 3 > max_length:
                (a if len(a) > len(b) else b).pop()
        return Batch(a, b)


def triple(t, c='term', w='summary'):
    return {'t': t, 'c': c, 'w': w}


class InputTests(unittest.TestCase):
    def test_dedup_keeps_origins_and_verbatim_contract(self):
        text, entities = p.entity_text([(0, '第二次', [triple('第二次')]),
                                        (1, '第二次', [triple('第二次', w='different summary')])])
        self.assertEqual(text, '第二次 is a term')
        self.assertEqual(len(entities), 1)
        self.assertEqual(len(entities[0]['origins']), 2)
        with self.assertRaisesRegex(ValueError, 'Nonverbatim'):
            p.entity_text([(0, '第二次', [triple('第二次排序')])])

    def test_identical_text_on_both_sides_has_distinct_spans(self):
        text, entities = p.entity_text([(0, 'alpha', [triple('alpha')])])
        tokenizer = CharacterTokenizer()
        batch = tokenizer(text, text)
        accepted, dropped = p.align_entities(batch, batch, [entities, entities])
        self.assertEqual(len(accepted), 2)
        self.assertFalse(dropped)
        self.assertLess(accepted[0]['token_span'][1], accepted[1]['token_span'][0])
        self.assertEqual([r['sequence_id'] for r in accepted], [0, 1])

    def test_partial_entity_is_excluded_not_mislabelled_complete(self):
        text, entities = p.entity_text([(0, 'abcdef', [triple('abcdef')])])
        tokenizer = CharacterTokenizer()
        full = tokenizer(text, '')
        kept = tokenizer(text, '', truncation='longest_first', max_length=8)
        accepted, dropped = p.align_entities(full, kept, [entities, []])
        self.assertFalse(accepted)
        self.assertEqual(len(dropped), 1)
        self.assertGreater(dropped[0]['retained_tokens'], 0)
        self.assertLess(dropped[0]['retained_tokens'], dropped[0]['full_tokens'])

    def test_row_negative_label_blank_answer_multiple_references(self):
        row = {'id': 'x', 'sample_id': 'ADS:train:1', 'kind': 'synthetic_candidate',
               'label': -0.1, 'references': ['甲', '', '乙'], 'answer': '',
               'reference_triples': [[triple('甲')], [], [triple('乙')]], 'answer_triples': []}
        encoded, audit = p.encode_row(row, CharacterTokenizer())
        self.assertEqual(encoded['label'], -0.1)
        self.assertEqual(len(encoded['entity_spans']), 2)
        self.assertEqual([r['origins'][0]['slot'] for r in encoded['entity_spans']], [0, 2])
        self.assertEqual(audit['entities_kept'], 2)
        row['reference_triples'] = [[], [], []]
        encoded, audit = p.encode_row(row, CharacterTokenizer())
        self.assertFalse(encoded['has_entities'])
        self.assertEqual(len(encoded['knowledge']['input_ids']), 3)


if __name__ == '__main__':
    unittest.main()
