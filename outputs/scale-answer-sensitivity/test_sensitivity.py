import unittest
import sensitivity as m


class Checks(unittest.TestCase):
    def fixture(self):
        inputs=[dict(sample_id=str(i),question_id=str(i//2),question='q'+str(i//2),answer='answer'+str(i)) for i in range(4)]
        labels=[dict(sample_id=str(i),question_id=str(i//2),gold_raw=i) for i in range(4)]
        return inputs,labels

    def test_donors(self):
        inputs,labels=self.fixture();cases=m.design(inputs,labels)
        byid={r['sample_id']:r for r in inputs}
        for r in cases:
            self.assertNotEqual(r['sample_id'],r['same_donor_id'])
            self.assertEqual(r['question_id'],byid[r['same_donor_id']]['question_id'])
            self.assertNotEqual(r['question_id'],r['cross_donor_question_id'])
            self.assertEqual(r['answers']['empty'],'')
            self.assertEqual(r['answers']['original'],byid[r['sample_id']]['answer'])

    def test_order_independence(self):
        inputs,labels=self.fixture()
        self.assertEqual(m.design(inputs,labels),m.design(inputs[::-1],labels[::-1]))

    def test_invalid_ids(self):
        inputs,labels=self.fixture()
        with self.assertRaises(ValueError):m.design(inputs,labels[:-1])
        with self.assertRaises(ValueError):m.design(inputs+[inputs[0]],labels)

    def test_diagnostic_statistics(self):
        inputs,labels=self.fixture();cases=m.design(inputs,labels)
        for r in cases:
            r['scores']={k:dict(clipped=v,unclipped=v) for k,v in {
                'original':r['gold_original'],'same_question':r['same_donor_gold'],
                'empty':0,'cross_question':9}.items()}
        report=m.statistics_for(cases)
        self.assertEqual(report['same_question_direction']['fraction'],1)
        self.assertEqual(report['cross_question']['score_ge_8_count'],4)
        self.assertEqual(report['original']['mean_absolute_change'],0)


if __name__=='__main__':unittest.main()
