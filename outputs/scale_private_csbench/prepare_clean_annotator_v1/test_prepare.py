import unittest
from prepare import score,index,labeled
class Tests(unittest.TestCase):
    def test_score_scale(self):
        self.assertEqual(score(dict(score_original=3,score_max_original=4,score_0_10=7.5)),7.5)
        with self.assertRaises(AssertionError):score(dict(score_original=3,score_max_original=4,score_0_10=3))
    def test_duplicates(self):
        with self.assertRaises(AssertionError):index([{'sample_id':'x'},{'sample_id':'x'}])
    def test_no_text_correction(self):
        r=dict(sample_id='x',question_id='q',question='?',references=['a','b'],blank_passthrough=False)
        g=dict(question_id='q',score_original=0,score_max_original=4,score_0_10=0)
        out=labeled(r,g,'O ( ♣1','unverified');self.assertEqual(out['answer'],'O ( ♣1');self.assertEqual(out['references'],['a','b'])
if __name__=='__main__':unittest.main()
