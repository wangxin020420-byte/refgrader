import unittest
import audit_marking_lineage as a
class Tests(unittest.TestCase):
 def test_explicit_vs_question_points(self):
  self.assertEqual(a.scan_record({'answer':'1. (5分) 2+3=5'},['answer']),[])
  self.assertTrue(a.scan_record({'answer':'评卷员：某人 评分：1.5'},['answer']))
 def test_ignore_score_field(self):
  self.assertEqual(a.scan_record({'source_score':2,'answer':'学生答案'},['answer']),[])
 def test_nested_request(self):
  self.assertTrue(a.scan_record({'messages':[{'content':'得分：2'}]},['messages']))
if __name__=='__main__':unittest.main()
