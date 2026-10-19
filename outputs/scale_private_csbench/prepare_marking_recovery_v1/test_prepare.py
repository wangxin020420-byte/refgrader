import unittest
import prepare as m
class Tests(unittest.TestCase):
 def test_metadata_only(self):
  source='答案表格\n评卷员：张老师（06043），评分：1.5\n'
  clean,deleted=m.clean(source)
  self.assertEqual(clean,'答案表格\n');self.assertEqual(len(deleted),1)
 def test_ambiguous_suffix(self):
  source='答案\n评卷员：张老师 1960133 评分：302 即可正确 2个1'
  self.assertEqual(m.clean(source),(source,[]))
 def test_preserve_content(self):
  for source in ['1.（5分）答案','2+3=5\n+2','', '学生讨论评分：2']:
   self.assertEqual(m.clean(source),(source,[]))
 def test_no_cross_line_deletion(self):
  source='评卷员：姓名\n学生公式x=3\n评分：2'
  self.assertEqual(m.clean(source),(source,[]))
if __name__=='__main__':unittest.main()
