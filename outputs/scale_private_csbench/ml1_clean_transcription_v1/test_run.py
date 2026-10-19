import json
import unittest
from PIL import Image
from run import masked, parse

class Test(unittest.TestCase):
    def test_pixels_and_extra_notes(self):
        original = Image.new('RGB', (966, 614), (13, 24, 35))
        boxes = [[15, 38, 445, 590], [445, 30, 955, 335]]
        result = masked(original, boxes)
        for box in boxes:
            self.assertEqual(result.crop(box).tobytes(), original.crop(box).tobytes())
        for point in [(20, 605), (850, 530), (340, 20)]:
            self.assertEqual(result.getpixel(point), (255, 255, 255))
    def test_preserve_wrong_answer(self):
        obj = {'text':'F: 0, 0, 8, [无法辨认]', 'issues':['第四行无法辨认']}
        self.assertEqual(parse(json.dumps(obj)), obj)
    def test_reject_bad_schema_and_markings(self):
        for obj in [{'text':'评卷员：张', 'issues':[]}, {'text':'x', 'issues':'none'}, {'text':'', 'issues':[]}]:
            with self.assertRaises(AssertionError):
                parse(json.dumps(obj))

if __name__ == '__main__': unittest.main()
