import csv
from pathlib import Path
import tempfile
import unittest

from scripts.compare_reviewed_history import KEYS, SCENARIOS, bootstrap_paired, decompose, load_rows


def row(teacher, prediction):
    return {"teacher": teacher, **dict.fromkeys(KEYS, prediction)}


class CompareReviewedHistoryTests(unittest.TestCase):
    def fixture(self):
        old = {("Q_1", "ANS_1"):row(0, 0), ("Q_1", "ANS_2"):row(2, 2),
               ("Q_1", "ANS_3"):row(4, 1), ("Q_2", "ANS_4"):row(0, 0),
               ("Q_2", "ANS_5"):row(4, 3)}
        new = {key: dict(value) for key, value in old.items() if key != ("Q_1", "ANS_3")}
        new[("Q_2", "ANS_5")] = row(3, 3)
        return old, new, {"Q_1":4, "Q_2":4}, {"excluded":{"Q_1":{"ANS_3":"excluded"}}, "corrected_scores":{"Q_2":{"ANS_5":3}}}

    def test_exact_decomposition_and_no_mutation(self):
        old, new, maxima, policy = self.fixture()
        report, _, changes, datasets = decompose(old, new, maxima, policy)
        self.assertEqual(report["counts"], {"old_test":5,"retained_test":4,"excluded_test":1,"relabelled_test":1})
        self.assertEqual(old[("Q_2","ANS_5")]["teacher"],4)
        self.assertEqual(len(changes),1)
        for key in KEYS:
            effects=report["effects"][key]
            net=report["scenarios"][SCENARIOS[0]][key]["mae"]-report["scenarios"][SCENARIOS[-1]][key]["mae"]
            self.assertAlmostEqual(sum(x["mae_reduction"] for x in effects),net)
        result=bootstrap_paired(datasets[SCENARIOS[2]],new,maxima,iterations=20)
        self.assertEqual(result["scores"][KEYS[0]]["mae_reduction"],0)

    def test_unexplained_sample_removal_is_rejected(self):
        old,new,maxima,policy=self.fixture()
        policy["excluded"]={}
        with self.assertRaisesRegex(ValueError,"Removed IDs"):
            decompose(old,new,maxima,policy)

    def test_unexplained_label_change_is_rejected(self):
        old,new,maxima,policy=self.fixture()
        policy["corrected_scores"]={}
        with self.assertRaisesRegex(ValueError,"label does not match"):
            decompose(old,new,maxima,policy)

    def test_new_answer_id_is_rejected(self):
        old,new,maxima,policy=self.fixture()
        new[("Q_1","unexpected")]=row(0,0)
        with self.assertRaisesRegex(ValueError,"subset"):
            decompose(old,new,maxima,policy)

    def test_full_answer_ids_are_not_truncated(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"compare.csv"
            with path.open('w',newline='',encoding='utf-8') as handle:
                writer=csv.DictWriter(handle,fieldnames=['question','student_id','teacher',*KEYS])
                writer.writeheader()
                for sid in ('ANS_Q_01','ANS_Q_02'):
                    writer.writerow({'question':'Q_1','student_id':sid,**row(1,1)})
            self.assertEqual(len(load_rows(path)),2)


if __name__=='__main__':
    unittest.main()
