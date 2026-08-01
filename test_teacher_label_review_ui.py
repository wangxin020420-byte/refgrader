import csv
import json
import tempfile
import unittest
from pathlib import Path

from sample_quality import default_policy_path
from scripts.review_teacher_labels import TeacherLabelReviewStore


class TeacherLabelReviewStoreTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.prepared = self.root / "data" / "csbench"
        self.report = (
            self.prepared
            / "quality_control"
            / "reports"
            / "review_run"
        )
        self.decisions = (
            self.prepared
            / "quality_control"
            / "reviews"
            / "review_run_decisions.jsonl"
        )
        image = self.prepared / "student_images" / "CO_1" / "A.jpg"
        image.parent.mkdir(parents=True)
        image.write_bytes(b"not-a-real-jpeg")
        self.image = image
        question_image = (
            self.prepared / "reference_images" / "questions" / "CO_1.png"
        )
        question_image.parent.mkdir(parents=True)
        question_image.write_bytes(b"not-a-real-png")
        self.question_image = question_image
        optimized_rubric = (
            self.prepared
            / "rubrics"
            / "optimized"
            / "CO"
            / "CO_1_rubric_standard.json"
        )
        optimized_rubric.parent.mkdir(parents=True)
        optimized_rubric.write_text(
            json.dumps(
                [
                    {
                        "id": "step_1",
                        "item": "final answer",
                        "points": 5,
                        "standard_answer_text": "37H",
                        "role": "final",
                        "task_semantics": "conclusion_dominant",
                    }
                ]
            ),
            encoding="utf-8",
        )
        (self.prepared / "rubrics" / "active_rubric_set.json").write_text(
            json.dumps(
                {
                    "questions": {
                        "CO_1": {
                            "optimized_rubric": (
                                "rubrics/optimized/CO/"
                                "CO_1_rubric_standard.json"
                            )
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        (self.prepared / "answer_metadata.jsonl").parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        (self.prepared / "answer_metadata.jsonl").write_text(
            json.dumps(
                {
                    "answer_id": "A",
                    "question_id": "CO_1",
                    "raw_text": "37H",
                    "student_image": str(image),
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (self.prepared / "exam_database.json").write_text(
            json.dumps(
                [
                    {
                        "question_id": "CO_1",
                        "total_score": 5,
                        "question_text": "Find the answer.",
                        "question_image": str(question_image),
                        "ref_text": "37H",
                        "official_rubric": "Correct answer: 5 points.",
                        "optimized_rubric_path": str(optimized_rubric),
                    }
                ]
            ),
            encoding="utf-8",
        )
        (self.prepared / "teacher_scores.json").write_text(
            json.dumps({"A": {"CO_1": 2}}),
            encoding="utf-8",
        )
        self.write_candidates(
            "model_avg_score",
            {
                "question_id": "CO_1",
                "answer_id": "A",
                "review_priority": "P1",
                "candidate_type": "possible_teacher_under_score",
                "teacher_score": "2",
                "reference_score_key": "model_avg_score",
                "reference_score": "5",
                "teacher_minus_reference": "-3",
                "absolute_difference": "3",
                "max_score": "5",
                "model_avg_score": "5",
                "route": "POS",
                "extraction_quality": "high",
                "U_E": "0",
                "U_S": "0.1",
                "U_R": "0.2",
                "student_image": str(image),
            },
        )
        self.write_candidates(
            "three_way_core_score",
            {
                "question_id": "CO_1",
                "answer_id": "A",
                "review_priority": "P2",
                "candidate_type": "possible_teacher_under_score",
                "teacher_score": "2",
                "reference_score_key": "three_way_core_score",
                "reference_score": "4.5",
                "teacher_minus_reference": "-2.5",
                "absolute_difference": "2.5",
                "max_score": "5",
                "three_way_core_score": "4.5",
                "final_calibrated_score": "4.7",
                "route": "BND",
                "student_image": str(image),
            },
        )

    def write_candidates(self, source, row):
        path = self.report / "batch" / source / "teacher_label_candidates.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)

    def make_store(self):
        return TeacherLabelReviewStore(
            prepared_dir=self.prepared,
            report_dir=self.report,
            decisions_path=self.decisions,
        )

    def test_duplicate_candidate_sources_are_merged(self):
        store = self.make_store()
        self.assertEqual(len(store.candidates), 1)
        candidate = store.candidates[0]
        self.assertEqual(candidate["review_priority"], "P1")
        self.assertEqual(candidate["model_avg_score"], 5.0)
        self.assertEqual(candidate["three_way_core_score"], 4.5)
        self.assertEqual(
            set(candidate["candidate_sources"]),
            {"model_avg_score", "three_way_core_score"},
        )
        self.assertTrue(candidate["image_available"])

    def test_state_contains_question_reference_and_active_rubric(self):
        store = self.make_store()
        context = store.state()["question_contexts"]["CO_1"]
        self.assertEqual(context["question_text"], "Find the answer.")
        self.assertEqual(context["reference_answer"], "37H")
        self.assertEqual(
            context["official_rubric"],
            "Correct answer: 5 points.",
        )
        self.assertTrue(context["question_image_available"])
        self.assertEqual(
            context["active_rubric"]["source"],
            "active_rubric_set",
        )
        self.assertEqual(
            context["active_rubric"]["items"][0]["id"],
            "step_1",
        )
        self.assertEqual(
            context["active_rubric"]["items"][0]["points"],
            5.0,
        )

    def test_question_image_is_limited_to_reference_images(self):
        store = self.make_store()
        self.assertEqual(
            store.resolve_question_image_path("CO_1"),
            self.question_image.resolve(),
        )
        store.questions["CO_1"]["question_image"] = str(
            self.root / "outside-question.png"
        )
        (self.root / "outside-question.png").write_bytes(b"x")
        self.assertIsNone(store.resolve_question_image_path("CO_1"))

    def test_preliminary_screening_can_limit_manual_queue(self):
        screening = (
            self.prepared
            / "quality_control"
            / "reviews"
            / "review_run_initial_screening.csv"
        )
        screening.parent.mkdir(parents=True, exist_ok=True)
        screening.write_text(
            "question_id,answer_id,initial_class,confidence,"
            "initial_finding,recommended_action\n"
            "CO_1,A,teacher_label_high_suspicion,high,"
            "check label,manual review\n",
            encoding="utf-8",
        )
        store = TeacherLabelReviewStore(
            prepared_dir=self.prepared,
            report_dir=self.report,
            decisions_path=self.decisions,
            screening_path=screening,
            screening_only=True,
        )
        self.assertEqual(len(store.candidates), 1)
        self.assertEqual(
            store.candidates[0]["initial_class"],
            "teacher_label_high_suspicion",
        )
        self.assertTrue(store.state()["screening_only"])

    def test_decision_is_upserted_atomically(self):
        store = self.make_store()
        store.save_decision(
            {
                "question_id": "CO_1",
                "answer_id": "A",
                "decision": "ambiguous",
                "reviewer": "tester",
            }
        )
        store.save_decision(
            {
                "question_id": "CO_1",
                "answer_id": "A",
                "decision": "corrected",
                "corrected_score": 3.5,
                "reason_code": "teacher_score_suspected",
                "review_note": "reviewed",
                "reviewer": "tester",
            }
        )
        rows = [
            json.loads(line)
            for line in self.decisions.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["decision"], "corrected")
        self.assertEqual(rows[0]["corrected_score"], 3.5)
        self.assertEqual(store.state()["counts"]["reviewed"], 1)

    def test_invalid_corrected_score_is_rejected(self):
        store = self.make_store()
        with self.assertRaisesRegex(ValueError, "between 0 and 5"):
            store.save_decision(
                {
                    "question_id": "CO_1",
                    "answer_id": "A",
                    "decision": "corrected",
                    "corrected_score": 8,
                }
            )
        self.assertFalse(self.decisions.exists())

    def test_confirmed_noise_can_activate_existing_policy_format(self):
        store = self.make_store()
        store.save_decision(
            {
                "question_id": "CO_1",
                "answer_id": "A",
                "decision": "confirmed_noise",
                "reason_code": "teacher_score_suspected",
                "reviewer": "tester",
            }
        )
        result = store.activate_policy("review-policy")
        policy = json.loads(
            default_policy_path(self.prepared).read_text(encoding="utf-8")
        )
        self.assertEqual(result["excluded"], 1)
        self.assertIn("A", policy["excluded"]["CO_1"])
        self.assertEqual(policy["policy_id"], "review-policy")

    def test_image_outside_prepared_student_images_is_not_served(self):
        store = self.make_store()
        candidate = dict(store.candidates[0])
        candidate["student_image"] = str(self.root / "outside.jpg")
        (self.root / "outside.jpg").write_bytes(b"x")
        self.assertIsNone(store.resolve_image_path(candidate))


if __name__ == "__main__":
    unittest.main()
