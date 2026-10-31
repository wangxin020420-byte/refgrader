"""Unlabelled, single-process developmental scoring integration.

Not a formal experiment protocol: input provenance, prompt and risk design,
visual extraction quality, and result-transfer handling remain to be audited.
The caller supplies a mock or real client and a frozen manifest. No evaluation
or teacher-score/review data is read here.
"""

from pathlib import Path
import re

from label_free_action import decide_action
from label_free_checkpoint import Checkpoint
from label_free_inputs import load_unlabeled_answers, load_unlabeled_question
from label_free_model import grade_one
from label_free_risk_candidates import evidence_availability_risk, item_score_range_risk
from label_free_bnd_diagnostic import diagnose_bnd
from label_free_question_snapshot import load_candidate_question
from scripts.verify_label_free_question_snapshot import verify


def _image(root, subdir, question_id, basename, *, required):
    if not re.fullmatch(r"[A-Za-z0-9_]+", question_id) or not re.fullmatch(r"[A-Za-z0-9_]+", basename):
        raise ValueError("Unsafe image identifier")
    folder = Path(root) / subdir
    candidates = [folder / (basename + suffix) for suffix in (".png", ".jpg", ".jpeg")]
    present = [p for p in candidates if p.is_file()]
    if len(present) > 1 or (required and len(present) != 1):
        raise ValueError(f"Ambiguous or missing image: {basename}")
    if not present:
        return None
    path = present[0]
    if path.is_symlink() or not path.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("Unsafe image path")
    return path


def run_questions(root, question_ids, split, checkpoint_dir, manifest, client, *, m, loss_parameters,
                  grader=grade_one, max_new=None, candidate_snapshot=None, risk_policy="baseline"):
    """Run with one writer; resume completed answers from the same manifest.

    A scoring failure is propagated and leaves that answer uncommitted. The next
    run retries it. This function deliberately does not catch/skip failures.
    """
    root = Path(root)
    if split not in {"calibration", "validation", "test"}:
        raise ValueError("Unsupported split")
    if not isinstance(question_ids, list) or not question_ids or len(set(question_ids)) != len(question_ids):
        raise ValueError("Question IDs must be a nonempty unique list")
    if max_new is not None and (isinstance(max_new, bool) or not isinstance(max_new, int) or max_new < 1):
        raise ValueError("max_new must be a positive integer")
    expected_config = {"question_ids": question_ids, "split": split, "m": m, "loss_parameters": loss_parameters}
    if risk_policy not in {"baseline", "candidate_max_v1"}:
        raise ValueError("Unsupported risk policy")
    if risk_policy != "baseline":
        expected_config["risk_policy"] = risk_policy
    candidates = {}
    if candidate_snapshot is not None:
        snapshot_report = verify(candidate_snapshot)
        expected_config["candidate_snapshot_sha256"] = snapshot_report["snapshot_sha256_normalized"]
        # Validate every requested question before making any model call.
        candidates = {qid: load_candidate_question(candidate_snapshot, qid) for qid in question_ids}
    if not isinstance(manifest, dict) or manifest.get("run_config") != expected_config:
        raise ValueError("Manifest does not match scoring configuration")
    checkpoint = Checkpoint(checkpoint_dir, manifest)
    counts = {"completed_before": 0, "completed_now": 0}
    for question_id in question_ids:
        if candidate_snapshot is None:
            inputs = load_unlabeled_question(root, question_id, split)
            context = {k: inputs[k] for k in ("question_id", "question_text", "reference_text", "total_score", "initial_rubric")}
            q_image = _image(root, "reference_images/questions", question_id, question_id, required=False)
            ref_args = {"reference_image": _image(root, "reference_images/standard_answers", question_id, question_id, required=False)}
            answers = inputs["answers"]
        else:
            candidate = candidates[question_id]
            context, q_image = candidate["context"], candidate["question_image"]
            ref_args = {"reference_images": candidate["reference_images"]}
            answers = load_unlabeled_answers(root, question_id, split)
        for answer in answers:
            answer_id = answer["answer_id"]
            key = question_id + "_" + answer_id
            if checkpoint.load(key) is not None:
                counts["completed_before"] += 1
                continue
            if max_new is not None and counts["completed_now"] >= max_new:
                return counts
            image = _image(root, "student_images/" + question_id, question_id, answer_id, required=True)
            observation = grader(client, context, image, question_image=q_image, **ref_args)
            if not isinstance(observation, dict) or observation.get("valid_probe_count") != 3:
                raise ValueError("Invalid scoring observation")
            risks = observation["risks"]
            evidence_states = None
            if risk_policy == "candidate_max_v1":
                limits = {item["id"]: item["points"] for item in context["initial_rubric"]}
                states = {"present": "readable", "missing": "unlocated", "unclear": "unreadable"}
                facts = observation["facts"]
                if set(facts) != set(limits):
                    raise ValueError("Evidence/rubric item mismatch")
                evidence_states = {}
                for item, fact in facts.items():
                    if fact.get("status") not in states:
                        raise ValueError("Unexpected extractor evidence state")
                    evidence_states[item] = {"state": states[fact["status"]], "source": "model_extraction",
                                             "description": "Model report; absence is not independently verified"}
                risks = {"U_E": evidence_availability_risk(evidence_states, limits),
                         "U_S": item_score_range_risk(observation["probes"], limits),
                         "U_R": observation["risks"]["U_R"]}
            action = decide_action(baseline_score=observation["baseline_score"], max_score=context["total_score"],
                                   risks=risks, valid_probe_count=3, m=m, loss_parameters=loss_parameters,
                                   fusion="equal_mean" if risk_policy == "baseline" else "noncompensatory_max")
            if "slot_fallback" in observation:
                fallback = observation["slot_fallback"]
                if (not isinstance(fallback, dict) or fallback.get("protocol") != "cpl5_visible_slots_candidate_v1" or
                        fallback.get("requires_human_review") is not True or fallback.get("score_change_authorized") is not False):
                    raise ValueError("Invalid fallback review contract")
                # Preserve the computed A3WA route; override only the release
                # action because mapped slots and crossouts are unverified.
                action["automatic_score"] = None
                action["requires_human_review"] = True
                action["action"] = "defer_unverified_slot_fallback"
                action["fallback_review_required"] = True
            if 'reference_fallback' in observation:
                reference = observation['reference_fallback']
                if (not isinstance(reference, dict) or reference.get('protocol') != 'marked_reference_transcription_v1' or
                        reference.get('requires_human_review') is not True or reference.get('score_change_authorized') is not False):
                    raise ValueError('Invalid reference fallback review contract')
                action['automatic_score'] = None
                action['requires_human_review'] = True
                action['action'] = 'defer_unverified_reference_fallback'
                action['reference_fallback_review_required'] = True
            if evidence_states is not None:
                action["candidate_risks"] = risks
                action["evidence_states"] = evidence_states
                if action["route"] == "BND":
                    action["bnd_diagnostic"] = diagnose_bnd(decision=action, observation=observation,
                                                            rubric=context["initial_rubric"])
            checkpoint.save(key, {"question_id": question_id, "answer_id": answer_id,
                                  "observation": observation, "decision": action})
            counts["completed_now"] += 1
            print(f"completed question={question_id} answer={answer_id} route={action['route']}", flush=True)
    return counts
