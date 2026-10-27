"""Offline-composable scoring boundary, not a live experiment runner.

The caller supplies a grading function. This module does not guarantee its
provenance, audit image/text contents, invoke a real model, or save results.
BND/NEG remain deferred under the conservative action policy.
"""
from copy import deepcopy

from label_free_action import decide_action
from label_free_inputs import load_unlabeled_question


def score_question_with_grader(root, question_id, split, grader, *, m, loss_parameters):
    """Grade validated inputs with a caller-provided grader and fixed policy.

    Only the initial rubric is loaded; no teacher score or review policy is
    read by this boundary. Callers must validate grader/config/input provenance.
    Any invalid output stops the batch before a result is returned.
    """
    if not callable(grader):
        raise ValueError("grader must be callable")
    inputs = load_unlabeled_question(root, question_id, split)
    context = {
        "question_id": inputs["question_id"],
        "question_text": inputs["question_text"],
        "reference_text": inputs["reference_text"],
        "total_score": inputs["total_score"],
        "initial_rubric": inputs["initial_rubric"],
    }
    results = []
    for answer in inputs["answers"]:
        # The grader receives only the validated answer record and question
        # context. Text/image contents still require independent label audit.
        grade = grader(deepcopy(context), deepcopy(answer))
        required = {"baseline_score", "risks", "valid_probe_count"}
        if not isinstance(grade, dict) or set(grade) != required:
            raise ValueError("Grader output must contain exactly baseline_score, risks and valid_probe_count")
        result = decide_action(
            baseline_score=grade["baseline_score"], max_score=inputs["total_score"],
            risks=grade["risks"], valid_probe_count=grade["valid_probe_count"],
            m=m, loss_parameters=loss_parameters,
        )
        results.append({"answer_id": answer["answer_id"], **result})
    return {
        "question_id": question_id,
        "split": split,
        "results": results,
        "teacher_labels_loaded_by_boundary": False,
        "grader_provenance_audited": False,
        "input_content_audited": False,
        "ready_for_live_scoring": False,
    }
