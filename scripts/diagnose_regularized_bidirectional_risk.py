import argparse
import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnose_bidirectional_credit_risk import (  # noqa: E402
    build_rows,
    load_json,
)


UNDER_FEATURES = (
    "U_E",
    "U_S",
    "U_R",
    "U_R_undercredit_existing",
    "U_R_allocation_undercredit",
    "U_R_allocation_disagreement",
    "U_R_deterministic_undercredit",
    "missing_judgement_risk",
)

OVER_FEATURES = (
    "U_E",
    "U_S",
    "U_R",
    "U_R_allocation_overcredit",
    "U_R_allocation_disagreement",
    "U_R_deterministic_overcredit",
    "missing_judgement_risk",
)


def _dependencies():
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RuntimeError(
            "scikit-learn is required for the regularized risk diagnostic"
        ) from exc
    return LogisticRegression, make_pipeline, StandardScaler


def _matrix(rows, feature_names):
    return [
        [float(row.get(feature, 0.0) or 0.0) for feature in feature_names]
        for row in rows
    ]


class _ConstantProbabilityModel:
    def __init__(self, probability):
        self.probability = float(probability)

    def predict_positive(self, rows, feature_names):
        return [self.probability] * len(rows)


class _LogisticProbabilityModel:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def predict_positive(self, rows, feature_names):
        matrix = _matrix(rows, feature_names)
        return [float(value) for value in self.pipeline.predict_proba(matrix)[:, 1]]


def fit_probability_model(rows, label_name, feature_names, regularization_c=1.0):
    labels = [bool(row[label_name]) for row in rows]
    positive_count = sum(labels)
    if positive_count == 0 or positive_count == len(labels):
        probability = (positive_count + 1.0) / (len(labels) + 2.0)
        return _ConstantProbabilityModel(probability)

    LogisticRegression, make_pipeline, StandardScaler = _dependencies()
    pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=float(regularization_c),
            class_weight="balanced",
            max_iter=2000,
            random_state=0,
            solver="liblinear",
        ),
    )
    pipeline.fit(_matrix(rows, feature_names), labels)
    return _LogisticProbabilityModel(pipeline)


def predict_directional_risk(
    train_rows,
    predict_rows,
    regularization_c=1.0,
):
    under_model = fit_probability_model(
        train_rows,
        "undercredit_unsafe",
        UNDER_FEATURES,
        regularization_c=regularization_c,
    )
    over_model = fit_probability_model(
        train_rows,
        "overcredit_unsafe",
        OVER_FEATURES,
        regularization_c=regularization_c,
    )
    under_probabilities = under_model.predict_positive(
        predict_rows, UNDER_FEATURES
    )
    over_probabilities = over_model.predict_positive(
        predict_rows, OVER_FEATURES
    )
    return [
        {
            "undercredit_probability": under,
            "overcredit_probability": over,
            "combined_probability": max(under, over),
        }
        for under, over in zip(under_probabilities, over_probabilities)
    ]


def select_threshold(
    probabilities,
    unsafe_labels,
    max_bnd_ratio=0.60,
    max_unsafe_pos_rate=0.10,
):
    candidates = []
    for threshold in sorted(set(probabilities)):
        pos_indexes = [
            index
            for index, probability in enumerate(probabilities)
            if probability <= threshold
        ]
        if not pos_indexes:
            continue
        bnd_ratio = 1.0 - len(pos_indexes) / len(probabilities)
        unsafe_pos_rate = (
            sum(unsafe_labels[index] for index in pos_indexes)
            / len(pos_indexes)
        )
        bnd_excess = max(0.0, bnd_ratio - max_bnd_ratio)
        unsafe_excess = max(
            0.0, unsafe_pos_rate - max_unsafe_pos_rate
        )
        candidates.append({
            "threshold": threshold,
            "bnd_ratio": bnd_ratio,
            "unsafe_pos_rate": unsafe_pos_rate,
            "feasible": bnd_excess <= 1e-12 and unsafe_excess <= 1e-12,
            "constraint_count": int(bnd_excess > 1e-12)
            + int(unsafe_excess > 1e-12),
            "constraint_excess": bnd_excess + unsafe_excess,
        })
    return min(
        candidates,
        key=lambda item: (
            item["constraint_count"],
            item["constraint_excess"],
            item["unsafe_pos_rate"],
            item["bnd_ratio"],
            item["threshold"],
        ),
    )


def inner_grouped_probabilities(rows, regularization_c=1.0):
    questions = sorted({row["question_id"] for row in rows})
    if len(questions) < 2:
        raise ValueError("Inner grouped predictions require at least two questions")
    predictions = []
    for heldout_question in questions:
        inner_train = [
            row for row in rows if row["question_id"] != heldout_question
        ]
        inner_heldout = [
            row for row in rows if row["question_id"] == heldout_question
        ]
        directional = predict_directional_risk(
            inner_train,
            inner_heldout,
            regularization_c=regularization_c,
        )
        predictions.extend(
            {
                "question_id": row["question_id"],
                "student_id": row["student_id"],
                "unsafe": bool(row["unsafe"]),
                **probability,
            }
            for row, probability in zip(inner_heldout, directional)
        )
    return predictions


def nested_grouped_oof(
    rows,
    max_bnd_ratio=0.60,
    max_unsafe_pos_rate=0.10,
    regularization_c=1.0,
):
    questions = sorted({row["question_id"] for row in rows})
    if len(questions) < 3:
        raise ValueError("Nested grouped OOF requires at least three questions")

    folds = []
    oof_rows = []
    for heldout_question in questions:
        outer_train = [
            row for row in rows if row["question_id"] != heldout_question
        ]
        outer_heldout = [
            row for row in rows if row["question_id"] == heldout_question
        ]
        inner_predictions = inner_grouped_probabilities(
            outer_train,
            regularization_c=regularization_c,
        )
        threshold = select_threshold(
            [row["combined_probability"] for row in inner_predictions],
            [row["unsafe"] for row in inner_predictions],
            max_bnd_ratio=max_bnd_ratio,
            max_unsafe_pos_rate=max_unsafe_pos_rate,
        )
        outer_predictions = predict_directional_risk(
            outer_train,
            outer_heldout,
            regularization_c=regularization_c,
        )
        heldout_pos = 0
        heldout_unsafe_pos = 0
        for row, prediction in zip(outer_heldout, outer_predictions):
            route = (
                "POS"
                if prediction["combined_probability"] <= threshold["threshold"]
                else "BND"
            )
            heldout_pos += int(route == "POS")
            heldout_unsafe_pos += int(route == "POS" and row["unsafe"])
            oof_rows.append({
                **row,
                **prediction,
                "oof_threshold": threshold["threshold"],
                "oof_route": route,
                "inner_threshold_feasible": threshold["feasible"],
            })
        folds.append({
            "heldout_question": heldout_question,
            "outer_train_n": len(outer_train),
            "heldout_n": len(outer_heldout),
            "inner_threshold": threshold["threshold"],
            "inner_feasible": threshold["feasible"],
            "inner_bnd_ratio": threshold["bnd_ratio"],
            "inner_unsafe_pos_rate": threshold["unsafe_pos_rate"],
            "heldout_pos_n": heldout_pos,
            "heldout_unsafe_pos_n": heldout_unsafe_pos,
        })

    pos_rows = [row for row in oof_rows if row["oof_route"] == "POS"]
    bnd_ratio = 1.0 - len(pos_rows) / len(oof_rows)
    unsafe_pos_rate = (
        sum(row["unsafe"] for row in pos_rows) / len(pos_rows)
        if pos_rows else 1.0
    )
    constraints = {
        "max_bnd_ratio": max_bnd_ratio,
        "max_unsafe_pos_rate": max_unsafe_pos_rate,
        "bnd_ratio_within_budget": bnd_ratio <= max_bnd_ratio + 1e-12,
        "unsafe_pos_rate_within_budget": (
            unsafe_pos_rate <= max_unsafe_pos_rate + 1e-12
        ),
    }
    report = {
        "method": "nested_leave_one_question_out_regularized_logistic",
        "status": "diagnostic_only",
        "regularization_c": regularization_c,
        "n": len(oof_rows),
        "question_count": len(questions),
        "fold_count": len(folds),
        "inner_feasible_fold_count": sum(
            fold["inner_feasible"] for fold in folds
        ),
        "pos_n": len(pos_rows),
        "bnd_n": len(oof_rows) - len(pos_rows),
        "bnd_ratio": bnd_ratio,
        "unsafe_pos_n": sum(row["unsafe"] for row in pos_rows),
        "unsafe_pos_rate": unsafe_pos_rate,
        "constraints": constraints,
        "passed": all(
            constraints[key]
            for key in (
                "bnd_ratio_within_budget",
                "unsafe_pos_rate_within_budget",
            )
        ),
        "features": {
            "undercredit": list(UNDER_FEATURES),
            "overcredit": list(OVER_FEATURES),
        },
    }
    return report, folds, oof_rows


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-bnd-ratio", type=float, default=0.60)
    parser.add_argument("--max-unsafe-pos-rate", type=float, default=0.10)
    parser.add_argument("--regularization-c", type=float, default=1.0)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    rows = build_rows(load_json(config_path), config_path)
    report, folds, oof_rows = nested_grouped_oof(
        rows,
        max_bnd_ratio=args.max_bnd_ratio,
        max_unsafe_pos_rate=args.max_unsafe_pos_rate,
        regularization_c=args.regularization_c,
    )
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv(output_dir / "folds.csv", folds)
    write_csv(output_dir / "cases.csv", oof_rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Report: {output_dir / 'summary.json'}")
    print(f"Folds: {output_dir / 'folds.csv'}")
    print(f"Cases: {output_dir / 'cases.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
