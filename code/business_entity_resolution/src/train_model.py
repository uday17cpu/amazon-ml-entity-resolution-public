"""Train a compact pair classifier on a deterministic sample of training S1 IDs.

The candidate index must be built from train_source2.tsv and train_source3.tsv.
Only candidate pairs are used as negatives, producing realistic hard negatives.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from common import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    find_candidates,
    iter_tsv,
    make_record,
    open_index,
    parse_id_list,
    source_from_id,
    stable_fraction,
)


def entity_f05(predicted: set[str], actual: set[str]) -> float:
    if not predicted and not actual:
        return 1.0
    if not predicted or not actual:
        return 0.0
    true_positives = len(predicted & actual)
    precision = true_positives / len(predicted)
    recall = true_positives / len(actual)
    if precision == 0 or recall == 0:
        return 0.0
    return 1.25 * precision * recall / (0.25 * precision + recall)


def select_threshold(
    model: HistGradientBoostingClassifier,
    validation_examples: list[tuple[set[str], list[tuple[str, list[float]]]]],
) -> tuple[float, float]:
    """Choose a precision-first threshold using complete Source-1 validation rows."""
    candidate_features = [
        features
        for _, candidates in validation_examples
        for _, features in candidates
    ]
    candidate_probabilities = (
        model.predict_proba(np.asarray(candidate_features, dtype=np.float32))[:, 1]
        if candidate_features
        else np.asarray([], dtype=np.float32)
    )
    probability_index = 0
    # A fixed grid is reproducible and avoids an expensive scan over every
    # distinct probability on a large validation candidate set.
    thresholds = np.linspace(0.05, 0.99, 95)
    best_threshold, best_score = 0.90, -1.0
    for threshold in thresholds:
        scores = []
        for actual, candidates in validation_examples:
            if candidates:
                probabilities = candidate_probabilities[probability_index : probability_index + len(candidates)]
                probability_index += len(candidates)
                predicted = {
                    entity_id
                    for (entity_id, _), probability in zip(candidates, probabilities)
                    if probability >= threshold
                }
            else:
                predicted = set()
            scores.append(entity_f05(predicted, actual))
        probability_index = 0
        score = float(np.mean(scores)) if scores else 0.0
        if score > best_score or (score == best_score and threshold > best_threshold):
            best_threshold, best_score = float(threshold), score
    return best_threshold, best_score


def selected_ground_truth(ground_truth: Path, sample_rate: float) -> dict[str, set[str]]:
    labels: dict[str, set[str]] = {}
    for row in iter_tsv(ground_truth, ("source1_entity_id", "matched_entity_ids")):
        entity_id = row["source1_entity_id"]
        if not entity_id or source_from_id(entity_id) != "S1":
            raise ValueError(f"{ground_truth}: invalid Source-1 label ID {entity_id!r}.")
        if entity_id in labels:
            raise ValueError(f"{ground_truth}: duplicate Source-1 label row {entity_id!r}.")
        if stable_fraction(entity_id) < sample_rate:
            targets = parse_id_list(row.get("matched_entity_ids"))
            invalid_targets = sorted(
                target
                for target in targets
                if source_from_id(target) not in {"S2", "S3"}
            )
            if invalid_targets:
                raise ValueError(
                    f"{ground_truth}: labels for {entity_id!r} include non-target IDs: "
                    f"{', '.join(invalid_targets[:5])}."
                )
            labels[entity_id] = targets
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source1", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--model-out", type=Path, required=True)
    parser.add_argument("--sample-rate", type=float, default=0.01, help="Fraction of S1 labels to sample.")
    parser.add_argument("--max-block-size", type=int, default=500)
    parser.add_argument("--max-candidates", type=int, default=75)
    parser.add_argument("--negative-per-source", type=int, default=4)
    args = parser.parse_args()

    if not 0 < args.sample_rate <= 1:
        raise SystemExit("--sample-rate must be in (0, 1].")
    if args.max_block_size < 1 or args.max_candidates < 1 or args.negative_per_source < 0:
        raise SystemExit(
            "--max-block-size and --max-candidates must be positive; "
            "--negative-per-source cannot be negative."
        )
    truth = selected_ground_truth(args.ground_truth, args.sample_rate)
    print(f"Sampled {len(truth):,} Source-1 label rows.")

    features: list[list[float]] = []
    labels: list[int] = []
    training_entities_with_pairs: set[str] = set()
    examples_by_group: dict[str, tuple[set[str], list[tuple[str, list[float]]]]] = {}
    true_links = retrieved_links = candidate_pairs = processed = 0
    selected_ids = sorted(truth)
    if len(selected_ids) < 2:
        raise SystemExit("Training sample must contain at least two Source-1 entities.")
    validation_groups = {
        entity_id
        for entity_id in selected_ids
        if stable_fraction(entity_id + "-validation") < 0.20
    }
    if not validation_groups or len(validation_groups) == len(selected_ids):
        validation_groups = {selected_ids[-1]}
    training_groups = set(selected_ids) - validation_groups

    connection = open_index(args.index, args.max_block_size)
    processed_source1_ids: set[str] = set()
    try:
        target_record_count = connection.execute(
            "SELECT COUNT(*) FROM records WHERE source IN ('S2', 'S3')"
        ).fetchone()[0]
        for row in iter_tsv(
            args.source1, ("entity_id", "business_name", "business_address", "country")
        ):
            source1_id = row["entity_id"]
            if not source1_id or source_from_id(source1_id) != "S1":
                raise ValueError(f"{args.source1}: invalid Source-1 ID {source1_id!r}.")
            if source1_id not in truth:
                continue
            if source1_id in processed_source1_ids:
                raise ValueError(f"{args.source1}: duplicate Source-1 ID {source1_id!r}.")
            processed_source1_ids.add(source1_id)
            query = make_record(row)
            if query.source != "S1":
                raise ValueError(f"{args.source1}: expected S1- ID, got {source1_id!r}.")
            target_ids = truth[source1_id]
            candidates = find_candidates(connection, query, args.max_block_size, args.max_candidates)
            candidate_ids = {candidate.entity_id for candidate, _, _ in candidates}
            true_links += len(target_ids)
            retrieved_links += len(target_ids & candidate_ids)
            candidate_pairs += len(candidates)
            if source1_id in validation_groups:
                examples_by_group[source1_id] = (
                    target_ids,
                    [(candidate.entity_id, vector) for candidate, _, vector in candidates],
                )
            elif source1_id in training_groups:
                negatives_added = 0
                # Candidates are already ordered by a cheap similarity score, so
                # early non-matches are deliberately hard negatives.
                for candidate, _, vector in candidates:
                    is_positive = candidate.entity_id in target_ids
                    if not is_positive and negatives_added >= args.negative_per_source:
                        continue
                    features.append(vector)
                    labels.append(int(is_positive))
                    training_entities_with_pairs.add(source1_id)
                    if not is_positive:
                        negatives_added += 1
            processed += 1
            if processed % 10_000 == 0:
                print(f"  processed {processed:,} sampled S1 records", flush=True)
    finally:
        connection.close()

    missing_source1_ids = set(selected_ids) - processed_source1_ids
    if missing_source1_ids:
        raise ValueError(
            f"{args.source1}: {len(missing_source1_ids):,} sampled label IDs are missing, "
            f"including {', '.join(sorted(missing_source1_ids)[:5])}."
        )
    if not features or len(set(labels)) < 2:
        raise SystemExit("Training sample contains only one class. Increase --sample-rate or improve blocking.")

    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int8)
    if not training_entities_with_pairs:
        raise SystemExit("Training split has no candidate pairs; increase --sample-rate or improve blocking.")
    model = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=250,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=42,
    )
    model.fit(x, y)
    validation_examples = [
        example for group, example in examples_by_group.items() if group in validation_groups
    ]
    selected_threshold, validation_f05 = select_threshold(model, validation_examples)
    print(f"Training pairs: {len(y):,}; positive rate: {y.mean():.3f}")
    print(
        f"Training Source-1 entities with candidate pairs: {len(training_entities_with_pairs):,} "
        f"of {len(training_groups):,}; "
        f"validation Source-1 entities: {len(validation_examples):,}"
    )
    print(f"Sampled-row candidate recall: {retrieved_links / max(true_links, 1):.4f}")
    possible_pairs = processed * target_record_count
    reduction = 1.0 - candidate_pairs / possible_pairs if possible_pairs else 0.0
    print(
        f"Final candidate pool: {candidate_pairs:,} pairs from {processed:,} sampled S1 rows "
        f"({candidate_pairs / max(processed, 1):.2f} average; "
        f"{reduction:.6%} reduction vs. {possible_pairs:,}-pair cross product)"
    )
    print(
        f"Validation macro F0.5 across all {len(validation_examples):,} held-out S1 entities: "
        f"{validation_f05:.4f}; selected threshold: {selected_threshold:.4f}"
    )

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_names": FEATURE_NAMES,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "threshold": selected_threshold,
            "sample_rate": args.sample_rate,
            "validation_source1_count": len(validation_examples),
            "validation_macro_f0_5": validation_f05,
            "max_block_size": args.max_block_size,
            "max_candidates": args.max_candidates,
        },
        args.model_out,
    )
    print(f"Saved model bundle to {args.model_out}")


if __name__ == "__main__":
    main()
