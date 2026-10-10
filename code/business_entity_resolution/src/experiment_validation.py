"""Compare blocking caps on a fresh Source-1-level training validation split.

This experiment uses training labels only and never writes production artifacts.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from common import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    blocking_keys,
    country_key,
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


def quantile(sorted_values: list[int], fraction: float) -> int:
    return sorted_values[min(int(np.ceil(fraction * len(sorted_values))) - 1, len(sorted_values) - 1)]


def missed_link_cause(connection, query, target_id: str, query_keys: set[str]) -> str:
    row = connection.execute(
        "SELECT source, country, name, address FROM records WHERE entity_id = ?",
        (target_id,),
    ).fetchone()
    if row is None:
        return "target_not_indexed"

    target_keys = blocking_keys(row["country"], row["name"], row["address"])
    shared_keys = query_keys & target_keys
    if not shared_keys:
        if country_key(query.country) != country_key(row["country"]):
            return "country_mismatch"
        return "no_shared_block"

    placeholders = ",".join("?" for _ in shared_keys)
    oversized = {
        entry[0]
        for entry in connection.execute(
            f"SELECT block_key FROM oversized_blocks WHERE block_key IN ({placeholders})",
            tuple(sorted(shared_keys)),
        )
    }
    if shared_keys <= oversized:
        return "oversized_block_only"
    return "candidate_rank_or_cap"


def run_cap(
    cap: int,
    source1_path: Path,
    ground_truth_path: Path,
    index_path: Path,
    sample_rate: float,
    validation_fraction: float,
    split_seed: str,
    max_block_size: int,
    negative_per_source: int,
    model_out: Path | None,
) -> None:
    truth: dict[str, set[str]] = {}
    for row in iter_tsv(ground_truth_path, ("source1_entity_id", "matched_entity_ids")):
        entity_id = row["source1_entity_id"]
        if not entity_id or source_from_id(entity_id) != "S1":
            raise ValueError(f"{ground_truth_path}: invalid Source-1 ID {entity_id!r}.")
        if entity_id in truth:
            raise ValueError(f"{ground_truth_path}: duplicate Source-1 row {entity_id!r}.")
        if stable_fraction(entity_id) < sample_rate:
            targets = parse_id_list(row.get("matched_entity_ids"))
            if any(source_from_id(target) not in {"S2", "S3"} for target in targets):
                raise ValueError(f"{ground_truth_path}: invalid target ID for {entity_id!r}.")
            truth[entity_id] = targets

    selected_ids = sorted(truth)
    validation_ids = {
        entity_id
        for entity_id in selected_ids
        if stable_fraction(f"{entity_id}-{split_seed}") < validation_fraction
        and stable_fraction(f"{entity_id}-validation") >= 0.20
    }
    if not validation_ids or len(validation_ids) == len(selected_ids):
        raise ValueError("Fresh validation split is empty or consumes all sampled entities.")
    training_ids = set(selected_ids) - validation_ids

    features: list[list[float]] = []
    labels: list[int] = []
    validation_examples: list[tuple[set[str], list[tuple[str, list[float]]]]] = []
    validation_candidate_counts: list[int] = []
    total_links = Counter()
    retrieved_links = Counter()
    missed_causes = Counter()
    processed: set[str] = set()

    connection = open_index(index_path, max_block_size)
    try:
        target_count = connection.execute(
            "SELECT COUNT(*) FROM records WHERE source IN ('S2', 'S3')"
        ).fetchone()[0]
        for row in iter_tsv(
            source1_path, ("entity_id", "business_name", "business_address", "country")
        ):
            entity_id = row["entity_id"]
            if entity_id not in truth:
                continue
            if entity_id in processed:
                raise ValueError(f"{source1_path}: duplicate Source-1 row {entity_id!r}.")
            processed.add(entity_id)
            query = make_record(row)
            candidates = find_candidates(connection, query, max_block_size, cap)
            candidate_ids = {candidate.entity_id for candidate, _, _ in candidates}
            actual = truth[entity_id]
            candidate_rows = [(candidate.entity_id, vector) for candidate, _, vector in candidates]
            if entity_id in validation_ids:
                validation_candidate_counts.append(len(candidate_ids))
                query_keys = blocking_keys(query.country, query.name, query.address)
                for target_id in actual:
                    target_source = source_from_id(target_id)
                    total_links[target_source] += 1
                    retrieved_links[target_source] += target_id in candidate_ids
                    if target_id not in candidate_ids:
                        missed_causes[
                            missed_link_cause(connection, query, target_id, query_keys)
                        ] += 1
                validation_examples.append((actual, candidate_rows))
            else:
                negatives_added = 0
                for target_id, vector in candidate_rows:
                    is_positive = target_id in actual
                    if not is_positive and negatives_added >= negative_per_source:
                        continue
                    features.append(vector)
                    labels.append(int(is_positive))
                    if not is_positive:
                        negatives_added += 1
    finally:
        connection.close()

    if processed != set(selected_ids):
        raise ValueError(
            f"{source1_path}: missing {len(set(selected_ids) - processed):,} sampled S1 rows."
        )
    if not features or len(set(labels)) < 2:
        raise ValueError("Experiment training split does not contain both classes.")

    train_x = np.asarray(features, dtype=np.float32)
    train_y = np.asarray(labels, dtype=np.int8)
    model = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=250,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=42,
    )
    model.fit(train_x, train_y)

    validation_features = [
        vector
        for _, candidates in validation_examples
        for _, vector in candidates
    ]
    probabilities = (
        model.predict_proba(np.asarray(validation_features, dtype=np.float32))[:, 1]
        if validation_features
        else np.asarray([], dtype=np.float32)
    )
    thresholds = np.linspace(0.05, 0.99, 95)
    best_threshold = 0.90
    best_score = -1.0
    for threshold in thresholds:
        probability_offset = 0
        scores = []
        for actual, candidates in validation_examples:
            end = probability_offset + len(candidates)
            predicted = {
                candidate_id
                for (candidate_id, _), probability in zip(
                    candidates, probabilities[probability_offset:end]
                )
                if probability >= threshold
            }
            probability_offset = end
            scores.append(entity_f05(predicted, actual))
        score = float(np.mean(scores)) if scores else 0.0
        if score > best_score or (score == best_score and threshold > best_threshold):
            best_threshold, best_score = float(threshold), score

    sorted_counts = sorted(validation_candidate_counts)
    total_candidates = sum(sorted_counts)
    possible_pairs = len(sorted_counts) * target_count
    print(f"\nCAP {cap}")
    print(
        f"Split seed={split_seed}; negatives_per_source={negative_per_source}; "
        f"sampled={len(selected_ids):,}; "
        f"train_S1={len(training_ids):,}; validation_S1={len(validation_candidate_counts):,}"
    )
    print(
        f"Candidates total={total_candidates:,}; mean={total_candidates / len(sorted_counts):.2f}; "
        f"median={quantile(sorted_counts, .50)}; p90={quantile(sorted_counts, .90)}; "
        f"p95={quantile(sorted_counts, .95)}; p99={quantile(sorted_counts, .99)}; "
        f"max={max(sorted_counts)}"
    )
    print(
        f"Reduction={100 * (1 - total_candidates / possible_pairs):.6f}% "
        f"vs {possible_pairs:,} validation cross-product pairs"
    )
    for source in ("S2", "S3"):
        recall = retrieved_links[source] / max(total_links[source], 1)
        print(
            f"{source} true_links={total_links[source]:,}; "
            f"retrieved={retrieved_links[source]:,}; recall={recall:.6f}"
        )
    total_true = sum(total_links.values())
    total_retrieved = sum(retrieved_links.values())
    print(
        f"Overall candidate recall={total_retrieved / max(total_true, 1):.6f} "
        f"({total_retrieved:,}/{total_true:,}); missed_link_causes={dict(missed_causes)}"
    )
    print(
        f"Validation macro F0.5={best_score:.6f}; selected threshold={best_threshold:.4f}; "
        f"feature_count={len(FEATURE_NAMES)}"
    )
    if model_out is not None:
        model_out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": model,
                "feature_names": FEATURE_NAMES,
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "threshold": best_threshold,
                "sample_rate": sample_rate,
                "validation_source1_count": len(validation_examples),
                "validation_macro_f0_5": best_score,
                "max_block_size": max_block_size,
                "max_candidates": cap,
                "experiment_split_seed": split_seed,
                "negative_per_source": negative_per_source,
            },
            model_out,
        )
        print(f"Saved isolated experiment model to {model_out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source1", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--sample-rate", type=float, default=0.01)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--split-seed", default="fresh-validation-v1")
    parser.add_argument("--max-block-size", type=int, default=500)
    parser.add_argument("--negative-per-source", type=int, nargs="+", default=[4, 20, 75])
    parser.add_argument("--caps", type=int, nargs="+", default=[75, 100, 150])
    parser.add_argument("--model-out", type=Path)
    args = parser.parse_args()
    if not 0 < args.sample_rate <= 1:
        raise SystemExit("--sample-rate must be in (0, 1].")
    if not 0 < args.validation_fraction < 1:
        raise SystemExit("--validation-fraction must be in (0, 1).")
    if args.max_block_size < 1 or any(value < 0 for value in args.negative_per_source):
        raise SystemExit("--max-block-size must be positive and negative counts cannot be negative.")
    if not args.caps or any(cap < 1 for cap in args.caps):
        raise SystemExit("All --caps values must be positive.")
    if len(set(args.caps)) != len(args.caps) or len(set(args.negative_per_source)) != len(
        args.negative_per_source
    ):
        raise SystemExit("--caps and --negative-per-source values must be unique.")

    for cap in args.caps:
        for negative_per_source in args.negative_per_source:
            run_cap(
                cap,
                args.source1,
                args.ground_truth,
                args.index,
                args.sample_rate,
                args.validation_fraction,
                args.split_seed,
                args.max_block_size,
                negative_per_source,
                args.model_out
                if cap == args.caps[-1]
                and negative_per_source == args.negative_per_source[-1]
                else None,
            )


if __name__ == "__main__":
    main()
