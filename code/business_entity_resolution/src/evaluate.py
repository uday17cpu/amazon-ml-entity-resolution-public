"""Evaluate a matching_results.tsv file with the challenge's macro F0.5 metric."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import iter_tsv, parse_id_list, source_from_id


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--matching", type=Path, required=True)
    args = parser.parse_args()

    predictions: dict[str, set[str]] = {}
    for row in iter_tsv(args.matching, ("source1_entity_id", "matched_entity_ids")):
        entity_id = row["source1_entity_id"]
        if not entity_id or source_from_id(entity_id) != "S1":
            raise ValueError(f"{args.matching}: invalid Source-1 ID {entity_id!r}.")
        if entity_id in predictions:
            raise ValueError(f"{args.matching}: duplicate Source-1 row {entity_id!r}.")
        matched_ids = parse_id_list(row.get("matched_entity_ids"))
        invalid_ids = sorted(
            target for target in matched_ids if source_from_id(target) not in {"S2", "S3"}
        )
        if invalid_ids:
            raise ValueError(
                f"{args.matching}: {entity_id!r} has invalid matched IDs: "
                + ", ".join(invalid_ids[:5])
            )
        predictions[entity_id] = matched_ids
    if not predictions:
        raise SystemExit("No prediction rows found.")
    total = 0.0
    count = 0
    actual_ids: set[str] = set()
    for row in iter_tsv(args.ground_truth, ("source1_entity_id", "matched_entity_ids")):
        entity_id = row["source1_entity_id"]
        if not entity_id or source_from_id(entity_id) != "S1":
            raise ValueError(f"{args.ground_truth}: invalid Source-1 ID {entity_id!r}.")
        if entity_id in actual_ids:
            raise ValueError(f"{args.ground_truth}: duplicate Source-1 row {entity_id!r}.")
        actual_ids.add(entity_id)
        if entity_id not in predictions:
            raise ValueError(f"{args.matching}: missing prediction row for {entity_id!r}.")
        total += entity_f05(predictions[entity_id], parse_id_list(row.get("matched_entity_ids")))
        count += 1
    extra_ids = predictions.keys() - actual_ids
    if extra_ids:
        raise ValueError(
            f"{args.matching}: contains {len(extra_ids):,} Source-1 IDs absent from ground truth."
        )
    print(f"Scored Source-1 rows: {count:,}")
    print(f"Macro F0.5: {total / count:.6f}")


if __name__ == "__main__":
    main()
