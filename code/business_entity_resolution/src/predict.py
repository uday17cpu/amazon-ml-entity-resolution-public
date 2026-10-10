"""Generate matching_results.tsv and candidate_pairs.tsv from an index and model."""

from __future__ import annotations

import argparse
import csv
import multiprocessing
import math
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from common import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    baseline_score,
    find_candidates,
    iter_tsv,
    make_record,
    open_index,
    source_from_id,
    write_tsv_row,
)


def load_model(path: Path | None):
    if path is None:
        return None, None, None
    import joblib

    bundle = joblib.load(path)
    feature_names = bundle.get("feature_names") if isinstance(bundle, dict) else None
    if not isinstance(feature_names, (tuple, list)) or tuple(feature_names) != FEATURE_NAMES:
        raise ValueError(f"{path}: model feature schema does not match the current inference pipeline.")
    if bundle.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError(f"{path}: model feature schema version does not match the current inference pipeline.")
    model = bundle.get("model")
    if not hasattr(model, "predict_proba") or getattr(model, "n_features_in_", None) != len(
        FEATURE_NAMES
    ):
        raise ValueError(f"{path}: model is not a compatible probability classifier.")
    threshold = bundle.get("threshold")
    if not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError(f"{path}: model bundle has an invalid decision threshold.")
    if 1 not in getattr(model, "classes_", ()):
        raise ValueError(f"{path}: model bundle does not contain the positive class.")
    max_block_size = bundle.get("max_block_size")
    max_candidates = bundle.get("max_candidates")
    if (
        not isinstance(max_block_size, int)
        or max_block_size < 1
        or not isinstance(max_candidates, int)
        or max_candidates < 1
    ):
        raise ValueError(f"{path}: model bundle is missing valid candidate-generation settings.")
    return model, float(threshold), (max_block_size, max_candidates)


def process_chunk(
    source1_path: Path,
    index_path: Path,
    model_path: Path | None,
    threshold: float,
    max_block_size: int,
    max_candidates: int,
    matching_path: Path,
    candidate_path: Path,
) -> tuple[int, int, int, int]:
    model, _, _ = load_model(model_path)
    connection = open_index(index_path, max_block_size)
    total = matched_sources = total_candidates = total_matches = 0
    with matching_path.open("w", encoding="utf-8", newline="") as matching_file, candidate_path.open(
        "w", encoding="utf-8", newline=""
    ) as candidate_file:
        matching_writer = csv.writer(matching_file, delimiter="\t", lineterminator="\n")
        candidate_writer = csv.writer(candidate_file, delimiter="\t", lineterminator="\n")
        matching_writer.writerow(["source1_entity_id", "matched_entity_ids"])
        candidate_writer.writerow(["source1_entity_id", "candidate_entity_ids"])
        try:
            for row in iter_tsv(
                source1_path, ("entity_id", "business_name", "business_address", "country")
            ):
                query = make_record(row)
                if query.source != "S1":
                    raise ValueError(f"{source1_path}: expected S1- ID, got {query.entity_id!r}.")
                candidates = find_candidates(connection, query, max_block_size, max_candidates)
                candidate_ids = [candidate.entity_id for candidate, _, _ in candidates]
                if len(candidate_ids) != len(set(candidate_ids)) or any(
                    source_from_id(entity_id) not in {"S2", "S3"} for entity_id in candidate_ids
                ):
                    raise ValueError(f"Invalid candidate IDs generated for {query.entity_id!r}.")
                if model is None:
                    scores = [baseline_score(vector) for _, _, vector in candidates]
                else:
                    scores = (
                        model.predict_proba([vector for _, _, vector in candidates])[:, 1].tolist()
                        if candidates
                        else []
                    )
                matched_ids = [
                    candidate.entity_id
                    for (candidate, _, _), score in zip(candidates, scores)
                    if score >= threshold
                ]
                if not set(matched_ids).issubset(candidate_ids):
                    raise ValueError(
                        f"Predicted IDs are outside the candidate pool for {query.entity_id!r}."
                    )
                write_tsv_row(candidate_writer, query.entity_id, candidate_ids)
                write_tsv_row(matching_writer, query.entity_id, matched_ids)
                total += 1
                total_candidates += len(candidate_ids)
                total_matches += len(matched_ids)
                matched_sources += int(bool(matched_ids))
        finally:
            connection.close()
    return total, matched_sources, total_candidates, total_matches


def process_chunk_task(
    arguments: tuple[Path, Path, Path | None, float, int, int, Path, Path]
) -> tuple[int, int, int, int]:
    return process_chunk(*arguments)


def predict_parallel(
    source1_path: Path,
    index_path: Path,
    model_path: Path | None,
    threshold: float,
    max_block_size: int,
    max_candidates: int,
    output_dir: Path,
    matching_path: Path,
    candidate_path: Path,
    workers: int,
    chunk_size: int,
) -> tuple[int, int, int, int]:
    required_columns = ("entity_id", "business_name", "business_address", "country")
    chunks: list[tuple[Path, Path, Path]] = []
    seen_source1_ids: set[str] = set()
    with tempfile.TemporaryDirectory(prefix="prediction-", dir=output_dir) as temporary_directory:
        temporary_root = Path(temporary_directory)
        chunk_file = None
        chunk_writer = None
        chunk_number = 0
        rows_in_chunk = 0
        for row in iter_tsv(source1_path, required_columns):
            query = make_record(row)
            if query.source != "S1":
                raise ValueError(f"{source1_path}: expected S1- ID, got {query.entity_id!r}.")
            if query.entity_id in seen_source1_ids:
                raise ValueError(f"{source1_path}: duplicate Source-1 ID {query.entity_id!r}.")
            seen_source1_ids.add(query.entity_id)
            if chunk_writer is None:
                chunk_input = temporary_root / f"source1-{chunk_number:05d}.tsv"
                chunk_file = chunk_input.open("w", encoding="utf-8", newline="")
                chunk_writer = csv.writer(chunk_file, delimiter="\t", lineterminator="\n")
                chunk_writer.writerow(required_columns)
                rows_in_chunk = 0
            chunk_writer.writerow([row[column] for column in required_columns])
            rows_in_chunk += 1
            if rows_in_chunk == chunk_size:
                chunk_file.close()
                chunks.append(
                    (
                        chunk_input,
                        temporary_root / f"matching-{chunk_number:05d}.tsv",
                        temporary_root / f"candidates-{chunk_number:05d}.tsv",
                    )
                )
                chunk_number += 1
                chunk_file = chunk_writer = None
        if chunk_file is not None:
            chunk_file.close()
            chunks.append(
                (
                    chunk_input,
                    temporary_root / f"matching-{chunk_number:05d}.tsv",
                    temporary_root / f"candidates-{chunk_number:05d}.tsv",
                )
            )

        task_args = [
            (
                chunk_input,
                index_path,
                model_path,
                threshold,
                max_block_size,
                max_candidates,
                chunk_matching,
                chunk_candidates,
            )
            for chunk_input, chunk_matching, chunk_candidates in chunks
        ]
        context = multiprocessing.get_context("spawn")
        totals = [0, 0, 0, 0]
        completed_rows = 0
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
            futures = [executor.submit(process_chunk_task, arguments) for arguments in task_args]
            for future in as_completed(futures):
                result = future.result()
                totals = [total + value for total, value in zip(totals, result)]
                completed_rows += result[0]
                if completed_rows and completed_rows % 100_000 < result[0]:
                    print(f"  predicted {completed_rows:,} S1 records", flush=True)

        with matching_path.open("w", encoding="utf-8", newline="") as matching_file, candidate_path.open(
            "w", encoding="utf-8", newline=""
        ) as candidate_file:
            matching_file.write("source1_entity_id\tmatched_entity_ids\n")
            candidate_file.write("source1_entity_id\tcandidate_entity_ids\n")
            for _, chunk_matching, chunk_candidates in chunks:
                with chunk_matching.open("r", encoding="utf-8", newline="") as part:
                    next(part, None)
                    for line in part:
                        matching_file.write(line)
                with chunk_candidates.open("r", encoding="utf-8", newline="") as part:
                    next(part, None)
                    for line in part:
                        candidate_file.write(line)
        return tuple(totals)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source1", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, help="joblib bundle produced by train_model.py")
    parser.add_argument("--threshold", type=float, help="Overrides threshold stored in the model bundle.")
    parser.add_argument("--max-block-size", type=int)
    parser.add_argument("--max-candidates", type=int)
    parser.add_argument("--baseline-threshold", type=float, default=0.91)
    parser.add_argument("--workers", type=int, default=1, help="Independent Source-1 workers.")
    parser.add_argument("--chunk-size", type=int, default=10_000, help="Source-1 rows per worker chunk.")
    args = parser.parse_args()

    model, model_threshold, model_candidate_config = load_model(args.model)
    threshold = (
        args.threshold
        if args.threshold is not None
        else (model_threshold if model_threshold is not None else args.baseline_threshold)
    )
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise SystemExit("The decision threshold must be in [0, 1].")
    if (args.max_block_size is not None and args.max_block_size < 1) or (
        args.max_candidates is not None and args.max_candidates < 1
    ):
        raise SystemExit("--max-block-size and --max-candidates must be positive.")
    if args.workers < 1 or args.chunk_size < 1:
        raise SystemExit("--workers and --chunk-size must be positive.")
    if model_candidate_config is None:
        max_block_size = args.max_block_size if args.max_block_size is not None else 500
        max_candidates = args.max_candidates if args.max_candidates is not None else 75
    else:
        max_block_size, max_candidates = model_candidate_config
        if args.max_block_size not in (None, max_block_size) or args.max_candidates not in (
            None,
            max_candidates,
        ):
            raise SystemExit(
                "Candidate-generation settings must match the trained model bundle "
                f"({max_block_size=}, {max_candidates=})."
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    matching_path = args.output_dir / "matching_results.tsv"
    candidate_path = args.output_dir / "candidate_pairs.tsv"

    if args.workers > 1:
        totals = predict_parallel(
            args.source1,
            args.index,
            args.model,
            threshold,
            max_block_size,
            max_candidates,
            args.output_dir,
            matching_path,
            candidate_path,
            args.workers,
            args.chunk_size,
        )
        total, matched_sources, total_candidates, total_matches = totals
        print(f"Wrote {matching_path} and {candidate_path}")
        print(
            f"S1 rows: {total:,}; candidate pairs: {total_candidates:,}; "
            f"predicted links: {total_matches:,}"
        )
        print(f"S1 rows with >=1 predicted match: {matched_sources:,}; threshold: {threshold:.3f}")
        return

    connection = open_index(args.index, max_block_size)
    with open(matching_path, "w", encoding="utf-8", newline="") as matching_file, open(
        candidate_path, "w", encoding="utf-8", newline=""
    ) as candidate_file:
        matching_writer = csv.writer(matching_file, delimiter="\t", lineterminator="\n")
        candidate_writer = csv.writer(candidate_file, delimiter="\t", lineterminator="\n")
        matching_writer.writerow(["source1_entity_id", "matched_entity_ids"])
        candidate_writer.writerow(["source1_entity_id", "candidate_entity_ids"])
        total = matched_sources = total_candidates = total_matches = 0
        seen_source1_ids: set[str] = set()
        try:
            for row in iter_tsv(
                args.source1, ("entity_id", "business_name", "business_address", "country")
            ):
                query = make_record(row)
                if query.source != "S1":
                    raise ValueError(f"{args.source1}: expected S1- ID, got {query.entity_id!r}.")
                if query.entity_id in seen_source1_ids:
                    raise ValueError(f"{args.source1}: duplicate Source-1 ID {query.entity_id!r}.")
                seen_source1_ids.add(query.entity_id)
                candidates = find_candidates(connection, query, max_block_size, max_candidates)
                candidate_ids = [candidate.entity_id for candidate, _, _ in candidates]
                if len(candidate_ids) != len(set(candidate_ids)) or any(
                    source_from_id(entity_id) not in {"S2", "S3"} for entity_id in candidate_ids
                ):
                    raise ValueError(f"Invalid candidate IDs generated for {query.entity_id!r}.")
                if model is None:
                    scores = [baseline_score(vector) for _, _, vector in candidates]
                else:
                    scores = (
                        model.predict_proba([vector for _, _, vector in candidates])[:, 1].tolist()
                        if candidates
                        else []
                    )
                matched_ids = [
                    candidate.entity_id
                    for (candidate, _, _), score in zip(candidates, scores)
                    if score >= threshold
                ]
                if not set(matched_ids).issubset(candidate_ids):
                    raise ValueError(f"Predicted IDs are outside the candidate pool for {query.entity_id!r}.")
                write_tsv_row(candidate_writer, query.entity_id, candidate_ids)
                write_tsv_row(matching_writer, query.entity_id, matched_ids)
                total += 1
                total_candidates += len(candidate_ids)
                total_matches += len(matched_ids)
                matched_sources += int(bool(matched_ids))
                if total % 100_000 == 0:
                    print(f"  predicted {total:,} S1 records", flush=True)
        finally:
            connection.close()

    print(f"Wrote {matching_path} and {candidate_path}")
    print(f"S1 rows: {total:,}; candidate pairs: {total_candidates:,}; predicted links: {total_matches:,}")
    print(f"S1 rows with >=1 predicted match: {matched_sources:,}; threshold: {threshold:.3f}")


if __name__ == "__main__":
    main()
