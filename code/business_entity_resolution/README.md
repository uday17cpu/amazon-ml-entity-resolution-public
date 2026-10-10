# Business Entity Resolution

This is a reproducible, local-only solution for the Amazon ML Challenge
business entity-resolution task. It never calls an external API, database,
geocoder, or pre-trained language model.

## Approach

1. Normalize business names and addresses using deterministic generic rules.
2. Create a disk-backed SQLite index over Source 2 and Source 3 with multiple
   blocking keys: normalized full name, name tokens/pairs, address numbers, and
   address token prefixes. The index precomputes keys whose blocks exceed the
   configured size, avoiding repeated counts of common-token blocks at query
   time.
3. For each Source 1 entity, retrieve candidates from non-oversized blocks,
   combine blocks, and retain the top candidates by an inexpensive similarity
   score. This exact final set is written to `candidate_pairs.tsv`.
4. Score those candidate pairs with a `HistGradientBoostingClassifier` trained
   using hard negatives from the same candidate pipeline.
5. Retain every score above the threshold selected once using macro F0.5 on a
   held-out set of Source-1 entities. Validation includes entities with no
   candidates, including true singletons.

All input files are read as tab-separated files by their column names. Source
IDs must use the `S1-`, `S2-`, and `S3-` prefixes. Country is an open text field:
normalization does not map it through a fixed country list, so France and
unseen country strings remain valid. Every Source-1 input row is written to
both output files, including rows with no candidates.

## Installation

Use Python 3.9+ and a virtual environment:

```bash
cd code/business_entity_resolution
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set the path to the supplied challenge resource directory (when it is at the
repository root):

```bash
export DATA_DIR="$PWD/../../student_resource"
```

## 1. Build the train index

```bash
python src/build_index.py \
  --source2 "$DATA_DIR/dataset/train/train_source2.tsv" \
  --source3 "$DATA_DIR/dataset/train/train_source3.tsv" \
  --index artifacts/train_targets.sqlite
```

## 2. Train and select the threshold

Start with one percent of Source 1 labels, then increase `--sample-rate` if
runtime and disk capacity allow. The command splits selected Source-1 IDs
deterministically before creating training pairs. It generates positives from
retrieved ground-truth links and hard negatives from the highest-ranked
non-matching candidates. The complete validation candidate pool is scored with
the same blocking, features, classifier, and multi-match threshold logic used
at inference; a threshold is selected once and stored with the feature schema.
Ground truth is only read by this training command.

```bash
python src/train_model.py \
  --source1 "$DATA_DIR/dataset/train/train_source1.tsv" \
  --ground-truth "$DATA_DIR/dataset/train/train_ground_truth.tsv" \
  --index artifacts/train_targets.sqlite \
  --model-out artifacts/pair_model.joblib \
  --sample-rate 0.01
```

## 3. Build the test index and create outputs

```bash
python src/build_index.py \
  --source2 "$DATA_DIR/dataset/test/test_source2.tsv" \
  --source3 "$DATA_DIR/dataset/test/test_source3.tsv" \
  --index artifacts/test_targets.sqlite

python src/predict.py \
  --source1 "$DATA_DIR/dataset/test/test_source1.tsv" \
  --index artifacts/test_targets.sqlite \
  --model artifacts/pair_model.joblib \
  --output-dir ../../output \
  --workers 8
```

`predict.py` writes exactly `output/matching_results.tsv` and
`output/candidate_pairs.tsv`. The candidate file contains the exact candidate
pool scored by the model. Matching is not top-1 constrained: all candidates
above the stored threshold are retained. Inference reads no ground-truth file.

For full-size inference, `--workers` processes independent Source-1 chunks in
parallel using separate SQLite connections. Chunk outputs are merged in source
order, preserving deterministic output ordering; use `--workers 1` for a
single-process run.

## 4. Validate and package

```bash
python "$DATA_DIR/utils/validate_submission.py" \
  --matching ../../output/matching_results.tsv \
  --candidate ../../output/candidate_pairs.tsv \
  --test-dir "$DATA_DIR/dataset/test"

python src/package_submission.py \
  --submission-root ../.. \
  --team-name "your_team_name" \
  --test-dir "$DATA_DIR/dataset/test" \
  --zip-out ../../your_team_name_submission.zip
```

The final ZIP must include both output files, this source package, and the
completed `Documentation_template.md`. Replace the team placeholders before
uploading.

`package_submission.py` invokes the official `utils/validate_submission.py`
before creating the ZIP. It stops without packaging if the validator is not
available or reports invalid output. For a local metric check when ground truth
is legitimately available, `src/evaluate.py` requires a prediction for every
ground-truth Source-1 entity and computes entity-level macro F0.5, assigning
1.0 to an empty prediction for an empty truth set.
