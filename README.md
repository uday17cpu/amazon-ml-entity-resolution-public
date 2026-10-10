# Amazon ML Challenge — Business Entity Resolution

A deterministic, local Python pipeline for matching each Source 1 business
record to its corresponding records in Source 2 and Source 3. Candidate
generation uses a disk-backed SQLite index and multiple name/address blocking
keys; a pairwise classifier scores only those candidates.

The project is designed to use only the challenge-provided TSV files. It does
not call external APIs, use external lookup data, or compare every Source 1
record against the full target cross-product.

## Repository contents

- [`code/business_entity_resolution/`](code/business_entity_resolution/README.md)
  contains installation, training, inference, validation, and packaging
  instructions.

Challenge datasets, generated predictions, SQLite indexes, and trained model
artifacts are intentionally not included. Obtain the authorized challenge
resources separately and place the `student_resource/` directory at the
repository root before running the full workflow.

## Quick start

Requirements: Python 3.9+ and the official challenge resource files, including
the validator.

```bash
cd code/business_entity_resolution
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export DATA_DIR="$PWD/../../student_resource"
```

Follow the full commands in the
[pipeline README](code/business_entity_resolution/README.md) to:

1. Build a Source 2/3 SQLite index from training data.
2. Train the pair model and select a threshold using a held-out Source 1
   validation split.
3. Build the test index and generate both submission TSVs.
4. Run the official validator and package the submission.

The full pipeline cannot run until the challenge resources are supplied. Do
not treat this repository alone as containing the challenge data or a trained
production model.

## Submission files

Inference writes the following tab-separated files under `output/`:

- `matching_results.tsv`: `source1_entity_id`, `matched_entity_ids`
- `candidate_pairs.tsv`: `source1_entity_id`, `candidate_entity_ids`

Each Source 1 entity has one row in each file. Candidate and match lists contain
only retrieved Source 2/3 IDs; an empty list is represented by an empty field.
See the pipeline README for the exact validator and packaging commands.

## Reproducibility and evaluation

Training and validation splits are made at the Source 1 entity level so labels
for one entity are not divided between partitions. The matching threshold is
chosen on validation data using entity-level macro F0.5, including entities
with no true matches. Inference does not read ground-truth labels.

Results depend on the challenge data, training options, and generated model.
No leaderboard score or perfect result is guaranteed; evaluate the complete
pipeline with the official validator and authorized validation labels.
