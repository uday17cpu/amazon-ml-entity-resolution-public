"""Validate the completed package layout and create the final submission zip."""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path


REQUIRED_FILES = (
    "output/matching_results.tsv",
    "output/candidate_pairs.tsv",
    "code/business_entity_resolution/README.md",
    "code/business_entity_resolution/requirements.txt",
    "Documentation_template.md",
)
SOURCE_DIRECTORY = Path("code/business_entity_resolution/src")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-root", type=Path, required=True)
    parser.add_argument("--team-name", required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--zip-out", type=Path, required=True)
    args = parser.parse_args()

    root = args.submission_root.resolve()
    missing = [name for name in REQUIRED_FILES if not (root / name).is_file()]
    if missing:
        raise SystemExit("Missing required files:\n- " + "\n- ".join(missing))

    validator = args.test_dir.parent.parent / "utils" / "validate_submission.py"
    if not validator.is_file():
        raise SystemExit(f"Could not find validator at {validator}")
    source_directory = root / SOURCE_DIRECTORY
    if not source_directory.is_dir():
        raise SystemExit(f"Missing source directory: {source_directory}")
    source_files = sorted(path for path in source_directory.glob("*.py") if path.is_file())
    if not source_files:
        raise SystemExit(f"No Python source files found in {source_directory}")
    command = [
        sys.executable,
        str(validator),
        "--matching",
        str(root / "output/matching_results.tsv"),
        "--candidate",
        str(root / "output/candidate_pairs.tsv"),
        "--test-dir",
        str(args.test_dir),
    ]
    result = subprocess.run(command, check=False)
    if result.returncode:
        raise SystemExit("Output validation failed; zip was not created.")

    args.zip_out.parent.mkdir(parents=True, exist_ok=True)
    package_files = [root / name for name in REQUIRED_FILES]
    package_files.extend(source_files)
    with zipfile.ZipFile(args.zip_out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(set(package_files)):
            archive.write(path, path.relative_to(root))
    print(f"Created {args.zip_out}")


if __name__ == "__main__":
    main()
