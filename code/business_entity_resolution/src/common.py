"""Shared, dependency-light utilities for business entity resolution.

The functions in this module deliberately use only deterministic text processing.
They do not look up businesses, addresses, or any external data.
"""

from __future__ import annotations

import csv
import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from rapidfuzz.fuzz import ratio as rapid_ratio


TOKEN_RE = re.compile(r"[a-z0-9]+")
MULTISPACE_RE = re.compile(r"\s+")

# These are generic text expansions, not a country-specific address database.
REPLACEMENTS = {
    "&": " and ",
    "pvt ltd": "private limited",
    "pvt": "private",
    "ltd": "limited",
    "inc": "incorporated",
    "corp": "corporation",
    "co": "company",
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "blvd": "boulevard",
    "hwy": "highway",
    "apt": "apartment",
}
REPLACEMENT_RE = re.compile(
    r"\b(" + "|".join(re.escape(key) for key in REPLACEMENTS) + r")\b"
)
COMMON_TOKENS = {
    "and", "the", "for", "with", "company", "incorporated", "corporation",
    "limited", "private", "road", "street", "avenue", "building", "near",
}


def normalize_text(value: str | None) -> str:
    """Return an ASCII, lower-case, token-preserving representation of text."""
    value = value or ""
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    value = value.lower().replace("&", " and ")
    value = REPLACEMENT_RE.sub(lambda match: REPLACEMENTS[match.group(1)], value)
    return " ".join(TOKEN_RE.findall(value))


def tokens(value: str) -> tuple[str, ...]:
    return tuple(TOKEN_RE.findall(value))


def token_set(value: str) -> set[str]:
    return set(tokens(value))


def numeric_tokens(value: str) -> tuple[str, ...]:
    return tuple(token for token in tokens(value) if token.isdigit() and len(token) >= 2)


def country_key(value: str | None) -> str:
    """Use country only as a generic partition; supports unseen labels such as France."""
    return normalize_text(value).replace(" ", "_") or "unknown"


def blocking_keys(country: str, name: str, address: str) -> set[str]:
    """Generate several inexpensive, complementary blocking keys for one record."""
    c = country_key(country)
    name_tokens = [t for t in tokens(name) if len(t) >= 3 and t not in COMMON_TOKENS]
    address_tokens = [t for t in tokens(address) if len(t) >= 4 and t not in COMMON_TOKENS]
    keys: set[str] = set()

    if len(name) >= 4:
        keys.add(f"{c}|name_exact|{name}")
    if len(address) >= 8:
        keys.add(f"{c}|address_exact|{address}")
    # Individual name tokens are useful when legal suffixes or word order differ.
    for token in sorted(set(name_tokens))[:6]:
        keys.add(f"{c}|name_token|{token}")
    # Pair keys reduce large common-token blocks while tolerating word reordering.
    for i, left in enumerate(sorted(set(name_tokens))[:5]):
        for right in sorted(set(name_tokens))[i + 1 : 5]:
            keys.add(f"{c}|name_pair|{left}|{right}")
    for number in numeric_tokens(address):
        keys.add(f"{c}|address_number|{number}")
    if address_tokens:
        keys.add(f"{c}|address_prefix|{'|'.join(address_tokens[:2])}")
    return keys


def jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    left_set, right_set = set(left), set(right)
    if not left_set and not right_set:
        return 1.0
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def ratio(left: str, right: str) -> float:
    """Return a fast normalized edit-similarity score for two strings."""
    return rapid_ratio(left, right) / 100.0


FEATURE_NAMES = (
    "name_ratio",
    "name_token_jaccard",
    "address_ratio",
    "address_token_jaccard",
    "number_jaccard",
    "name_exact",
    "address_exact",
    "shared_block_count",
    "same_country",
)
FEATURE_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class Record:
    entity_id: str
    source: str
    country: str
    name: str
    address: str
    numbers: str


def make_record(row: dict[str, str]) -> Record:
    entity_id = row.get("entity_id")
    if not entity_id:
        raise ValueError("TSV row has an empty entity_id.")
    source = source_from_id(entity_id)
    name = normalize_text(row.get("business_name", ""))
    address = normalize_text(row.get("business_address", ""))
    return Record(
        entity_id=entity_id,
        source=source,
        country=row.get("country", ""),
        name=name,
        address=address,
        numbers=" ".join(numeric_tokens(address)),
    )


def source_from_id(entity_id: str) -> str:
    source, separator, suffix = entity_id.partition("-")
    if not separator or not suffix or source not in {"S1", "S2", "S3"}:
        raise ValueError(f"Invalid entity ID {entity_id!r}; expected an S1-, S2-, or S3- ID.")
    return source


def feature_vector(query: Record, target: Record, votes: int) -> list[float]:
    name_present = bool(query.name and target.name)
    address_present = bool(query.address and target.address)
    numbers_present = bool(query.numbers and target.numbers)
    country_present = bool(query.country.strip() and target.country.strip())
    return [
        ratio(query.name, target.name) if name_present else 0.0,
        jaccard(tokens(query.name), tokens(target.name)) if name_present else 0.0,
        ratio(query.address, target.address) if address_present else 0.0,
        jaccard(tokens(query.address), tokens(target.address)) if address_present else 0.0,
        jaccard(query.numbers.split(), target.numbers.split()) if numbers_present else 0.0,
        float(bool(query.name and query.name == target.name)),
        float(bool(query.address and query.address == target.address)),
        min(votes, 8) / 8.0,
        float(
            country_present
            and country_key(query.country) == country_key(target.country)
        ),
    ]


def baseline_score(features: list[float]) -> float:
    """Conservative score used until a trained model is supplied."""
    name, name_tokens, address, address_tokens, numbers, name_exact, address_exact, votes, _ = features
    score = (
        0.38 * name
        + 0.18 * name_tokens
        + 0.24 * address
        + 0.10 * address_tokens
        + 0.06 * numbers
        + 0.08 * name_exact
        + 0.05 * address_exact
        + 0.03 * votes
    )
    return min(score, 1.0)


def iter_tsv(
    path: str | Path, required_columns: Iterable[str] = ()
) -> Iterator[dict[str, str]]:
    required_columns = tuple(required_columns)
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError(f"{path}: TSV is empty or has no header.")
        if None in reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError(f"{path}: TSV header contains an empty or duplicate column.")
        missing = set(required_columns) - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"{path}: TSV is missing required columns: {', '.join(sorted(missing))}."
            )
        for line_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"{path}:{line_number}: row has more fields than the TSV header.")
            if any(row.get(column) is None for column in required_columns):
                raise ValueError(f"{path}:{line_number}: row has fewer fields than the TSV header.")
            yield row


def parse_id_list(value: str | None) -> set[str]:
    return {item for item in (value or "").split(",") if item}


def open_index(
    path: str | Path, max_block_size: int | None = None
) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    if max_block_size is not None:
        try:
            row = connection.execute(
                "SELECT value FROM index_config WHERE name = 'max_block_size'"
            ).fetchone()
        except sqlite3.OperationalError as error:
            connection.close()
            raise ValueError(
                f"{path}: index lacks the oversized-block lookup; rebuild it with build_index.py."
            ) from error
        if row is None or int(row["value"]) != max_block_size:
            connection.close()
            indexed_limit = row["value"] if row is not None else "unknown"
            raise ValueError(
                f"{path}: index max_block_size is {indexed_limit}, but {max_block_size} was requested. "
                "Rebuild the index with matching settings."
            )
    return connection


def find_candidates(
    connection: sqlite3.Connection,
    query: Record,
    max_block_size: int,
    max_candidates: int,
) -> list[tuple[Record, int, list[float]]]:
    """Return final candidates ranked by an inexpensive pre-score.

    Blocks larger than max_block_size are ignored to prevent a common word from
    causing an unbounded candidate set. The returned list is exactly the set that
    later receives a model score, so it is suitable for candidate_pairs.tsv.
    """
    if query.source != "S1":
        raise ValueError(f"Candidate queries must be Source 1 records, got {query.entity_id!r}.")
    if max_block_size < 1 or max_candidates < 1:
        raise ValueError("max_block_size and max_candidates must both be positive.")
    keys = sorted(blocking_keys(query.country, query.name, query.address))
    if not keys:
        return []

    key_values = ", ".join("(?)" for _ in keys)
    sql = f"""
        WITH query_keys(block_key) AS (VALUES {key_values}),
        eligible_keys AS (
            SELECT q.block_key
            FROM query_keys AS q
            WHERE NOT EXISTS (
                SELECT 1
                FROM oversized_blocks AS o
                WHERE o.block_key = q.block_key
            )
        ),
        candidate_votes AS (
            SELECT b.entity_id, COUNT(*) AS votes
            FROM blocks AS b
            JOIN eligible_keys AS k ON k.block_key = b.block_key
            GROUP BY b.entity_id
        )
        SELECT r.entity_id, r.source, r.country, r.name, r.address, r.numbers,
               cv.votes
        FROM candidate_votes AS cv
        JOIN records AS r ON r.entity_id = cv.entity_id
        WHERE r.source IN ('S2', 'S3')
    """
    rows = connection.execute(sql, keys).fetchall()

    ranked: list[tuple[Record, int, list[float]]] = []
    for row in rows:
        if row["source"] not in {"S2", "S3"} or source_from_id(row["entity_id"]) != row["source"]:
            raise ValueError(f"Invalid target record in blocking index: {row['entity_id']!r}.")
        target = Record(
            entity_id=row["entity_id"],
            source=row["source"],
            country=row["country"],
            name=row["name"],
            address=row["address"],
            numbers=row["numbers"],
        )
        votes = row["votes"]
        features = feature_vector(query, target, votes)
        ranked.append((target, votes, features))
    ranked.sort(key=lambda item: (baseline_score(item[2]), item[1], item[0].entity_id), reverse=True)
    return ranked[:max_candidates]


def write_tsv_row(writer: csv.writer, source1_id: str, ids: Iterable[str]) -> None:
    writer.writerow([source1_id, ",".join(sorted(set(ids)))])


def stable_fraction(entity_id: str) -> float:
    # Deterministic sampling without relying on Python's randomized hash function.
    import hashlib

    digest = hashlib.blake2b(entity_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64
