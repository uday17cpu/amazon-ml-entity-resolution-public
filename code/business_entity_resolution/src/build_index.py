"""Create an on-disk SQLite blocking index over Source 2 and Source 3."""

from __future__ import annotations

import argparse
import os
import sqlite3
from pathlib import Path

from common import blocking_keys, iter_tsv, make_record


def create_schema(connection: sqlite3.Connection, max_block_size: int) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        PRAGMA temp_store=FILE;
        CREATE TABLE records (
            entity_id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            country TEXT NOT NULL,
            name TEXT NOT NULL,
            address TEXT NOT NULL,
            numbers TEXT NOT NULL
        );
        CREATE TABLE blocks (
            block_key TEXT NOT NULL,
            entity_id TEXT NOT NULL
        );
        CREATE TABLE oversized_blocks (
            block_key TEXT PRIMARY KEY
        );
        CREATE TABLE index_config (
            name TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """
    )
    connection.execute(
        "INSERT INTO index_config VALUES ('max_block_size', ?)",
        (str(max_block_size),),
    )


def insert_source(
    connection: sqlite3.Connection, path: Path, batch_size: int, expected_source: str
) -> int:
    record_rows: list[tuple[str, str, str, str, str, str]] = []
    block_rows: list[tuple[str, str]] = []
    count = 0
    for row in iter_tsv(path, ("entity_id", "business_name", "business_address", "country")):
        record = make_record(row)
        if record.source != expected_source:
            raise ValueError(
                f"{path}: expected {expected_source}- IDs, got {record.entity_id!r}."
            )
        record_rows.append(
            (record.entity_id, record.source, record.country, record.name, record.address, record.numbers)
        )
        block_rows.extend((key, record.entity_id) for key in blocking_keys(record.country, record.name, record.address))
        count += 1
        if len(record_rows) >= batch_size:
            connection.executemany("INSERT INTO records VALUES (?, ?, ?, ?, ?, ?)", record_rows)
            connection.executemany("INSERT INTO blocks VALUES (?, ?)", block_rows)
            connection.commit()
            record_rows.clear()
            block_rows.clear()
        if count % 250_000 == 0:
            print(f"  {path.name}: {count:,} records indexed", flush=True)
    if record_rows:
        connection.executemany("INSERT INTO records VALUES (?, ?, ?, ?, ?, ?)", record_rows)
        connection.executemany("INSERT INTO blocks VALUES (?, ?)", block_rows)
        connection.commit()
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source2", required=True, type=Path)
    parser.add_argument("--source3", required=True, type=Path)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--replace", action="store_true", help="Replace an existing index at --index.")
    parser.add_argument("--batch-size", type=int, default=20_000)
    parser.add_argument("--max-block-size", type=int, default=500)
    args = parser.parse_args()

    if args.batch_size < 1 or args.max_block_size < 1:
        raise SystemExit("--batch-size and --max-block-size must be positive.")
    if args.index.exists():
        if not args.replace:
            raise SystemExit(f"Index already exists: {args.index}. Use --replace to rebuild it.")
        args.index.unlink()
    args.index.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(args.index)
    create_schema(connection, args.max_block_size)
    try:
        total = insert_source(connection, args.source2, args.batch_size, "S2")
        total += insert_source(connection, args.source3, args.batch_size, "S3")
        print("Creating SQLite indexes and oversized-block lookup. This can take several minutes...", flush=True)
        connection.execute("CREATE INDEX idx_blocks_key ON blocks(block_key)")
        connection.execute("CREATE INDEX idx_blocks_entity ON blocks(entity_id)")
        connection.execute(
            """
            INSERT INTO oversized_blocks (block_key)
            SELECT block_key FROM blocks
            GROUP BY block_key
            HAVING COUNT(*) > ?
            """,
            (args.max_block_size,),
        )
        connection.commit()
    finally:
        connection.close()
    size_gb = os.path.getsize(args.index) / 1024**3
    print(f"Done: {total:,} target records indexed in {args.index} ({size_gb:.2f} GB).")


if __name__ == "__main__":
    main()
