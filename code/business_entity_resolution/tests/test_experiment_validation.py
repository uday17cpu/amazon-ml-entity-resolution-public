import sqlite3
import tempfile
import unittest
from pathlib import Path

from common import blocking_keys, make_record
from experiment_validation import entity_f05, missed_link_cause


class ExperimentValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.connection = sqlite3.connect(Path(self.temp_dir.name) / "targets.sqlite")
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(
            """
            CREATE TABLE records (
                entity_id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                country TEXT NOT NULL,
                name TEXT NOT NULL,
                address TEXT NOT NULL,
                numbers TEXT NOT NULL
            );
            CREATE TABLE blocks (block_key TEXT NOT NULL, entity_id TEXT NOT NULL);
            CREATE TABLE oversized_blocks (block_key TEXT PRIMARY KEY);
            """
        )
        self.query = make_record(
            {
                "entity_id": "S1-query",
                "business_name": "North Harbor Books",
                "business_address": "12 Lake Road",
                "country": "US",
            }
        )
        self.target = make_record(
            {
                "entity_id": "S2-target",
                "business_name": "North Harbor Books",
                "business_address": "12 Lake Rd",
                "country": "US",
            }
        )
        self._insert_record(self.target)
        self.connection.commit()

    def tearDown(self):
        self.connection.close()
        self.temp_dir.cleanup()

    def _insert_record(self, record):
        self.connection.execute(
            "INSERT INTO records VALUES (?, ?, ?, ?, ?, ?)",
            (
                record.entity_id,
                record.source,
                record.country,
                record.name,
                record.address,
                record.numbers,
            ),
        )
        self.connection.executemany(
            "INSERT INTO blocks VALUES (?, ?)",
            [(key, record.entity_id) for key in blocking_keys(
                record.country, record.name, record.address
            )],
        )

    def test_entity_f05_includes_correct_singletons_and_penalizes_false_matches(self):
        self.assertEqual(entity_f05(set(), set()), 1.0)
        self.assertEqual(entity_f05({"S2-false"}, set()), 0.0)
        self.assertEqual(entity_f05({"S2-true"}, {"S2-true"}), 1.0)
        self.assertEqual(entity_f05({"S2-predicted"}, {"S3-actual"}), 0.0)

    def test_missed_link_diagnoses_country_mismatch(self):
        target = make_record(
            {
                "entity_id": "S3-other-country",
                "business_name": "North Harbor Books",
                "business_address": "12 Lake Road",
                "country": "CA",
            }
        )
        self._insert_record(target)
        self.assertEqual(
            missed_link_cause(
                self.connection,
                self.query,
                target.entity_id,
                blocking_keys(self.query.country, self.query.name, self.query.address),
            ),
            "country_mismatch",
        )

    def test_missed_link_diagnoses_no_shared_block(self):
        target = make_record(
            {
                "entity_id": "S3-unshared",
                "business_name": "Cedar Valley Grocers",
                "business_address": "900 Ridge Avenue",
                "country": "US",
            }
        )
        self._insert_record(target)
        self.assertEqual(
            missed_link_cause(
                self.connection,
                self.query,
                target.entity_id,
                blocking_keys(self.query.country, self.query.name, self.query.address),
            ),
            "no_shared_block",
        )

    def test_missed_link_diagnoses_oversized_only_blocks(self):
        shared = blocking_keys(self.query.country, self.query.name, self.query.address) & (
            blocking_keys(self.target.country, self.target.name, self.target.address)
        )
        self.connection.executemany(
            "INSERT INTO oversized_blocks VALUES (?)", [(key,) for key in shared]
        )
        self.assertEqual(
            missed_link_cause(
                self.connection,
                self.query,
                self.target.entity_id,
                blocking_keys(self.query.country, self.query.name, self.query.address),
            ),
            "oversized_block_only",
        )


if __name__ == "__main__":
    unittest.main()
