"""Tests for long-form assessment loading."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from learning_assessment.loader import LONG_COLUMNS, load


class LoadLongFormTests(unittest.TestCase):
    def test_load_returns_one_row_per_student_objective_assessment(self):
        with tempfile.TemporaryDirectory() as directory:
            datapath = Path(directory)
            pd.DataFrame(
                {
                    "id": [1, 2],
                    "name": ["Ada", "Grace"],
                    "Newton": [2.0, None],
                    "FBD": [1.0, 2.0],
                }
            ).to_csv(datapath / "Quiz 2026-09-01.csv", index=False)
            pd.DataFrame(
                {
                    "id": [1, 2],
                    "name": ["Ada", "Grace"],
                    "Calculus": [2.0, 1.0],
                }
            ).to_csv(datapath / "Exam 2026-09-08.csv", index=False)

            data = load(datapath)

        self.assertEqual(list(data.columns), LONG_COLUMNS)
        self.assertEqual(len(data), 6)

        # A missing submission remains missing rather than being changed into
        # a score or confused with an objective absent from the assessment.
        missing_score = data.loc[
            (data["id"] == 2) & (data["Objective"] == "Newton"), "Score"
        ]
        self.assertTrue(missing_score.isna().all())

        # Newton was not assessed on the Exam, so no structural rows exist.
        structural_rows = data.loc[
            data["Assignment"].str.startswith("Exam")
            & (data["Objective"] == "Newton")
        ]
        self.assertTrue(structural_rows.empty)

    def test_load_rejects_missing_identifier_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            datapath = Path(directory)
            pd.DataFrame({"id": [1], "Newton": [2.0]}).to_csv(
                datapath / "Quiz 2026-09-01.csv", index=False
            )

            with self.assertRaisesRegex(ValueError, "missing columns: name"):
                load(datapath)

    def test_load_rejects_empty_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "No assessment CSV files"):
                load(directory)


if __name__ == "__main__":
    unittest.main()
