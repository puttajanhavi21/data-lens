"""Run all tests with: python -m unittest discover -s tests -v"""

import unittest

import numpy as np
import pandas as pd

from analysis import clean_data, csv_for_download, load_csv, make_charts, profile_data


class AnalysisTests(unittest.TestCase):
    def test_valid_csv_normalizes_whitespace_and_missing_values(self):
        frame = load_csv(b" amount ,group\n1, A \ninf,   \n")
        self.assertEqual(frame.columns.tolist(), ["amount", "group"])
        self.assertEqual(frame.loc[0, "group"], "A")
        self.assertTrue(frame.iloc[1].isna().all())

    def test_invalid_inputs_are_explained(self):
        invalid_files = [b"", b"a,b\n", b"a,a\n1,2", b",b\n1,2", b"a,b\n1,2,3",
                         b"a,b\n1", b"a\n\xff", b"a\n\x00", b"a\n1e200"]
        for contents in invalid_files:
            with self.subTest(contents=contents), self.assertRaises(ValueError):
                load_csv(contents)

    def test_row_column_and_size_limits(self):
        for contents in [b"a\n" + b"1\n" * 20001,
                         (",".join(f"c{i}" for i in range(101)) + "\n" + ",".join(["1"] * 101)).encode(),
                         b"a" * (10 * 1024 * 1024 + 1)]:
            with self.assertRaises(ValueError):
                load_csv(contents)

    def test_profile_counts_and_statistics(self):
        frame = pd.DataFrame({"amount": [1, 2, 2, np.nan], "group": ["a", "b", "b", "c"]})
        result = profile_data(frame)
        self.assertEqual(result["missing_cells"], 1)
        self.assertEqual(result["duplicates"], 1)
        self.assertEqual(result["missing_percent"], 12.5)
        self.assertEqual(result["column_stats"][0]["median"], 2)
        self.assertIsNone(result["column_stats"][1]["mean"])

    def test_cleaning_copy_fill_drop_and_all_missing(self):
        frame = pd.DataFrame({"amount": [1., np.nan, 3., 3.], "group": ["a", None, "a", "a"],
                              "empty": [np.nan] * 4})
        original = frame.copy(deep=True)
        cleaned = clean_data(frame, "fill", True)
        self.assertEqual(len(cleaned), 3)
        self.assertEqual(cleaned.loc[1, "amount"], 2)
        self.assertEqual(cleaned.loc[1, "group"], "a")
        self.assertEqual(cleaned["empty"].tolist(), [0, 0, 0])
        pd.testing.assert_frame_equal(frame, original)
        with self.assertRaisesRegex(ValueError, "every row"):
            clean_data(frame, "drop")
        self.assertEqual(len(clean_data(frame.drop(columns="empty"), "drop")), 3)

    def test_export_neutralizes_text_formulas_without_changing_numbers(self):
        frame = pd.DataFrame({"=name": ["=SUM(A1)", "@formula"], "'=name": ["hello", "world"], "amount": [-1, 2]})
        exported = csv_for_download(frame).decode("utf-8-sig")
        self.assertIn("'=SUM(A1)", exported)
        self.assertIn("'@formula", exported)
        self.assertIn("hello,-1", exported)
        self.assertEqual(frame.iloc[0, 0], "=SUM(A1)")

    def test_charts_handle_constant_and_empty_numeric_columns(self):
        frame = pd.DataFrame({"constant": [1, 1], "empty": [np.nan, np.nan]})
        charts = make_charts(frame, "constant")
        self.assertEqual(len(charts), 2)
        self.assertTrue(all(chart["image"].startswith("iVBOR") for chart in charts))


if __name__ == "__main__":
    unittest.main()
