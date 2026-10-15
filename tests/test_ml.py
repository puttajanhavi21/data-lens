"""Run from the project folder: python -m unittest discover -s tests -v."""

import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from ml import train_model


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame(
            {
                "amount": np.arange(60, dtype=float),
                "region": ["north", "south", "west"] * 20,
                "purchased": ["yes", "no"] * 30,
                "price": np.arange(60, dtype=float) * 3 + 7,
            }
        )

    def test_classification_report_and_target_exclusion(self):
        result = train_model(self.data, "classification", "purchased", ["amount", "region", "purchased"])
        self.assertEqual(result["features"], ["amount", "region"])
        self.assertEqual(result["train_rows"] + result["test_rows"], 60)
        self.assertEqual(result["details"]["confusion_matrix"]["labels"], ["no", "yes"])
        self.assertEqual(sum(map(sum, result["details"]["confusion_matrix"]["values"])), result["test_rows"])
        self.assertIn("Majority baseline accuracy", result["metrics"])
        self.assertTrue(0 <= result["metrics"]["Macro F1"] <= 1)
        json.dumps(result, allow_nan=False)

    def test_regression_beats_mean_baseline_for_linear_data(self):
        result = train_model(self.data, "regression", "price", ["amount", "region"])
        self.assertLess(result["metrics"]["RMSE"], 0.000001)
        self.assertGreater(result["metrics"]["Mean baseline RMSE"], result["metrics"]["RMSE"])
        self.assertAlmostEqual(result["metrics"]["R2"], 1.0)
        json.dumps(result, allow_nan=False)

    def test_clustering_uses_numeric_features_and_counts_every_row(self):
        frame = pd.DataFrame({"x": [0, 0.1, 0.2, 10, 10.1, 10.2], "label": ["a"] * 6})
        result = train_model(frame, "clustering", n_clusters=2)
        self.assertEqual(result["features"], ["x"])
        self.assertEqual(sum(result["details"]["cluster_sizes"].values()), len(frame))
        self.assertGreater(result["metrics"]["Silhouette score"], 0.9)
        self.assertEqual(result["test_rows"], 0)
        json.dumps(result, allow_nan=False)

    def test_imputer_fits_only_training_rows(self):
        # A spy on the actual fit input guards against accidentally moving
        # preprocessing above train_test_split during a future edit.
        frame = self.data[["amount", "price"]].copy()
        frame.loc[[0, 4, 9], "amount"] = np.nan
        seen_indices = []
        original_fit = SimpleImputer.fit

        def record_fit(imputer, X, y=None):
            seen_indices.extend(X.index.tolist())
            return original_fit(imputer, X, y)

        with patch.object(SimpleImputer, "fit", record_fit):
            train_model(frame, "regression", "price", ["amount"])
        expected_train, expected_test = train_test_split(frame, test_size=0.2, random_state=42)
        self.assertEqual(set(seen_indices), set(expected_train.index))
        self.assertTrue(set(seen_indices).isdisjoint(set(expected_test.index)))

    def test_missing_values_and_unseen_test_category_are_supported(self):
        frame = self.data.copy()
        frame.loc[[0, 1, 2], "amount"] = np.nan
        frame.loc[[3, 4, 5], "region"] = None
        frame["empty_number"] = np.nan
        frame["empty_text"] = pd.Series([None] * len(frame), dtype=object)
        # Every row has a different category, so test categories are all unseen.
        frame["row_name"] = [f"record-{i}" for i in range(len(frame))]
        result = train_model(frame, "regression", "price", ["amount", "region", "empty_number", "empty_text", "row_name"])
        self.assertTrue(np.isfinite(result["metrics"]["RMSE"]))
        json.dumps(result, allow_nan=False)

    def test_duplicate_rows_removed_before_split(self):
        frame = pd.concat([self.data, self.data], ignore_index=True)
        result = train_model(frame, "regression", "price", ["amount", "region"])
        self.assertEqual(result["row_count"], len(self.data))
        self.assertTrue(any("60 exact duplicate" in note for note in result["notes"]))

    def test_missing_targets_are_removed(self):
        frame = self.data.copy()
        frame.loc[[1, 2], "price"] = np.nan
        result = train_model(frame, "regression", "price", ["amount"])
        self.assertEqual(result["row_count"], 58)

    def test_training_does_not_modify_uploaded_data(self):
        frame = self.data.copy()
        frame.loc[0, "amount"] = np.inf
        before = frame.copy(deep=True)
        train_model(frame, "regression", "price", ["amount", "region"])
        pd.testing.assert_frame_equal(frame, before)

    def test_invalid_inputs_have_readable_errors(self):
        cases = [
            ({"task": "other"}, "Choose classification"),
            ({"task": "classification", "target": "missing"}, "target column"),
            ({"task": "classification", "target": "purchased", "features": []}, "at least one"),
            ({"task": "regression", "target": "price", "features": ["missing"]}, "do not exist"),
            ({"task": "regression", "target": "price", "test_size": 0.99}, "test fraction"),
            ({"task": "regression", "target": "price", "test_size": True}, "test fraction"),
            ({"task": "regression", "target": "purchased"}, "numeric target"),
            ({"task": "clustering", "n_clusters": 1}, "2 and 10"),
            ({"task": "clustering", "n_clusters": 2.5}, "integer"),
            ({"task": "clustering", "features": ["region"]}, "numeric input"),
        ]
        for settings, message in cases:
            with self.subTest(settings=settings):
                with self.assertRaisesRegex(ValueError, message):
                    train_model(self.data, **settings)

    def test_one_class_and_rare_classes_rejected(self):
        frame = self.data.copy()
        frame["purchased"] = "yes"
        with self.assertRaisesRegex(ValueError, "at least two target classes"):
            train_model(frame, "classification", "purchased", ["amount"])
        frame.loc[0, "purchased"] = "no"
        with self.assertRaisesRegex(ValueError, "at least two rows"):
            train_model(frame, "classification", "purchased", ["amount"])

    def test_too_many_classes_and_tiny_test_set_rejected(self):
        with self.assertRaisesRegex(ValueError, "at most 20"):
            train_model(self.data, "classification", "price", ["amount"])
        frame = pd.DataFrame({"x": range(12), "target": ["a", "b", "c", "d"] * 3})
        with self.assertRaisesRegex(ValueError, "one row per class"):
            train_model(frame, "classification", "target", ["x"], test_size=0.1)

    def test_rare_class_cannot_disappear_from_evaluation(self):
        frame = self.data.copy()
        frame["purchased"] = ["rare"] * 2 + ["common"] * 58
        with self.assertRaisesRegex(ValueError, "rare class is missing"):
            train_model(frame, "classification", "purchased", ["amount"])

    def test_tiny_dataset_and_impossible_clusters_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least 10"):
            train_model(self.data.head(4), "regression", "price", ["amount"])
        with self.assertRaisesRegex(ValueError, "more rows"):
            train_model(pd.DataFrame({"x": [1, 2]}), "clustering", n_clusters=2)
        with self.assertRaisesRegex(ValueError, "fewer distinct"):
            train_model(pd.DataFrame({"x": [1] * 10}), "clustering", n_clusters=2)

    def test_limits_and_empty_features(self):
        with self.assertRaisesRegex(ValueError, "20,000"):
            train_model(pd.DataFrame({"x": range(20001)}), "clustering")
        frame = pd.DataFrame({f"x{i}": range(20) for i in range(31)})
        with self.assertRaisesRegex(ValueError, "at most 30"):
            train_model(frame, "clustering")
        frame = pd.DataFrame({"empty": [np.nan] * 20, "target": range(20)})
        with self.assertRaisesRegex(ValueError, "no usable values"):
            train_model(frame, "regression", "target", ["empty"])


if __name__ == "__main__":
    unittest.main()
