"""Small, readable machine learning workflows for Data Lens.

The important order is: split the rows, learn preprocessing on training rows,
fit the model, and finally evaluate on untouched test rows. A scikit-learn
Pipeline keeps those steps together so test data cannot leak into training.
"""

import math
import time

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _numeric_columns(frame):
    """Booleans are categories; ordinary integer and decimal columns are numeric."""
    return [
        column
        for column in frame.columns
        if is_numeric_dtype(frame[column])
        and not is_bool_dtype(frame[column])
        and not is_complex_dtype(frame[column])
    ]


def _prepare_features(frame):
    """Normalize missing values without learning anything from the data."""
    result = frame.copy()
    numeric = _numeric_columns(result)
    categorical = [column for column in result.columns if column not in numeric]

    for column in numeric:
        # Missing values and infinity will be filled by the training-only imputer.
        result[column] = pd.to_numeric(result[column]).astype(float)
        result[column] = result[column].replace([np.inf, -np.inf], np.nan)
    for column in categorical:
        # Categories need one consistent type. Do not turn missing values into
        # the literal strings "nan" or "<NA>".
        result[column] = result[column].map(
            lambda value: np.nan if pd.isna(value) else str(value)
        ).astype(object)
    return result, numeric, categorical


def _preprocessor(numeric, categorical):
    """Build simple, separate recipes for numbers and text categories."""
    transformers = []
    if numeric:
        numeric_steps = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numbers", numeric_steps, numeric))
    if categorical:
        categorical_steps = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent", keep_empty_features=True),
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore", max_categories=20, sparse_output=False
                    ),
                ),
            ]
        )
        transformers.append(("categories", categorical_steps, categorical))
    return ColumnTransformer(transformers)


def train_model(df, task, target=None, features=None, test_size=0.2, n_clusters=3):
    """Train one beginner-friendly model and return a JSON-compatible report.

    Args:
        df: A pandas DataFrame, usually read from an uploaded CSV.
        task: "classification", "regression", or "clustering".
        target: The outcome column for classification or regression.
        features: An optional list of input columns. None selects columns
            automatically. The target is always removed from the inputs.
        test_size: Fraction of supervised rows reserved for evaluation (0.1–0.4).
        n_clusters: Number of groups for K-means (2–10).

    Raises:
        ValueError: A readable explanation of invalid data or settings.
    """
    started = time.perf_counter()
    notes = []

    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Upload a non-empty dataset before training a model.")
    if len(df) > 20000:
        raise ValueError("Use at most 20,000 rows for this learning project.")
    if not df.columns.is_unique or not all(isinstance(c, str) for c in df.columns):
        raise ValueError("Column names must be unique text values.")
    if task not in ("classification", "regression", "clustering"):
        raise ValueError("Choose classification, regression, or clustering.")

    supervised = task != "clustering"
    if supervised:
        if not isinstance(target, str) or target not in df.columns:
            raise ValueError("Select an existing target column.")
        if (
            isinstance(test_size, bool)
            or not isinstance(test_size, (int, float))
            or not math.isfinite(test_size)
            or not 0.1 <= test_size <= 0.4
        ):
            raise ValueError("The test fraction must be between 0.1 and 0.4.")
    else:
        target = None
        if isinstance(n_clusters, bool) or not isinstance(n_clusters, int):
            raise ValueError("The number of clusters must be an integer from 2 to 10.")
        if not 2 <= n_clusters <= 10:
            raise ValueError("Choose between 2 and 10 clusters.")

    if features is None:
        features = [column for column in df.columns if column != target]
        if not supervised:
            features = _numeric_columns(df[features])
            notes.append("Clustering automatically uses numeric columns only.")
        else:
            usable = [
                column for column in features
                if not is_complex_dtype(df[column])
                and not pd.api.types.is_datetime64_any_dtype(df[column])
            ]
            if len(usable) != len(features):
                notes.append("Automatic feature selection skipped date/time and complex-number columns.")
            features = usable
    elif not isinstance(features, list) or not all(isinstance(c, str) for c in features):
        raise ValueError("Features must be a list of column names.")
    else:
        # Keep the original order, remove duplicates, and never include the answer.
        features = list(dict.fromkeys(c for c in features if c != target))

    if not features:
        raise ValueError("Select at least one input feature other than the target.")
    if len(features) > 30:
        raise ValueError("Choose at most 30 input features.")
    if any(column not in df.columns for column in features):
        raise ValueError("One or more selected feature columns do not exist.")
    for column in features:
        if is_complex_dtype(df[column]) or pd.api.types.is_datetime64_any_dtype(df[column]):
            raise ValueError(f"Convert '{column}' to ordinary numbers or text first.")
    if not supervised and len(_numeric_columns(df[features])) != len(features):
        raise ValueError("Clustering requires numeric input columns only.")

    working = df[features + ([target] if supervised else [])].copy()
    if supervised:
        if task == "regression":
            # A regression target must really be numeric; silent conversion of
            # arbitrary labels into missing values could hide a wrong task choice.
            if is_complex_dtype(working[target]) or pd.api.types.is_datetime64_any_dtype(working[target]):
                raise ValueError("Regression needs an ordinary numeric target column.")
            converted = pd.to_numeric(working[target], errors="coerce")
            if (working[target].notna() & converted.isna()).any():
                raise ValueError("Regression needs a numeric target column.")
            working[target] = converted.replace([np.inf, -np.inf], np.nan)
        before = len(working)
        working = working.dropna(subset=[target])
        missing_count = before - len(working)
        if missing_count:
            notes.append(f"Removed {missing_count} rows with a missing or non-finite target.")
        before = len(working)
        working = working.drop_duplicates()
        duplicate_count = before - len(working)
        if duplicate_count:
            notes.append(f"Removed {duplicate_count} exact duplicate feature-and-target rows before splitting.")
        if len(working) < 10:
            raise ValueError("Supervised learning needs at least 10 usable, unique rows.")
    elif len(working) <= n_clusters:
        raise ValueError("Use more rows than the number of clusters.")

    X, numeric, categorical = _prepare_features(working[features])
    if X.isna().all().all():
        raise ValueError("The selected features contain no usable values.")
    if X.isna().any().any():
        notes.append("Missing numeric values use the median; missing categories use the most frequent value. Entirely empty features use scikit-learn's constant fallback.")
    if categorical:
        notes.append("Text features are one-hot encoded, with at most 20 encoded categories per column. Unseen test categories are ignored.")
    notes.append("Remove identifiers and columns that reveal the answer before training.")

    report = {
        "task": task,
        "algorithm": "",
        "target": target,
        "features": features,
        "row_count": int(len(working)),
        "train_rows": 0,
        "test_rows": 0,
        "metrics": {},
        "notes": notes,
        "details": {},
        "elapsed_seconds": 0.0,
        "parameters": {"random_state": 42},
    }

    if not supervised:
        # Clustering has no known answer and no train/test accuracy. Scaling
        # prevents a large-unit feature from dominating Euclidean distance.
        prepared = _preprocessor(numeric, []).fit_transform(X)
        if np.unique(prepared, axis=0).shape[0] < n_clusters:
            raise ValueError("There are fewer distinct feature combinations than requested clusters. Choose fewer clusters or different features.")
        model = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        labels = model.fit_predict(prepared)
        score = None
        if 1 < len(np.unique(labels)) < len(labels):
            try:
                score = float(
                    silhouette_score(
                        prepared,
                        labels,
                        sample_size=min(1000, len(X)),
                        random_state=42,
                    )
                )
            except ValueError:
                # A small random sample can miss a very small cluster.
                notes.append("Silhouette score is unavailable because the evaluation sample does not contain enough different groups.")
        report["algorithm"] = "K-means"
        report["train_rows"] = len(X)
        report["metrics"] = {"Silhouette score": score, "Inertia": float(model.inertia_)}
        report["details"] = {
            "cluster_sizes": {
                str(group): int((labels == group).sum())
                for group in range(n_clusters)
            },
            "silhouette_sample_rows": min(1000, len(X)),
        }
        report["parameters"]["n_clusters"] = n_clusters
        notes.append("Cluster numbers are arbitrary labels. Silhouette measures separation; inertia measures within-cluster spread. Neither is predictive accuracy.")
    else:
        stratify = None
        if task == "classification":
            y = working[target].astype(str)
            counts = y.value_counts()
            if len(counts) < 2:
                raise ValueError("Classification needs at least two target classes.")
            if len(counts) > 20:
                raise ValueError("Classification supports at most 20 target classes. Use regression for a continuous numeric outcome.")
            if counts.min() < 2:
                raise ValueError("Every target class needs at least two rows for a train/test split.")
            test_rows = math.ceil(len(y) * test_size)
            if min(test_rows, len(y) - test_rows) < len(counts):
                raise ValueError("The train and test sets each need at least one row per class. Add data or change the test fraction.")
            stratify = y
            estimator = LogisticRegression(max_iter=1000, random_state=42)
            report["algorithm"] = "Logistic regression"
        else:
            y = working[target].astype(float)
            if y.nunique() < 2:
                raise ValueError("Regression needs at least two different target values.")
            estimator = LinearRegression()
            report["algorithm"] = "Linear regression"

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=stratify
        )
        if task == "classification" and (
            y_train.nunique() != y.nunique() or y_test.nunique() != y.nunique()
        ):
            raise ValueError("A rare class is missing from the train or test split. Add examples of that class or increase the test fraction.")
        model = Pipeline(
            [("preprocess", _preprocessor(numeric, categorical)), ("model", estimator)]
        )
        # fit() learns medians, category vocabularies, scaling, and model weights
        # from X_train only. predict() reuses those values on the test set.
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        report["train_rows"] = len(X_train)
        report["test_rows"] = len(X_test)
        report["parameters"]["test_size"] = float(test_size)
        notes.append("Preprocessing is fitted on training rows only. Metrics use the held-out test rows; results from one small split can vary.")
        notes.append("This random split assumes independent rows. Time series, repeated people, or grouped records need a different split strategy.")

        if task == "classification":
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, predictions, average="macro", zero_division=0
            )
            majority = y_train.mode().iloc[0]
            baseline = np.repeat(majority, len(y_test))
            labels = sorted(y.unique().tolist())
            report["metrics"] = {
                "Accuracy": float(accuracy_score(y_test, predictions)),
                "Macro precision": float(precision),
                "Macro recall": float(recall),
                "Macro F1": float(f1),
                "Majority baseline accuracy": float(accuracy_score(y_test, baseline)),
            }
            report["details"] = {
                "confusion_matrix": {
                    "labels": labels,
                    "values": confusion_matrix(
                        y_test, predictions, labels=labels
                    ).tolist(),
                },
                "majority_class": str(majority),
            }
            notes.append("Classification scores are fractions from 0 to 1. Macro scores give each class equal weight; confusion-matrix rows are actual classes and columns are predictions.")
        else:
            baseline = np.repeat(float(y_train.mean()), len(y_test))
            # R² is not informative when the test target has no variation.
            r2 = float(r2_score(y_test, predictions)) if y_test.nunique() > 1 else None
            report["metrics"] = {
                "MAE": float(mean_absolute_error(y_test, predictions)),
                "RMSE": float(np.sqrt(mean_squared_error(y_test, predictions))),
                "R2": r2,
                "Mean baseline RMSE": float(np.sqrt(mean_squared_error(y_test, baseline))),
            }
            report["details"] = {"training_target_mean": float(y_train.mean())}
            notes.append("MAE and RMSE use the target's units; lower is better. R2 can be negative. The baseline always predicts the training target mean.")
            if r2 is None:
                notes.append("R2 is unavailable because all test targets have the same value.")

    report["elapsed_seconds"] = round(time.perf_counter() - started, 4)
    return report
