"""CSV loading, descriptive statistics, cleaning, and charts.

These are ordinary functions: each takes data in and returns a result.
They do not know anything about Flask or web requests.
"""

import base64
import csv
from io import BytesIO, StringIO

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

MAX_ROWS = 20_000
MAX_COLUMNS = 100
MAX_BYTES = 10 * 1024 * 1024


def load_csv(file_bytes):
    """Read a small UTF-8, comma-separated CSV and explain bad input clearly."""
    if len(file_bytes) > MAX_BYTES:
        raise ValueError("Choose a CSV smaller than 10 MB.")
    try:
        text = file_bytes.decode("utf-8-sig")  # Also accepts Excel's UTF-8 BOM.
    except UnicodeDecodeError as error:
        raise ValueError("Save your file as CSV UTF-8 and upload it again.") from error
    if not text.strip() or "\x00" in text:
        raise ValueError("The file is empty or is not a readable CSV.")

    # Check headers before pandas automatically renames repeated column names.
    try:
        reader = csv.reader(StringIO(text), strict=True)
        headers = [name.strip() for name in next(reader)]
        if not headers or any(not name or len(name) > 100 for name in headers):
            raise ValueError("Every column needs a name between 1 and 100 characters.")
        if len(headers) != len(set(headers)):
            raise ValueError("Column names must be unique, including after trimming spaces.")
        if len(headers) > MAX_COLUMNS:
            raise ValueError("Use at most 100 columns.")
        row_count = 0
        for row in reader:
            if not row:  # Ignore completely blank lines, as pandas does.
                continue
            if len(row) != len(headers):
                raise ValueError("Each CSV row must have the same number of fields as the header.")
            row_count += 1
            if row_count > MAX_ROWS:
                raise ValueError("Use at most 20,000 rows for this learning project.")
        frame = pd.read_csv(StringIO(text), nrows=MAX_ROWS + 1)
    except (csv.Error, StopIteration, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        raise ValueError("The CSV could not be read. Check its header, commas, and quotes.") from error

    if frame.empty:
        raise ValueError("The CSV needs a header and at least one data row.")
    frame.columns = headers
    # Whitespace-only strings and infinity are not useful observations.
    for name in frame.select_dtypes(include=["object", "string"]).columns:
        frame[name] = frame[name].str.strip().replace("", np.nan)
    frame = frame.replace([np.inf, -np.inf], np.nan)
    numeric = frame.select_dtypes(include="number")
    if not numeric.empty and (numeric.abs() > 1e100).any().any():
        raise ValueError("Some numbers are too large to analyze reliably. Check their units or format.")
    return frame


def profile_data(frame):
    """Build a plain dictionary that both a web page and JSON can use."""
    numeric_names = frame.select_dtypes(include="number").columns.tolist()
    column_stats = []
    for name in frame.columns:
        series = frame[name]
        row = {
            "name": name,
            "dtype": "numeric" if name in numeric_names else "categorical",
            "missing": int(series.isna().sum()),
            "unique": int(series.nunique()),
            "mean": None, "median": None, "std": None,
            "min": None, "max": None, "outliers": None,
        }
        if name in numeric_names and series.notna().any():
            for label in ["mean", "median", "std", "min", "max"]:
                value = getattr(series, label)()
                row[label] = round(float(value), 4) if pd.notna(value) else None
            q1, q3 = series.quantile([0.25, 0.75])
            iqr = q3 - q1
            row["outliers"] = int(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum())
        column_stats.append(row)
    missing = int(frame.isna().sum().sum())
    return {
        "rows": len(frame), "columns": len(frame.columns),
        "missing_cells": missing,
        "missing_percent": round(100 * missing / frame.size, 2),
        "duplicates": int(frame.duplicated().sum()),
        "numeric_count": len(numeric_names),
        "categorical_count": len(frame.columns) - len(numeric_names),
        "column_stats": column_stats,
    }


def clean_data(frame, missing_strategy="leave", remove_duplicates=False):
    """Clean a COPY for exploration/export. ML has its own train-only pipeline."""
    if missing_strategy not in {"leave", "fill", "drop"}:
        raise ValueError("Choose leave, fill, or drop for missing values.")
    cleaned = frame.copy()
    if remove_duplicates:
        cleaned = cleaned.drop_duplicates()
    if missing_strategy == "drop":
        cleaned = cleaned.dropna()
    elif missing_strategy == "fill":
        for name in cleaned.columns:
            if pd.api.types.is_numeric_dtype(cleaned[name]):
                median = cleaned[name].median() if cleaned[name].notna().any() else 0
                cleaned[name] = cleaned[name].fillna(median)
            else:
                modes = cleaned[name].mode()
                fill_value = modes.iloc[0] if not modes.empty else "Unknown"
                cleaned[name] = cleaned[name].fillna(fill_value)
    if cleaned.empty:
        raise ValueError("That would remove every row. Try filling missing values instead.")
    return cleaned.reset_index(drop=True)


def csv_for_download(frame):
    """Stop text cells from being interpreted as formulas by spreadsheet apps."""
    safe = frame.copy()

    def escape_cell(value):
        if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
            return "'" + value
        return value

    for name in safe.columns:
        if not pd.api.types.is_numeric_dtype(safe[name]):
            safe[name] = safe[name].map(escape_cell)
    # Escape headers last: two different names can become equal when prefixed.
    safe.columns = [escape_cell(str(name)) for name in safe.columns]
    return safe.to_csv(index=False).encode("utf-8-sig")


def figure_to_base64(figure):
    """Embed a PNG directly in HTML, so charts do not need an internet service."""
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=120, bbox_inches="tight", facecolor="#ffffff")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def make_charts(frame, selected_column=None):
    """Small, predictable plots. Limit crowded charts to ten columns."""
    charts = []
    numeric = frame.select_dtypes(include="number")
    if selected_column in numeric.columns and numeric[selected_column].notna().any():
        figure = Figure(figsize=(6, 3.6), layout="constrained")
        axis = figure.subplots()
        axis.hist(numeric[selected_column].dropna(), bins=20, color="#187d75", edgecolor="white")
        axis.set(xlabel=selected_column, ylabel="Number of rows")
        axis.spines[["top", "right"]].set_visible(False)
        charts.append({"title": f"Distribution of {selected_column}", "image": figure_to_base64(figure)})

    missing = frame.isna().sum().sort_values(ascending=False).head(10)
    if missing.sum() > 0:
        figure = Figure(figsize=(6, 3.6), layout="constrained")
        axis = figure.subplots()
        missing = missing[missing > 0]
        axis.barh([str(name)[:30] for name in missing.index], missing.values, color="#d6854e")
        axis.set_xlabel("Missing cells (up to 10 columns)")
        axis.spines[["top", "right"]].set_visible(False)
        charts.append({"title": "Where values are missing", "image": figure_to_base64(figure)})

    # Constant columns have undefined correlation, so leave them out.
    usable = [name for name in numeric if numeric[name].nunique() > 1][:10]
    if len(usable) >= 2:
        correlation = numeric[usable].corr()
        figure = Figure(figsize=(6, 4.6), layout="constrained")
        axis = figure.subplots()
        picture = axis.imshow(correlation, cmap="BrBG", vmin=-1, vmax=1)
        labels = [str(name)[:20] for name in usable]
        axis.set_xticks(range(len(labels)), labels, rotation=40, ha="right")
        axis.set_yticks(range(len(labels)), labels)
        figure.colorbar(picture, ax=axis, shrink=0.7, label="Pearson correlation")
        charts.append({"title": "Numeric relationships · first 10 usable columns", "image": figure_to_base64(figure)})
    return charts
