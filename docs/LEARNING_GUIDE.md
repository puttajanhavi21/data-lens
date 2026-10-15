# Learn Data Lens by rebuilding it

The goal is to understand the decisions, not memorize every line. Work on a separate copy, build one small part at a time, and explain it aloud before moving on.

You should first be comfortable with Python variables, lists, dictionaries, loops, functions, imports, and `try`/`except`. For HTML, learn a heading, a form, an input, a button, and a table. You can learn the remaining ideas while building this project.

## A useful routine

For each stage: run the existing feature, read its function, write a smaller version without looking, compare the behavior, and then add validation. Keep a short note explaining one thing you misunderstood and how you resolved it. Do not rewrite the entire application in one sitting.

## Stage 1: inspect data without a website

**Read:** `generate_samples.py`, then `load_csv` and `profile_data` in `analysis.py`.

A Pandas `DataFrame` is a table with named columns. A `Series` is one column. NumPy provides numerical operations and repeatable random sample generation.

Try this in a temporary Python script from the project folder:

```python
import pandas as pd

frame = pd.read_csv("sample_data/classification.csv")
print(frame.head())
print(frame.shape)
print(frame.dtypes)
print(frame.isna().sum())
```

`frame.shape` is a `(rows, columns)` pair. `isna()` produces a table of booleans; `sum()` counts `True` values. This is why `frame.isna().sum().sum()` counts all missing cells.

In the real loader, CSV bytes are decoded, headers are checked, and every row must have the right number of fields. Names are trimmed, whitespace-only cells and infinity become missing values, and unusually large numbers are rejected. This normalized upload is what later pages call the **original dataset**.

In `profile_data`, each column produces a small dictionary. The result is a dictionary of plain values that a template or explanation function can use. `int(...)` and `float(...)` convert library-specific scalar types into ordinary Python values.

**Write yourself:** a function returning row count, column count, missing-cell count, and duplicate count. Then add a numeric column's mean and median.

**Check:** use a five-row CSV whose answer you can calculate by hand. Include one missing value and one duplicate. Explain why a mean and median can differ.

## Stage 2: clean data and draw one chart

**Read:** `clean_data`, `csv_for_download`, `figure_to_base64`, and `make_charts` in `analysis.py`.

`frame.copy()` keeps a transformation from unexpectedly changing the input. The cleaning function offers three missing-value choices: leave values alone, fill them, or remove rows containing them. Duplicate removal is a separate choice.

Numeric filling uses the median; categorical filling uses the most frequent value, called the mode. Completely empty columns need a fallback because they have no median or mode. The exploratory cleaner uses `0` for numeric columns and `Unknown` for text columns.

```python
def fill_numeric_column(series):
    replacement = series.median()
    return series.fillna(replacement)
```

This miniature example is deliberately incomplete: what happens when every value is missing? Compare it with the actual implementation and add the missing case yourself.

Every cleaning action in the app starts from the original upload, so choices are **not cumulative**. Set the full combination you want each time. Reset returns to the normalized original. Modeling also starts from that original; its preprocessing is covered in Stage 3.

Matplotlib creates a figure in memory. `BytesIO` is a file-like buffer in memory. Base64 turns image bytes into text that an HTML image element can carry. The charts do not depend on an external chart service.

CSV downloads prefix formula-leading text cells or headers with an apostrophe so spreadsheet software treats them as text. This can change how those particular text values look if you upload the exported file again.

**Write yourself:** a cleaner that returns a copy, followed by one histogram. Add duplicate removal after you understand the missing-value branch.

**Check:** confirm that the original DataFrame did not change. Try a column that is entirely missing. Try a cleaning choice that would remove every row and explain the app's validation message.

## Stage 3: train one model correctly

**Read:** `_prepare_features`, `_preprocessor`, and `train_model` in `ml.py`.

A leading underscore marks a helper intended for use inside the module. `train_model` is the main function the Flask application calls. Start with its classification branch; leave clustering until the supervised flow makes sense.

The **target**, usually called `y`, is the outcome you want to predict. The **features**, usually called `X`, are the input columns. A feature that contains the answer, or information unavailable at prediction time, can make evaluation misleading.

For supervised learning, the function drops rows with missing targets and removes identical selected feature-and-target rows before splitting. It requires at least ten usable unique rows. Classification also requires enough rows per class to put classes in both train and test sets.

The main sequence is:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

This is a teaching fragment, not the complete implementation: it assumes the imports and variables already exist. The actual classification code also uses `stratify=y` to help preserve class proportions.

`random_state=42` makes the random choices reproducible with the same data and environment. The number 42 is not a special machine learning setting. The default test fraction is 0.2: approximately 20% of usable rows are reserved for evaluation.

### Follow the pipeline

| Input | Preprocessing | Why |
| --- | --- | --- |
| Numeric columns | Median imputation, then scaling | Replace missing values and put features on comparable scales |
| Text/category columns | Most-frequent imputation, then one-hot encoding | Convert categories into numeric indicator columns |

`ColumnTransformer` applies each recipe to the right columns. `Pipeline` attaches preprocessing to the estimator. Calling `model.fit(X_train, y_train)` fits every learned preprocessing step on **training rows only**. Calling `predict(X_test)` reuses those learned values.

One-hot encoding can create several columns from one categorical feature. The implementation limits the number of encoded categories per feature and tolerates unseen test categories. Read the encoder settings and explain their purpose.

### Add the other tasks

Logistic regression predicts classes even though its name includes “regression.” Linear regression predicts numeric quantities. Both are supervised: the training data includes known target values.

K-means is unsupervised: it groups numeric feature patterns without a target. Scaling and imputation use the clustering dataset; there is no held-out prediction accuracy claim. The app uses a sample of up to 1,000 rows when computing silhouette to limit computation. Cluster numbers are arbitrary labels, not ordered rankings.

**Write yourself:** first fit logistic regression on two complete numeric columns, then add a pipeline, missing data, and a categorical column. Add regression next. Finally add clustering.

**Check:** intentionally include missing feature values, a one-class target, and an invalid target selection. Explain why the app can fill a missing feature but should not invent a missing supervised target.

## Stage 4: save results with SQL

**Read:** `connect`, `init_db`, `add_dataset`, `save_run`, and `list_runs` in `database.py`.

SQLite is a database stored in a file. A table has rows and named columns, similar to a spreadsheet but with relationships and query rules. `CREATE TABLE IF NOT EXISTS` initializes the schema without deleting existing records.

The `datasets` table has an identifier and an owner-session value. The `model_runs` table refers to the dataset identifier and stores the model report as JSON. JSON is a text format for dictionaries, lists, strings, numbers, booleans, and null values.

`json.dumps(result)` turns a report into text; `json.loads(text)` reconstructs it. The database code rejects non-finite JSON numbers rather than silently writing invalid reports.

Parameterized SQL keeps values separate from instructions:

```python
connection.execute(
    "SELECT * FROM datasets WHERE id = ? AND owner = ?",
    (dataset_id, owner),
)
```

`commit()` makes writes persistent. The context manager closes each connection after use. `JOIN` in `list_runs` connects model runs with their dataset records so the app can retrieve only the current owner's history.

**Write yourself:** create a tiny SQLite table, insert a report, and read it back. Use a temporary database while experimenting.

**Check:** save two runs and confirm their order. Explain why model settings must be saved with scores if you want a meaningful comparison.

## Stage 5: connect Python to a browser

**Read:** `app.py`, then `templates/` and `static/`.

Flask maps a URL and HTTP method to a Python function called a route handler. A GET usually asks for a page; a POST submits a form that changes something or starts work. `create_app` constructs and configures the Flask application, which also lets tests create an isolated version.

```python
@app.get("/")
def home():
    return render_template("index.html")
```

This is a small illustrative route, not a replacement for the actual application's home route. The decorator associates the URL with the function. A template is HTML with places for Python-supplied values; Jinja evaluates those places before the browser receives HTML.

Trace one real form: find its `action` and `method`, find the matching route, list the form fields the route reads, and follow the return statement to its template or redirect. Then trace the data passed into that template.

The browser session associates uploads with one browser. A separate CSRF token is checked on form submissions. The server creates a persistent signing secret automatically. Do not put uploaded data or API credentials in the cookie.

**Write yourself:** a minimal Flask page, a CSV upload form, and a results table. Add one feature at a time from earlier stages.

**Check:** submit an empty form, a bad CSV, and a valid CSV. Confirm that errors explain what the user can change instead of displaying an internal traceback.

## Stage 6: explain the results

**Read:** `local_explanation`, `ai_configured`, and `ai_explanation` in `insights.py`.

Start with local rules. An `if` statement checks whether values are missing, then adds a sentence explaining the issue. Every statement comes from a computed result or a general explanation of a metric. This is easy to inspect and runs without a key.

The optional AI function builds an explicit summary payload. It calls the OpenAI Responses API using a model configured in `.env`. The Flask route requires consent before that call. No raw CSV rows are sent, but names, numeric summaries, metrics, and the user's question are included.

The `try`/`except` blocks keep a missing package or failed API request from breaking the ordinary analysis. A warning explains that the app is showing local rules instead. Raw provider errors and credentials are not displayed in the browser.

**Write yourself:** add a local explanation for one statistic, such as a high missing-value percentage. State a limitation instead of guessing why the data looks that way.

**Check:** run without a key and confirm that local explanations work. If you configure an API account, inspect the constructed payload in code and verify a consented request; never print the key.

## Stage 7: verify and rebuild without copying

**Read:** the tests in `tests/`. Run `python -m unittest discover -s tests -v` with the virtual environment's Python.

A useful test checks a behavior that matters: invalid CSVs are rejected, the original data stays unchanged after cleaning, model metrics are valid, or one browser cannot retrieve another browser's dataset. A test that repeats a function's exact implementation is less useful.

Create a new practice folder and rebuild these pieces from your own notes:

1. Load a CSV and return a small profile dictionary.
2. Clean a copy and draw a histogram.
3. Split rows, fit a preprocessing/model pipeline, and compute metrics.
4. Save and retrieve one model report in SQLite.
5. Add a Flask form and display the results.
6. Explain the result with one local rule.
7. Add validation and run meaningful checks.

Then add an improvement you chose yourself. Good first changes are a numeric range statistic, a target-distribution chart, or an extra explanation rule. Record what you changed and why in your own development log.

You are ready to discuss this project when you can predict what a function returns, explain its main edge cases, and rebuild a smaller version without viewing the solution.

## Metric reference

| Metric | Meaning | Interpretation |
| --- | --- | --- |
| Accuracy | Correct classification predictions divided by all test predictions | Higher is better, but imbalance can mislead |
| Precision | Among predictions of a class, how many are correct? | Useful when false positives are costly |
| Recall | Among actual members of a class, how many are found? | Useful when missed positives are costly |
| Macro F1 | Mean of class-specific F1 scores, giving classes equal weight | Summarizes precision and recall across classes |
| MAE | Mean absolute prediction error | Lower is better; measured in target units |
| RMSE | Square root of mean squared error | Lower is better; large errors count more heavily |
| R-squared | One minus model squared error divided by variation around the test-target mean | Can be negative; unavailable here for a constant test target |
| Silhouette | How close a point is to its own cluster versus other clusters | Roughly -1 to 1; not predictive accuracy |
| Inertia | Sum of squared distances to assigned cluster centers | Depends on scaling and cluster count; do not compare blindly |

The classification baseline predicts the **training majority class**. The regression baseline predicts the **training target mean**. They are evaluated on the same test rows as their corresponding models.

## Plain-language glossary

| Term | Meaning in this project |
| --- | --- |
| HTTP | The request/response protocol used between browser and server |
| Route | A URL and method mapped to a Python function |
| Jinja | The template system that combines HTML with server-provided values |
| Pandas | The library used to manipulate tables of data |
| NumPy | The library used for numerical arrays and computations |
| SQL | The language used to define, query, and update database tables |
| EDA | Exploratory data analysis: checking patterns, quality, and relationships |
| ML | Machine learning: fitting patterns from examples or grouping observations |
| Pipeline | An ordered combination of preprocessing steps and a model |
| Train/test split | Reserving some rows to evaluate a model after fitting it |
| Leakage | Information reaching training that would be unavailable in a fair evaluation |
| Baseline | A simple reference strategy used to judge a more complicated model |
| API | A defined way for one program to request work from another |
| Environment variable | A setting supplied outside ordinary source code, such as an API key |

For an answer you cannot explain yet, use the official references in the README, then return to the matching function in this project.
