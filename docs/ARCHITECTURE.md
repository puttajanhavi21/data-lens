# How the project fits together

This application has one Python server and ordinary HTML pages. There is no separate JavaScript frontend framework, cloud database, or background task queue to learn first.

## One request, from beginning to end

```text
Browser form
    |
    | HTTP request
    v
app.py (Flask route validates the request)
    |
    +--> analysis.py --> Pandas DataFrame --> statistics / cleaning / charts
    |
    +--> ml.py -------> Scikit-learn pipeline --> metrics
    |
    +--> insights.py -> local rules OR optional OpenAI request
    |
    +--> database.py -> SQLite records
    |
    v
Jinja template + CSS
    |
    | HTML response
    v
Browser displays the result
```

`app.py` coordinates the work. The other modules contain functions with a narrower responsibility. For example, a model function should compute model results; it should not decide which HTML page to display.

## Route map

| Method and path | Handler | What it does |
| --- | --- | --- |
| `GET /` | `index` | Display the home page |
| `POST /upload` | `upload` | Validate and store an uploaded CSV |
| `POST /sample/<name>` | `sample` | Load an included synthetic dataset |
| `GET /dashboard` | `dashboard` | Build the profile, preview, charts, and latest results |
| `POST /clean` | `clean` | Rebuild the working copy using the chosen recipe |
| `POST /reset` | `reset` | Restore the normalized original to the working copy |
| `GET /download` | `download` | Export the working copy as CSV |
| `POST /train` | `train` | Evaluate a model using the original data |
| `POST /explain` | `explain` | Generate and save a local or consented AI explanation |
| `GET /history` | `history` | List the browser session's saved model runs |
| `POST /select/<dataset_id>` | `select_dataset` | Reopen a dataset owned by this browser session |

`prepare_request` runs before each route. It prepares the session, checks POST form tokens, and retrieves the current dataset. `require_dataset` gives routes a shared way to reject requests when no valid dataset is selected.

## Example: upload a CSV

1. A user chooses a file and submits an HTML form.
2. Flask receives the file in a POST request.
3. The route checks the form token and the upload limits.
4. `analysis.py` parses and validates the CSV.
5. The application saves the dataset locally and records its metadata in SQLite.
6. A results page shows a preview, profile, and charts.

The generated file identifier is separate from the original filename. A filename supplied by a browser should never be trusted as a path on the server.

## Example: train a supervised model

1. The form supplies a task, target, feature selection, and training settings.
2. The application loads the original dataset belonging to the browser session.
3. `ml.py` validates the requested columns, removes unusable target rows, and removes identical selected feature-and-target rows.
4. A train/test split reserves rows for final evaluation.
5. A pipeline learns feature preprocessing from training data and fits a model.
6. The fitted pipeline transforms the held-out rows and predicts their targets.
7. Metrics compare predictions with the real test targets. A simple baseline provides context.
8. SQLite stores the settings and results for later review.

This model object lives only for the request. The application stores the experiment results, not a reusable trained model file.

## Why use a pipeline?

Suppose a numeric feature has missing values. The imputer needs to learn a replacement, such as the training median. A scaler then learns the training mean and spread. A model learns its coefficients from those transformed values.

```text
Training rows --> fit preprocessing --> fit model
Test rows -----> use fitted preprocessing --> predict --> evaluate
```

`ColumnTransformer` applies numeric and categorical preprocessing to the correct columns. `Pipeline` keeps those steps attached to the model. This reduces the chance of using a different transformation when evaluating it.

Never fit an imputer or scaler on the whole dataset before the split. Information from the test set would influence training. The [Scikit-learn common pitfalls guide](https://scikit-learn.org/stable/common_pitfalls.html) explains this failure and how pipelines help avoid it.

## Why exploration and model preprocessing differ

The cleaning controls let a user see what a transformation does and download its output. A full-dataset median is useful for that exploratory view.

Model evaluation has a different requirement: its learned preprocessing must use only the training rows. Training therefore reads the original upload and performs its own preprocessing inside a pipeline. It does not silently reuse values filled in the exploratory view.

This separation is a deliberate design choice. Explain it during your demo so users understand which data the model sees.

## The database

SQLite is a relational database stored in a local file. Python includes the `sqlite3` module, so no separate database server needs to be installed.

The two main kinds of records are:

| Record | Purpose |
| --- | --- |
| Dataset | Identify the uploaded data, its owner session, name, and metadata; the identifier determines its saved location |
| Model run | Associate a task, settings, metrics, and time with a dataset |

Some structured results are encoded as JSON text inside SQL records. SQL manages relationships and retrieval; JSON conveniently preserves a small metrics dictionary.

SQL queries use `?` placeholders and pass values separately. For example:

```python
cursor.execute(
    "SELECT * FROM datasets WHERE id = ? AND owner = ?",
    (dataset_id, owner),
)
```

The values in `(dataset_id, owner)` fill the two placeholders in order. Do not insert user text into SQL with an f-string.

## Browser sessions and form tokens

The app generates a secret for signing the session cookie and persists it in `instance/secret.key` unless configured otherwise. The session identifies the current browser and helps associate it with its uploaded datasets. The cookie is signed to detect modification; it is not a place to store the data itself or an API key.

A CSRF token is a random value included in forms and checked on state-changing requests. It helps reject a forged form submission from another site. It is separate from dataset ownership checks.

These controls support a local learning app. They do not replace accounts, authentication, authorization design, HTTPS, deployment configuration, or operational security for a public service.

## The explanation boundary

The local path constructs a plain-English explanation from computed statistics using explicit Python conditions. You can read every rule.

The optional generative path sends a bounded summary and the user's question to OpenAI after consent. This call may fail independently of the analysis. The app can still display its computed results and local explanation.

Raw uploaded rows are not part of the AI request. Column names and entered questions can still reveal information, which is why the form asks for consent and the README describes what is sent.

## Limitations that suggest future work

There is one straightforward request/response path. Large jobs would block a request, so this project deliberately limits file size and model inputs. Scaling it later could involve a background worker, job status, and stronger resource isolation.

Evaluation is a reproducible educational split, not proof of business readiness. Real data may require grouped or time-based splitting, cross-validation, careful label definitions, and external validation.

K-means groups numeric patterns; it does not discover objectively correct business categories. Interpreting and validating those groups remains a human task.
