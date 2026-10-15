# Data Lens

A Flask app for exploring CSV data, cleaning it, training simple machine learning models, and explaining results in plain English.

## Live deployment

The repository includes a GitHub Actions workflow that runs the test suite on every push and pull request, plus a `render.yaml` blueprint for a free Render web service.

To deploy from GitHub:

1. Create a new Render Blueprint and select this repository.
2. Render reads `render.yaml`, installs the pinned requirements, and starts Gunicorn.
3. Open the generated `onrender.com` URL.

The free service uses ephemeral storage. Uploaded CSVs, run history, and the SQLite database can be cleared when the service restarts or is redeployed. Use synthetic or non-sensitive data for the public demo. A persistent disk or managed database is needed for durable user data.

Start with the included synthetic datasets. No API key is required for data analysis, model training, or local explanations. An optional OpenAI integration adds genuine generative AI explanations.

## Start here

1. Extract the ZIP into a normal folder.
2. Install Python **3.11 or 3.12** if needed.
3. Open a terminal inside the extracted `data-lens` folder, where `app.py` lives.
4. Follow the commands for your operating system below.

You need an internet connection for the first dependency installation. After that, all features except the optional AI explanation run locally.

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

If you installed Python 3.11, replace `-3.12` with `-3.11`. Using the virtual environment's Python directly avoids PowerShell activation-policy problems.

### macOS or Linux

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python app.py
```

Use a Python 3.11 or 3.12 interpreter for the first command. On some Linux installations, you must install your distribution's Python `venv` package first.

Open **http://127.0.0.1:5000** in your browser. Leave the terminal running. To stop the app, press **Ctrl+C** in that terminal.

## Your first demo

1. Choose the classification sample on the home page, or upload `sample_data/classification.csv`.
2. Review the row count, column types, missing values, duplicates, and charts.
3. Try removing duplicates and filling missing values. Download the cleaned CSV.
4. Train a classification model with **`churn`** as the target.
5. Read the test-set metrics and compare them with the simple baseline.
6. Read the local explanation, then visit the saved run history.
7. Repeat with the regression sample and target **`price`**.
8. Try clustering with the clustering sample; clustering has no target column.

The sample data is synthetic. Its scores do not demonstrate performance on real customer or business data.

| Included file | Rows | Suggested task and target |
| --- | --- | --- |
| `sample_data/classification.csv` | 244, including 4 duplicate rows | Classification; target `churn` |
| `sample_data/regression.csv` | 250 | Regression; target `price` |
| `sample_data/clustering.csv` | 240 | Clustering; no target; start with 3 clusters |

## What you can do

| Feature | Implementation |
| --- | --- |
| Upload and validate CSV files | Flask and Pandas |
| Profile data and explore distributions | Pandas, NumPy, Matplotlib |
| Fill missing values, remove duplicates, or drop incomplete rows | Pandas |
| Download a cleaned CSV | Flask file response |
| Classification | Logistic regression in a Scikit-learn pipeline |
| Regression | Linear regression in a Scikit-learn pipeline |
| Clustering | K-means with numeric features |
| Evaluate models and compare stored runs | Scikit-learn metrics and SQLite |
| Explain findings without a key | Explicit rules in `insights.py` |
| Request a generative AI explanation | Optional OpenAI Responses API |

Cleaning for exploration and preprocessing for model training are separate. Each cleaning action starts from the **original dataset**, so cleaning choices are not cumulative. Model training also starts from the original. Supervised training removes rows without a target and identical selected feature-and-target rows, then splits the data before learning how to impute, encode, and scale features. This helps prevent test-set information from leaking into training. Keep the original upload for model evaluation: re-uploading data filled using the entire dataset would introduce preprocessing leakage.

“Original” means the parsed upload after basic normalization: trimming text whitespace and treating blank strings and infinite values as missing. CSV exports also prefix formula-leading text cells and headers with an apostrophe for safe opening in spreadsheet software; those exported text values may therefore look slightly different when re-uploaded.

## Optional generative AI

1. Copy `.env.example` to `.env` in the project folder.
2. Add your own `OPENAI_API_KEY` and configure `OPENAI_MODEL` using a model available to your API account.
3. Restart the app.
4. Select the AI explanation option and give consent in the form.

On Windows, copy the file with `Copy-Item .env.example .env`. On macOS/Linux, use `cp .env.example .env`.

The AI request sends a limited summary, column names, model metrics when available, and your question to OpenAI. It does **not** send raw CSV rows. Column names and a question can still contain sensitive information, so review what you enter. API usage may incur charges. The AI is asked to explain the computed results; it does not train the machine learning models. Check its explanation against the displayed statistics.

The default local explanation is rule-based, not generative AI. The app remains useful when no key is configured or an API request fails. Never commit `.env` or share your API key.

## Read the code in this order

| File or folder | What to learn |
| --- | --- |
| `generate_samples.py` | Create repeatable practice data with NumPy |
| `analysis.py` | Read, inspect, clean, and plot a DataFrame |
| `ml.py` | Prepare features, split data, train, and evaluate |
| `database.py` | Store and retrieve records using parameterized SQL |
| `insights.py` | Turn results into explanations; optionally call an API |
| `app.py` | Connect browser requests to the Python functions |
| `templates/` | Display Python results as HTML using Jinja |
| `static/` | Style the pages |
| `tests/` | Check important behavior automatically |

Then use [the learning guide](docs/LEARNING_GUIDE.md) to rebuild the project in seven stages. [The architecture guide](docs/ARCHITECTURE.md) explains how the pieces connect. [The interview guide](docs/INTERVIEW_GUIDE.md) helps you describe what you actually built and understand.

## Run the checks

Windows:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

macOS/Linux:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

After activating a virtual environment, the equivalent command is `python -m unittest discover -s tests -v`.

The test suite covers 33 automated checks. See [verification notes](docs/VERIFICATION.md) for coverage and what was not tested.

To recreate the practice CSVs, run `python generate_samples.py` with the virtual environment's Python. This replaces the generated examples in `sample_data/`.

## Scope and storage

- Accepts comma-separated, UTF-8 CSV files, up to 10 MB, 20,000 rows, and 100 columns. Headers must be unique, nonempty, and at most 100 characters long.
- Model training accepts at most 30 selected feature columns.
- Supervised training requires at least ten usable unique rows and uses a held-out test set. Classification requires enough examples of each class for the split. K-means uses numeric columns and reports clustering metrics, not classification accuracy.
- Saved runs support comparison on the same dataset. Check that task, target, usable rows, and evaluation split are comparable before judging scores. This is experiment history, not monitoring of a deployed model or automatic drift detection.
- Uploads, the SQLite database, and an automatically generated session secret are stored locally under `instance/`.
- Browser-session ownership separates local browser sessions; it is not a user login system. The session lasts up to 30 days and is refreshed with use. Clearing browser cookies or allowing the session to expire can make earlier datasets inaccessible from that browser.
- The server binds to `127.0.0.1` with debugging disabled. This is a local learning app, not a public multi-user service.
- This version does not save trained models or provide an endpoint for predicting on new rows.

`instance/`, `.env`, and `.venv/` are ignored by Git. Use synthetic or appropriately shareable data while learning.

## Common setup problems

| Problem | Try this |
| --- | --- |
| `py` or `python3` is not found | Install Python, reopen the terminal, and check the executable name. |
| `No module named flask` | Install requirements using the same virtual environment's Python that runs `app.py`. |
| Browser cannot open the page | Keep the app terminal running and use the exact local address above. |
| Port 5000 is occupied | Stop the other local server or change the port in the final `app.run(...)` line in `app.py`. |
| CSV is rejected | Export as UTF-8 CSV, check the header row, and reduce its size if it exceeds the limits. |
| Model cannot train | Read the validation message; check target values, row counts, selected features, and missing data. |
| AI explanation fails | Verify your API account, configured model, key, and network; local explanations still work. |

## Official learning references

- [Flask documentation](https://flask.palletsprojects.com/en/stable/) for routes, templates, sessions, and testing.
- [Pandas getting started](https://pandas.pydata.org/docs/getting_started/index.html) for DataFrames and CSV operations.
- [Scikit-learn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html) for consistent preprocessing and preventing data leakage.
- [OpenAI Python library reference](https://developers.openai.com/api/reference/python) for the optional API integration.

This project's `requirements.txt` pins its dependency versions; online documentation may describe newer versions.
