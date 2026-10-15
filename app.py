"""Start here: Flask connects a browser request to our small Python modules.

Run: python app.py
Open: http://127.0.0.1:5000
"""

from datetime import timedelta
from io import BytesIO
import json
from pathlib import Path
import secrets
from uuid import uuid4

import os
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, abort, flash, g, redirect, render_template, request, send_file, session, url_for
from werkzeug.exceptions import HTTPException
from werkzeug.utils import secure_filename

from analysis import MAX_BYTES, clean_data, csv_for_download, load_csv, make_charts, profile_data
import database
from insights import ai_configured, ai_explanation, local_explanation
from ml import train_model

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def create_app(test_config=None):
    """An app factory lets tests use a temporary folder instead of your data."""
    app = Flask(__name__)
    app.config.update(
        DATA_DIR=BASE_DIR / "instance",
        MAX_CONTENT_LENGTH=MAX_BYTES + 1024 * 1024,  # Leave room for form fields.
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        PERMANENT_SESSION_LIFETIME=timedelta(days=30),
    )
    if test_config:
        app.config.update(test_config)

    data_dir = Path(app.config["DATA_DIR"])
    data_dir.mkdir(parents=True, exist_ok=True)
    upload_dir = data_dir / "uploads"
    upload_dir.mkdir(exist_ok=True)
    database_path = data_dir / "data_lens.db"
    database.init_db(database_path)

    # A stable key keeps your browser's history accessible after restarting.
    secret_file = data_dir / "secret.key"
    if not app.config.get("SECRET_KEY"):
        if os.getenv("SECRET_KEY"):
            app.config["SECRET_KEY"] = os.environ["SECRET_KEY"]
        else:
            if not secret_file.exists():
                secret_file.write_text(secrets.token_hex(32), encoding="utf-8")
            app.config["SECRET_KEY"] = secret_file.read_text(encoding="utf-8").strip()

    @app.before_request
    def prepare_request():
        # This identifies a local browser session; it is not a login system.
        if "owner" not in session:
            session["owner"] = uuid4().hex
            session.permanent = True
        if "csrf_token" not in session:
            session["csrf_token"] = secrets.token_urlsafe(32)
        if request.method == "POST":
            token = request.form.get("csrf_token", "")
            if not secrets.compare_digest(token, session["csrf_token"]):
                abort(400, description="The form expired. Reload the page and try again.")
        g.dataset = database.get_dataset(database_path, session.get("dataset_id", ""), session["owner"])

    @app.context_processor
    def template_helpers():
        return {"csrf_token": lambda: session["csrf_token"], "current_dataset": g.get("dataset")}

    def require_dataset():
        if not g.dataset:
            abort(404, description="Load a CSV or a sample dataset first.")
        return g.dataset

    def dataset_path(original=False):
        dataset = require_dataset()
        suffix = "original" if original else "working"
        return upload_dir / f"{dataset['id']}_{suffix}.csv"

    def read_dataset(original=False):
        path = dataset_path(original)
        if not path.exists():
            abort(404, description="This dataset file is missing. Upload it again.")
        return pd.read_csv(path)

    def latest_run():
        runs = database.list_runs(database_path, session["owner"], require_dataset()["id"])
        return runs[0]["result"] if runs else None

    def store_dataset(frame, name):
        dataset_id = uuid4().hex
        # Generated IDs, never user-supplied paths, name the files on disk.
        for suffix in ("original", "working"):
            frame.to_csv(upload_dir / f"{dataset_id}_{suffix}.csv", index=False)
        database.add_dataset(database_path, dataset_id, session["owner"], name, len(frame), len(frame.columns))
        session["dataset_id"] = dataset_id

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.post("/upload")
    def upload():
        file = request.files.get("file")
        if not file or not file.filename or not file.filename.lower().endswith(".csv"):
            flash("Choose a .csv file to get started.", "error")
            return redirect(url_for("index"))
        try:
            frame = load_csv(file.read(MAX_BYTES + 1))
            name = secure_filename(file.filename)[:120] or "dataset.csv"
            store_dataset(frame, name)
        except ValueError as error:
            flash(str(error), "error")
            return redirect(url_for("index"))
        flash("Your dataset is ready to explore.", "success")
        return redirect(url_for("dashboard"))

    @app.post("/sample/<name>")
    def sample(name):
        if name not in {"classification", "regression", "clustering"}:
            abort(404)
        frame = load_csv((BASE_DIR / "sample_data" / f"{name}.csv").read_bytes())
        store_dataset(frame, f"sample_{name}.csv")
        flash("Synthetic sample loaded. It is designed for learning.", "success")
        return redirect(url_for("dashboard"))

    @app.get("/dashboard")
    def dashboard():
        dataset = require_dataset()
        frame = read_dataset()
        profile = profile_data(frame)
        numeric_columns = frame.select_dtypes(include="number").columns.tolist()
        selected = request.args.get("column", "")
        if selected not in numeric_columns:
            selected = numeric_columns[0] if numeric_columns else None
        run = latest_run()
        insight = json.loads(dataset["insight_json"]) if dataset["insight_json"] else local_explanation(profile, run)

        # Model features come from the original, even after EDA cleaning.
        original = read_dataset(original=True)
        feature_options = []
        for name in original.columns:
            numeric = pd.api.types.is_numeric_dtype(original[name])
            usable = original[name].notna().any()
            likely_id = name.lower() == "id" or name.lower().endswith("_id")
            selected_by_default = usable and not likely_id and (numeric or original[name].nunique() <= 20)
            feature_options.append({"name": name, "selected": selected_by_default,
                                    "kind": "numeric" if numeric else "categorical"})
        return render_template(
            "dashboard.html", dataset=dataset, profile=profile,
            preview=frame.head(10).to_html(index=False, classes="data-table", border=0, escape=True, na_rep="—"),
            numeric_columns=numeric_columns, selected_column=selected,
            charts=make_charts(frame, selected), insight=insight, last_run=run,
            feature_options=feature_options, cleaning_applied=bool(dataset["cleaning_note"]),
            cleaning_note=dataset["cleaning_note"], ai_configured=ai_configured(),
        )

    @app.post("/clean")
    def clean():
        dataset = require_dataset()
        strategy = request.form.get("missing_strategy", "leave")
        remove_duplicates = request.form.get("remove_duplicates") == "on"
        try:
            # Start fresh every time, so the controls describe the whole recipe.
            cleaned = clean_data(read_dataset(original=True), strategy, remove_duplicates)
            cleaned.to_csv(dataset_path(), index=False)
            note = f"Missing values: {strategy}. Remove duplicates: {'yes' if remove_duplicates else 'no'}. Started from the original."
            database.update_cleaning(database_path, dataset["id"], session["owner"], note)
            flash("Working copy updated. Charts and downloads now use this copy.", "success")
        except ValueError as error:
            flash(str(error), "error")
        return redirect(url_for("dashboard", _anchor="prepare"))

    @app.post("/reset")
    def reset():
        dataset = require_dataset()
        dataset_path().write_bytes(dataset_path(original=True).read_bytes())
        database.update_cleaning(database_path, dataset["id"], session["owner"], "")
        flash("Original data restored to the working copy.", "success")
        return redirect(url_for("dashboard"))

    @app.get("/download")
    def download():
        frame = read_dataset()
        return send_file(BytesIO(csv_for_download(frame)), mimetype="text/csv", as_attachment=True,
                         download_name="data_lens_working.csv")

    @app.post("/train")
    def train():
        dataset = require_dataset()
        try:
            features = request.form.getlist("features")
            if not features:
                raise ValueError("Select at least one input feature.")
            task = request.form.get("task", "classification")
            # Ignore controls that do not apply to the selected task.
            test_size = float(request.form.get("test_size", "0.2")) if task != "clustering" else 0.2
            clusters = int(request.form.get("n_clusters", "3")) if task == "clustering" else 3
            result = train_model(read_dataset(original=True), task,
                                 target=request.form.get("target") or None, features=features,
                                 test_size=test_size, n_clusters=clusters)
            database.save_run(database_path, dataset["id"], result)
            flash("Model evaluated. Results were saved to run history.", "success")
        except ValueError as error:
            flash(str(error), "error")
        return redirect(url_for("dashboard", _anchor="model"))

    @app.post("/explain")
    def explain():
        dataset = require_dataset()
        mode = request.form.get("mode", "local")
        if mode not in {"local", "ai"}:
            abort(400, description="Choose local or AI explanations.")
        if mode == "ai" and request.form.get("consent") != "on":
            flash("Tick the summary-sharing checkbox before requesting AI explanations.", "error")
            return redirect(url_for("dashboard", _anchor="insights"))
        profile = profile_data(read_dataset())
        run = latest_run()
        if mode == "ai":
            insight = ai_explanation(profile, run, request.form.get("question", "")[:500])
        else:
            insight = local_explanation(profile, run)
        database.save_insight(database_path, dataset["id"], session["owner"], insight)
        return redirect(url_for("dashboard", _anchor="insights"))

    @app.get("/history")
    def history():
        runs = database.list_runs(database_path, session["owner"])
        return render_template("history.html", runs=runs, chart=None)

    @app.post("/select/<dataset_id>")
    def select_dataset(dataset_id):
        if not database.get_dataset(database_path, dataset_id, session["owner"]):
            abort(404)
        session["dataset_id"] = dataset_id
        return redirect(url_for("dashboard"))

    @app.errorhandler(HTTPException)
    def http_error(error):
        message = "Choose a CSV smaller than 10 MB." if error.code == 413 else error.description
        return render_template("error.html", code=error.code, message=message), error.code

    @app.errorhandler(500)
    def unexpected_error(error):
        return render_template("error.html", code=500,
                               message="Something went wrong. Check the terminal for details and try a sample dataset."), 500

    return app


if __name__ == "__main__":
    create_app().run(host=os.getenv("HOST", "127.0.0.1"),
                     port=int(os.getenv("PORT", "5000")), debug=False)
