"""SQLite stores dataset metadata and evaluation results, not trained models.

The ? placeholders keep values separate from SQL instructions.
closing() ensures every short-lived connection is closed after use.
"""

from contextlib import closing
import json
import sqlite3


def connect(database_path):
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def init_db(database_path):
    with closing(connect(database_path)) as connection:
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS datasets (
                id TEXT PRIMARY KEY,
                owner TEXT NOT NULL,
                name TEXT NOT NULL,
                row_count INTEGER NOT NULL,
                column_count INTEGER NOT NULL,
                cleaning_note TEXT NOT NULL DEFAULT '',
                insight_json TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS model_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_id TEXT NOT NULL REFERENCES datasets(id),
                result_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
        """)
        connection.commit()


def add_dataset(database_path, dataset_id, owner, name, rows, columns):
    with closing(connect(database_path)) as connection:
        connection.execute(
            "INSERT INTO datasets (id, owner, name, row_count, column_count) VALUES (?, ?, ?, ?, ?)",
            (dataset_id, owner, name, rows, columns),
        )
        connection.commit()


def get_dataset(database_path, dataset_id, owner):
    with closing(connect(database_path)) as connection:
        row = connection.execute("SELECT * FROM datasets WHERE id = ? AND owner = ?", (dataset_id, owner)).fetchone()
        return dict(row) if row else None


def update_cleaning(database_path, dataset_id, owner, note):
    with closing(connect(database_path)) as connection:
        connection.execute(
            "UPDATE datasets SET cleaning_note = ?, insight_json = NULL WHERE id = ? AND owner = ?",
            (note, dataset_id, owner),
        )
        connection.commit()


def save_insight(database_path, dataset_id, owner, insight):
    with closing(connect(database_path)) as connection:
        connection.execute("UPDATE datasets SET insight_json = ? WHERE id = ? AND owner = ?",
                           (json.dumps(insight, allow_nan=False), dataset_id, owner))
        connection.commit()


def save_run(database_path, dataset_id, result):
    with closing(connect(database_path)) as connection:
        connection.execute("INSERT INTO model_runs (dataset_id, result_json) VALUES (?, ?)",
                           (dataset_id, json.dumps(result, allow_nan=False)))
        # A previous explanation may describe an older model; discard it.
        connection.execute("UPDATE datasets SET insight_json = NULL WHERE id = ?", (dataset_id,))
        connection.commit()


def list_runs(database_path, owner, dataset_id=None):
    query = """
        SELECT model_runs.id, model_runs.created_at, datasets.name AS dataset_name,
               model_runs.dataset_id, model_runs.result_json
        FROM model_runs JOIN datasets ON datasets.id = model_runs.dataset_id
        WHERE datasets.owner = ?
    """
    parameters = [owner]
    if dataset_id:
        query += " AND datasets.id = ?"
        parameters.append(dataset_id)
    query += " ORDER BY model_runs.id DESC LIMIT 100"
    with closing(connect(database_path)) as connection:
        rows = connection.execute(query, parameters).fetchall()
    runs = []
    for row in rows:
        run = dict(row)
        run["result"] = json.loads(run.pop("result_json"))
        runs.append(run)
    return runs
