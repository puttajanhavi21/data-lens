"""Integration checks use temporary folders and never contact an AI service."""

from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app import create_app


class AppTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.app = create_app({"TESTING": True, "SECRET_KEY": "test-only-secret",
                               "DATA_DIR": Path(self.folder.name)})
        self.client = self.app.test_client()
        self.client.get("/")
        with self.client.session_transaction() as state:
            self.token = state["csrf_token"]

    def tearDown(self):
        self.folder.cleanup()

    def post(self, path, data=None, **kwargs):
        return self.client.post(path, data={"csrf_token": self.token, **(data or {})}, **kwargs)

    def test_missing_session_and_csrf_are_rejected(self):
        self.assertEqual(self.client.get("/dashboard").status_code, 404)
        self.assertEqual(self.client.post("/sample/classification").status_code, 400)
        self.assertEqual(self.post("/sample/unknown").status_code, 404)

    def test_upload_clean_download_reset(self):
        csv = b"amount,group\n1,a\n2,b\n2,b\n,c\n"
        response = self.post("/upload", {"file": (BytesIO(csv), "demo.csv")})
        self.assertEqual(response.status_code, 302)
        response = self.client.get("/dashboard")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"demo.csv", response.data)
        self.post("/clean", {"missing_strategy": "fill", "remove_duplicates": "on"})
        self.assertIn(b"1.5,c", self.client.get("/download").data)
        self.post("/reset")
        self.assertIn(b",c", self.client.get("/download").data)
        response = self.post("/clean", {"missing_strategy": "unknown"}, follow_redirects=True)
        self.assertIn(b"Choose leave, fill, or drop", response.data)

    def test_invalid_upload_does_not_create_dataset(self):
        response = self.post("/upload", {"file": (BytesIO(b"a,a\n1,2"), "bad.csv")}, follow_redirects=True)
        self.assertIn(b"Column names must be unique", response.data)
        with self.client.session_transaction() as state:
            self.assertNotIn("dataset_id", state)

    def test_untrusted_csv_text_is_escaped(self):
        self.post("/upload", {"file": (BytesIO(b'name\n<script>alert(1)</script>'), "demo.csv")})
        response = self.client.get("/dashboard")
        self.assertNotIn(b"<script>alert(1)</script>", response.data)
        self.assertIn(b"&lt;script&gt;", response.data)

    def test_all_model_workflows_render_and_persist(self):
        cases = [
            ("classification", "churn", ["tenure_months", "monthly_charge", "support_tickets", "contract", "churn"]),
            ("regression", "price", ["area_sqft", "bedrooms", "age_years", "neighborhood", "price"]),
            ("clustering", "", ["annual_income_thousands", "spending_score", "monthly_visits"]),
        ]
        for task, target, features in cases:
            with self.subTest(task=task):
                self.post(f"/sample/{task}")
                response = self.post("/train", {"task": task, "target": target, "features": features,
                                                "test_size": "0.2", "n_clusters": "3"}, follow_redirects=True)
                self.assertEqual(response.status_code, 200)
                self.assertIn(b"Model evaluated", response.data)
        response = self.client.get("/history")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Logistic regression", response.data)
        self.assertIn(b"Linear regression", response.data)
        self.assertIn(b"K-means", response.data)

    def test_browser_sessions_cannot_read_each_others_data(self):
        self.post("/sample/classification")
        with self.client.session_transaction() as state:
            dataset_id = state["dataset_id"]
        other = self.app.test_client()
        other.get("/")
        with other.session_transaction() as state:
            other_token = state["csrf_token"]
            state["dataset_id"] = dataset_id
        self.assertEqual(other.get("/dashboard").status_code, 404)
        self.assertEqual(other.get("/download").status_code, 404)
        self.assertEqual(other.post(f"/select/{dataset_id}", data={"csrf_token": other_token}).status_code, 404)

    def test_eda_cleaning_does_not_supply_model_training_data(self):
        self.post("/sample/classification")
        self.post("/clean", {"missing_strategy": "fill"})
        with patch("app.train_model", side_effect=ValueError("Training intercepted")) as mock_train:
            self.post("/train", {"task": "classification", "target": "churn", "features": ["monthly_charge"]})
        self.assertGreater(mock_train.call_args.args[0].isna().sum().sum(), 0)

    def test_ai_requires_consent_and_local_mode_stays_local(self):
        self.post("/sample/classification")
        with patch("app.ai_explanation") as ai:
            self.post("/explain", {"mode": "ai"})
            self.post("/explain", {"mode": "local"})
            ai.assert_not_called()
        with patch("app.ai_explanation", return_value={"mode": "ai", "text": "Example explanation", "warning": None}) as ai:
            self.post("/explain", {"mode": "ai", "consent": "on", "question": "Explain missing values"})
            ai.assert_called_once()
        self.assertIn(b"Example explanation", self.client.get("/dashboard").data)


if __name__ == "__main__":
    unittest.main()
