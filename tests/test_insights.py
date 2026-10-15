"""Mock the API boundary: no real API keys, network calls, or paid requests."""

import json
import unittest
from unittest.mock import patch

import pandas as pd

from analysis import profile_data
from insights import ai_explanation, local_explanation


class InsightTests(unittest.TestCase):
    def setUp(self):
        self.profile = profile_data(pd.DataFrame({"age": [21, 31, 41], "name": ["Private Alice", "Private Bob", "Private Charlie"]}))

    def test_local_summary_is_labeled_honestly(self):
        result = local_explanation(self.profile)
        self.assertEqual(result["mode"], "local")
        self.assertIn("3 rows", result["text"])

    @patch.dict("os.environ", {"OPENAI_API_KEY": "", "OPENAI_MODEL": ""})
    def test_missing_configuration_falls_back(self):
        result = ai_explanation(self.profile)
        self.assertEqual(result["mode"], "local")
        self.assertIn("not configured", result["warning"])

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-fake-key", "OPENAI_MODEL": "test-model"})
    def test_api_payload_has_aggregates_but_no_raw_rows(self):
        with patch("openai.OpenAI") as client_class:
            client = client_class.return_value.__enter__.return_value
            client.responses.create.return_value.output_text = "A short explanation."
            result = ai_explanation(self.profile, question="What is missing?")
            arguments = client.responses.create.call_args.kwargs
        self.assertEqual(result["mode"], "ai")
        self.assertFalse(arguments["store"])
        self.assertNotIn("Private Alice", arguments["input"])
        self.assertNotIn("test-fake-key", arguments["input"])
        payload = json.loads(arguments["input"])
        self.assertEqual(payload["question"], "What is missing?")
        self.assertEqual(payload["working_dataset_summary"]["rows"], 3)

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-fake-key", "OPENAI_MODEL": "test-model"})
    def test_provider_failure_has_safe_local_fallback(self):
        from openai import OpenAIError
        with patch("openai.OpenAI") as client_class:
            client = client_class.return_value.__enter__.return_value
            client.responses.create.side_effect = OpenAIError("private provider details")
            result = ai_explanation(self.profile)
        self.assertEqual(result["mode"], "local")
        self.assertNotIn("private provider details", result["warning"])


if __name__ == "__main__":
    unittest.main()
