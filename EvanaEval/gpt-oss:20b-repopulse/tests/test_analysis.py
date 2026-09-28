"""Unit tests for :func:`repopulse.utils.analyze_repository`.

These tests patch the lower‑level API helpers so that no live network
requests are performed.
"""

import unittest
from unittest.mock import patch

from gpt-oss:20b-repopulse.utils import analyze_repository


class TestAnalyzeRepository(unittest.TestCase):
    @patch("repopulse.utils.get_repo_info")
    @patch("repopulse.utils.get_repo_languages")
    @patch("repopulse.utils.get_repo_contributors")
    def test_basic_analysis(self, mock_contrib, mock_lang, mock_info):
        # Arrange
        mock_info.return_value = {
            "name": "test-repo",
            "description": "A test repo",
            "stargazers_count": 42,
            "forks_count": 7,
        }
        mock_lang.return_value = {"Python": 1000, "HTML": 200}
        mock_contrib.return_value = [{"login": "alice"}, {"login": "bob"}]

        # Act
        result = analyze_repository("owner", "repo")

        # Assert
        self.assertEqual(result["name"], "test-repo")
        self.assertEqual(result["description"], "A test repo")
        self.assertEqual(result["stargazers_count"], 42)
        self.assertEqual(result["forks_count"], 7)
        self.assertEqual(result["language"], "Python")
        self.assertEqual(result["languages"], {"Python": 1000, "HTML": 200})
        self.assertEqual(result["contributors_count"], 2)


if __name__ == "__main__":
    unittest.main()

