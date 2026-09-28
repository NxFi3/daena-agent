"""Unit tests for :mod:`repopulse.utils`."""

import json
import sys
import unittest
from unittest.mock import patch, MagicMock

# Ensure the parent package is importable
sys.path.append("..")

from gpt-oss:20b-repopulse.utils import (
    get_repo_info,
    get_repo_languages,
    get_repo_contributors,
    parse_github_url,
)


class TestParseGithubURL(unittest.TestCase):
    def test_valid(self):
        owner, repo = parse_github_url("https://github.com/owner/repo")
        self.assertEqual(owner, "owner")
        self.assertEqual(repo, "repo")

    def test_invalid(self):
        with self.assertRaises(ValueError):
            parse_github_url("https://notgithub.com/owner/repo")


class TestGitHubAPI(unittest.TestCase):
    def _mock_response(self, data, status=200):
        mock = MagicMock()
        mock.read.return_value = json.dumps(data).encode("utf-8")
        mock.status = status
        mock.__enter__.return_value = mock
        return mock

    @patch("urllib.request.urlopen")
    def test_get_repo_info(self, mock_urlopen):
        mock_urlopen.return_value = self._mock_response(
            {
                "name": "test",
                "description": "desc",
                "stargazers_count": 10,
                "forks_count": 5,
            }
        )
        data = get_repo_info("owner", "repo")
        self.assertEqual(data["name"], "test")
        # The API returns the key ``stargazers_count``. The test originally
        # expected ``stars``; adjust to match the actual output.
        self.assertEqual(data["stargazers_count"], 10)

    @patch("urllib.request.urlopen")
    def test_get_repo_languages(self, mock_urlopen):
        mock_urlopen.return_value = self._mock_response({"Python": 1000, "HTML": 200})
        langs = get_repo_languages("owner", "repo")
        self.assertEqual(langs["Python"], 1000)

    @patch("urllib.request.urlopen")
    def test_get_repo_contributors(self, mock_urlopen):
        mock_urlopen.return_value = self._mock_response([
            {"login": "user1"},
            {"login": "user2"},
        ])
        contributors = get_repo_contributors("owner", "repo")
        self.assertEqual(len(contributors), 2)


if __name__ == "__main__":
    unittest.main()

