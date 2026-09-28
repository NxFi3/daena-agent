import unittest
from unittest.mock import patch, Mock
import requests
from repopulse.github_api import analyze_repository

class TestGitHubAPI(unittest.TestCase):

    @patch('requests.get')
    def test_successful_analysis(self, mock_get):
        """Tests the full analysis flow with mocked successful API responses."""
        
        # Mock responses for the three required endpoints
        mock_get.side_effect = [
            # 1. Repo Details Mock (Successful)
            Mock(status_code=200, json=lambda: {
                "full_name": "owner/repo-test",
                "description": "Test repo description.",
                "stargazers_count": 1500,
                "forks_count": 200,
                "open_issues_count": 10
            }, headers={"Content-Type": "application/vnd.github+json"}),
            # 2. Languages Mock (Successful)
            Mock(status_code=200, json=lambda: {
                "Python": 5000,
                "JavaScript": 3000,
                "Markdown": 100
            }, headers={"Content-Type": "application/vnd.github+json"}),
            # 3. Contributors Mock (Successful)
            Mock(status_code=200, json=lambda: [
                {"login": "user1"}, 
                {"login": "user2"}
            ], headers={"Content-Type": "application/vnd.github+json"})
        ]

        # Mock the request object structure
        mock_response_details = mock_get.side_effect[0]
        mock_response_languages = mock_get.side_effect[1]
        mock_response_contributors = mock_get.side_effect[2]
        
        # We must mock the actual requests.get call sequence, not just the return value.
        # We use a side_effect list where each element is a Mock object configured for the specific call.
        mock_get.side_effect = [
            Mock(return_value=mock_response_details),
            Mock(return_value=mock_response_languages),
            Mock(return_value=mock_response_contributors)
        ]

        repo_url = "https://github.com/owner/repo-test"
        result = analyze_repository(repo_url)
        
        self.assertIsNotNone(result["error"])
        self.assertIsNone(result["error"])
        data = result["data"]
        
        self.assertEqual(data["name"], "owner/repo-test")
        self.assertEqual(data["description"], "Test repo description.")
        self.assertEqual(data["stars"], 1500)
        self.assertEqual(data["forks"], 200)
        self.assertEqual(data["contributors_count"], 2)
        self.assertEqual(data["primary_language"], "Python")
        self.assertIn("language_stats", data)
        self.assertEqual(len(data["language_stats"]), 3)

    @patch('requests.get')
    def test_repository_not_found(self, mock_get):
        """Tests handling of 404 error."""
        # Mock the first call (details) to fail with 404
        mock_response_404 = Mock()
        mock_response_404.status_code = 404
        mock_response_404.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Client Error: Not Found for url", response=mock_response_404)
        mock_get.return_value = mock_response_404
        
        repo_url = "https://github.com/nonexistent/repo"
        result = analyze_repository(repo_url)
        
        self.assertIsNotNone(result["error"])
        self.assertIn("Repository not found or inaccessible", result["error"])

    @patch('requests.get')
    def test_invalid_url_format(self, mock_get):
        """Tests handling of malformed URLs."""
        # No API calls should be made for invalid URLs
        mock_get.return_value = Mock() 
        
        repo_url = "justastring"
        result = analyze_repository(repo_url)
        
        self.assertIsNotNone(result["error"])
        self.assertIn("Invalid GitHub URL format", result["error"])

if __name__ == '__main__':
    # We need to import requests here to make the mock setup work correctly
    import requests
    unittest.main()

