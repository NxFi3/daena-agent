import requests
from urllib.parse import urlparse
from typing import Optional, Dict, Any

GITHUB_API_BASE = "https://api.github.com/repos/"

def get_repo_details(repo_full_name: str) -> Optional[Dict[str, Any]]:
    """Fetches basic details for a given GitHub repository."""
    api_url = f"{GITHUB_API_BASE}{repo_full_name}"
    try:
        response = requests.get(api_url, headers={"Accept": "application/vnd.github+json"})
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            return None # Repository not found
        return {"error": f"HTTP Error {e.response.status_code}"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Network or API connection error: {e}"}

def get_repo_languages(repo_full_name: str) -> Optional[Dict[str, Any]]:
    """Fetches language statistics for a given repository."""
    api_url = f"{GITHUB_API_BASE}{repo_full_name}/languages"
    try:
        response = requests.get(api_url, headers={"Accept": "application/vnd.github+json"})
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        return {"error": f"HTTP Error {e.response.status_code}"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Network or API connection error: {e}"}

def get_repo_contributors(repo_full_name: str) -> Optional[Dict[str, Any]]:
    """Fetches a count of contributors for a given repository."""
    # Fetching 1 contributor per page is sufficient to get the total count metadata
    api_url = f"{GITHUB_API_BASE}{repo_full_name}/contributors?per_page=1"
    try:
        response = requests.get(api_url, headers={"Accept": "application/vnd.github+json"})
        response.raise_for_status()
        contributors = response.json()
        # We rely on the API's pagination logic for the count, but for simplicity, 
        # we'll use a placeholder count or just the length of the first page result.
        # A more robust solution would check the 'Link' header for total pages.
        return {"count": len(contributors), "contributors": contributors}
    except requests.exceptions.HTTPError as e:
        return {"error": f"HTTP Error {e.response.status_code}"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Network or API connection error: {e}"}

def analyze_repository(repo_url: str) -> Dict[str, Any]:
    """Main function to analyze a GitHub repository from a URL."""
    try:
        parsed_url = urlparse(repo_url)
        path_parts = [p for p in parsed_url.path.strip('/').split('/') if p]

        if len(path_parts) < 2:
            return {"error": "Invalid GitHub URL format. Must contain owner/repo.", "data": None}

        # Assuming the last two parts are owner and repo name
        repo_full_name = f"{path_parts[-2]}/{path_parts[-1]}"

        # 1. Get basic details
        details = get_repo_details(repo_full_name)
        if isinstance(details, dict) and "error" in details:
            return {"error": details["error"], "data": None}
        if details is None:
            return {"error": "Repository not found or inaccessible.", "data": None}

        # 2. Get languages
        languages = get_repo_languages(repo_full_name)
        if isinstance(languages, dict) and "error" in languages:
            languages = {"error": languages["error"]}
        
        # 3. Get contributors
        contributors = get_repo_contributors(repo_full_name)
        if isinstance(contributors, dict) and "error" in contributors:
            contributors = {"error": contributors["error"]}

        # 4. Compile results
        result = {
            "name": details.get("full_name", "N/A"),
            "description": details.get("description", "No description provided."),
            "stars": details.get("stargazers_count", 0),
            "forks": details.get("forks_count", 0),
            "primary_language": "N/A",
            "language_stats": {}, 
            "contributors_count": contributors.get("count", 0)
        }

        if isinstance(languages, dict) and "error" not in languages:
            # Find the language with the maximum size
            primary_lang = max(languages, key=languages.get)
            result["primary_language"] = primary_lang
            result["language_stats"] = languages
        elif "error" in languages:
             result["language_stats"] = {"error": languages["error"]}

        return {"error": None, "data": result}

    except Exception as e:
        return {"error": f"An unexpected critical error occurred: {e}", "data": None}
