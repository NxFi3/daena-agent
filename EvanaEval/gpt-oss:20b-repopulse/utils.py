"""Utility functions for interacting with the GitHub REST API.

All network calls use :mod:`urllib.request` and return plain Python
objects.  Errors are raised as :class:`RuntimeError` with a clear message.
"""

from __future__ import annotations

import json
import re
import urllib.request
from urllib.error import HTTPError, URLError
from typing import Dict, List, Tuple

GITHUB_API_BASE = "https://api.github.com"

# Regular expression to parse GitHub repository URLs.
_RE_GITHUB_URL = re.compile(
    r"https?://(?:www\.)?github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)"
)


def parse_github_url(url: str) -> Tuple[str, str]:
    """Parse a GitHub repository URL.

    Parameters
    ----------
    url:
        The URL to parse.

    Returns
    -------
    tuple
        ``(owner, repo)``.

    Raises
    ------
    ValueError
        If the URL is not a valid public GitHub repository URL.
    """

    match = _RE_GITHUB_URL.match(url.strip())
    if not match:
        raise ValueError(f"Not a valid GitHub repository URL: {url}")
    return match.group("owner"), match.group("repo")


def _github_get(path: str) -> Dict:
    """Internal helper to perform a GET request to the GitHub API.

    Parameters
    ----------
    path:
        API path, e.g. ``/repos/octocat/Hello-World``.

    Returns
    -------
    dict
        Parsed JSON response.
    """

    url = f"{GITHUB_API_BASE}{path}"
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read().decode("utf-8")
            return json.loads(data)
    except HTTPError as e:
        if e.code == 404:
            raise RuntimeError("Repository not found")
        raise RuntimeError(f"GitHub API error {e.code}: {e.reason}")
    except URLError as e:
        raise RuntimeError(f"Network error: {e.reason}")


def get_repo_info(owner: str, repo: str) -> Dict:
    """Return basic repository information.

    The returned dictionary contains the keys used by the front‑end:
    ``name``, ``description``, ``stars``, ``forks``.
    """

    data = _github_get(f"/repos/{owner}/{repo}")
    return {
        "name": data.get("name"),
        "description": data.get("description"),
        "stargazers_count": data.get("stargazers_count"),
        "forks_count": data.get("forks_count"),
    }


def get_repo_languages(owner: str, repo: str) -> Dict[str, int]:
    """Return a mapping of language to byte count.

    The GitHub API returns a JSON object where keys are language names and
    values are the number of bytes of code written in that language.
    """

    return _github_get(f"/repos/{owner}/{repo}/languages")


def get_repo_contributors(owner: str, repo: str) -> List[Dict]:
    """Return a list of contributor objects.

    Only the first page (up to 100 contributors) is retrieved.  The caller
    can count the list length to obtain the number of contributors.
    """

    # GitHub paginates contributors; we request the maximum per page.
    data = _github_get(f"/repos/{owner}/{repo}/contributors?per_page=100")
    if not isinstance(data, list):
        raise RuntimeError("Malformed contributors response")
    return data


def analyze_repository(owner: str, repo: str) -> Dict:
    """Return a combined analysis of a repository.

    The returned dictionary contains the following keys:
    ``name``, ``description``, ``stargazers_count``, ``forks_count``,
    ``language`` (primary language), ``languages`` (mapping of language to
    byte count), and ``contributors_count``.
    """

    info = get_repo_info(owner, repo)
    languages = get_repo_languages(owner, repo)
    contributors = get_repo_contributors(owner, repo)

    # Determine primary language as the one with the most bytes.
    primary_lang = None
    if languages:
        primary_lang = max(languages, key=languages.get)

    return {
        "name": info.get("name"),
        "description": info.get("description"),
        "stargazers_count": info.get("stargazers_count"),
        "forks_count": info.get("forks_count"),
        "language": primary_lang,
        "languages": languages,
        "contributors_count": len(contributors),
    }

