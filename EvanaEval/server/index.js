const express = require('express');
const cors = require('cors');
const fetch = require('node-fetch');
const dotenv = require('dotenv');

dotenv.config();

const app = express();
app.use(cors());

const GITHUB_API = 'https://api.github.com';
const token = process.env.GITHUB_TOKEN || '';
const headers = token
  ? { Authorization: `token ${token}` }
  : {};

/**
 * Helper to fetch from GitHub API with error handling.
 */
async function githubFetch(url) {
  const res = await fetch(url, { headers });
  if (!res.ok) {
    const error = new Error(`GitHub API error: ${res.status}`);
    error.status = res.status;
    error.body = await res.json();
    throw error;
  }
  return res.json();
}

/**
 * GET /api/repo?owner=facebook&repo=react
 * Returns repository info, recent commits, open issues, pull requests, and stats.
 */
app.get('/api/repo', async (req, res) => {
  const { owner, repo } = req.query;
  if (!owner || !repo) {
    return res.status(400).json({ error: 'Missing owner or repo query parameters' });
  }
  try {
    const [repoInfo, commits, issues, pulls] = await Promise.all([
      githubFetch(`${GITHUB_API}/repos/${owner}/${repo}`),
      githubFetch(`${GITHUB_API}/repos/${owner}/${repo}/commits?per_page=10`),
      githubFetch(`${GITHUB_API}/repos/${owner}/${repo}/issues?state=open&per_page=10`),
      githubFetch(`${GITHUB_API}/repos/${owner}/${repo}/pulls?state=open&per_page=10`),
    ]);

    // Filter out pull requests from issues list (GitHub returns PRs as issues)
    const filteredIssues = issues.filter((i) => !i.pull_request);

    res.json({ repoInfo, commits, issues: filteredIssues, pulls });
  } catch (err) {
    console.error(err);
    const status = err.status || 500;
    const message = err.body?.message || err.message;
    res.status(status).json({ error: message });
  }
});

const port = process.env.PORT || 4000;
app.listen(port, () => {
  console.log(`GitHub proxy server listening on port ${port}`);
});

