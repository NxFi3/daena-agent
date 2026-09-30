const express = require("express");
const fetch = require("node-fetch");
const cors = require("cors");
const path = require("path");

const app = express();
app.use(cors());

// Environment variable for GitHub token (optional)
const GITHUB_TOKEN = process.env.GITHUB_TOKEN;

// Helper to set auth header if token provided
const authHeaders = () => {
  if (GITHUB_TOKEN) {
    return { Authorization: `token ${GITHUB_TOKEN}` };
  }
  return {};
};

// Proxy endpoint to fetch repository data
// Support both /api/repo?owner=&repo= and /api/repo/:owner/:repo
app.get("/api/repo", async (req, res) => {
  const { owner, repo } = req.query;
  if (!owner || !repo) {
    return res.status(400).json({ error: "Missing owner or repo query parameters" });
  }
  try {
    const base = `https://api.github.com/repos/${owner}/${repo}`;
    const headers = authHeaders();

    // Fetch repository info
    const repoResp = await fetch(base, { headers });
    if (!repoResp.ok) {
      const text = await repoResp.text();
      if (repoResp.status === 403 && text.includes("API rate limit exceeded")) {
        return res.status(429).json({ error: "GitHub API rate limit exceeded. Please try again later or provide a personal access token." });
      }
      return res.status(repoResp.status).json({ error: text });
    }
    const repoInfo = await repoResp.json();

    // Recent commits (last 10)
    const commitsResp = await fetch(`${base}/commits?per_page=10`, { headers });
    const commits = commitsResp.ok ? await commitsResp.json() : [];

    // Open issues (first 10)
    const issuesResp = await fetch(`${base}/issues?state=open&per_page=10`, { headers });
    const issues = issuesResp.ok ? await issuesResp.json() : [];

    // Open pull requests (first 10)
    const pullsResp = await fetch(`${base}/pulls?state=open&per_page=10`, { headers });
    const pullRequests = pullsResp.ok ? await pullsResp.json() : [];

    // Languages
    const langsResp = await fetch(`${base}/languages`, { headers });
    const langs = langsResp.ok ? await langsResp.json() : {};
    const languages = Object.keys(langs);

    res.json({ repoInfo, commits, issues, pullRequests, stats: { languages } });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: e.message });
  }
});

// Serve static files from client build
const clientBuildPath = path.join(__dirname, "client", "dist");
app.use(express.static(clientBuildPath));

// Fallback to index.html for SPA routing
app.use((req, res) => {
  res.sendFile(path.join(clientBuildPath, "index.html"));
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server listening on port ${PORT}`);
});

