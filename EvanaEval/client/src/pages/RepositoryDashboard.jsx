import { useParams } from "react-router-dom";
import { useEffect, useState } from "react";

function RepositoryDashboard() {
  const { owner, repo } = useParams();
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    fetch(`/api/repo/${owner}/${repo}`)
      .then((res) => {
        if (!res.ok) {
          throw new Error(`HTTP ${res.status}`);
        }
        return res.json();
      })
      .then((json) => {
        setData(json);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, [owner, repo]);

  if (loading) return <p>Loading...</p>;
  if (error) return <p style={{ color: "red" }}>Error: {error}</p>;
  if (!data) return <p>No data.</p>;

  const { repoInfo, commits, issues, pullRequests, stats } = data;

  return (
    <div style={{ padding: "1rem" }}>
      <h2>{repoInfo.full_name}</h2>
      <p>{repoInfo.description}</p>
      <p>
        ⭐ {repoInfo.stargazers_count} | 🍴 {repoInfo.forks_count} | 🐛 {repoInfo.open_issues_count}
      </p>
      <h3>Recent Commits</h3>
      {commits.length ? (
        <ul>
          {commits.map((c) => (
            <li key={c.sha}>
              <a href={c.html_url} target="_blank" rel="noopener noreferrer">
                {c.commit.message.split("\n")[0]}
              </a>{" "}by {c.commit.author.name}
            </li>
          ))}
        </ul>
      ) : (
        <p>No recent commits found.</p>
      )}
      <h3>Open Issues</h3>
      {issues.length ? (
        <ul>
          {issues.map((i) => (
            <li key={i.id}>
              <a href={i.html_url} target="_blank" rel="noopener noreferrer">
                #{i.number} {i.title}
              </a>
            </li>
          ))}
        </ul>
      ) : (
        <p>No open issues.</p>
      )}
      <h3>Open Pull Requests</h3>
      {pullRequests.length ? (
        <ul>
          {pullRequests.map((p) => (
            <li key={p.id}>
              <a href={p.html_url} target="_blank" rel="noopener noreferrer">
                #{p.number} {p.title}
              </a>
            </li>
          ))}
        </ul>
      ) : (
        <p>No open pull requests.</p>
      )}
      <h3>Stats</h3>
      <p>Languages: {stats.languages.join(", ")}</p>
    </div>
  );
}

export default RepositoryDashboard;

