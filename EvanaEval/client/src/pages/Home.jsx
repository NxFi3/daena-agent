import { useState } from "react";
import { useNavigate } from "react-router-dom";

function Home() {
  const [owner, setOwner] = useState("facebook");
  const [repo, setRepo] = useState("react");
  const navigate = useNavigate();

  const handleSubmit = (e) => {
    e.preventDefault();
    navigate(`/repo/${owner}/${repo}`);
  };

  return (
    <div style={{ padding: "1rem" }}>
      <h1>GitHub Repository Dashboard</h1>
      <form onSubmit={handleSubmit} style={{ marginTop: "1rem" }}>
        <label>
          Owner:
          <input
            type="text"
            value={owner}
            onChange={(e) => setOwner(e.target.value)}
            required
          />
        </label>
        <label style={{ marginLeft: "1rem" }}>
          Repo:
          <input
            type="text"
            value={repo}
            onChange={(e) => setRepo(e.target.value)}
            required
          />
        </label>
        <button type="submit" style={{ marginLeft: "1rem" }}>
          View
        </button>
      </form>
    </div>
  );
}

export default Home;

