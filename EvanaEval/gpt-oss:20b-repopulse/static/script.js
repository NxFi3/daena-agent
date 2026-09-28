document.getElementById("analyzeBtn").addEventListener("click", () => {
    const url = document.getElementById("repoUrl").value.trim();
    const resultsDiv = document.getElementById("results");
    resultsDiv.innerHTML = "";
    if (!url) {
        resultsDiv.innerHTML = '<p class="error">Please enter a URL.</p>';
        return;
    }
    resultsDiv.innerHTML = "<p>Analyzing…</p>";
    fetch(`/api/analyze?url=${encodeURIComponent(url)}`)
        .then((res) => {
            if (!res.ok) {
                return res.json().then((err) => {
                    throw new Error(err.error || "Unknown error");
                });
            }
            return res.json();
        })
        .then((data) => {
            const html = `
                <h2>${data.name}</h2>
                <p>${data.description || "No description"}</p>
                <ul>
                    <li>⭐ Stars: ${data.stars}</li>
                    <li>🍴 Forks: ${data.forks}</li>
                    <li>💻 Primary Language: ${data.primary_language || "N/A"}</li>
                    <li>👥 Contributors: ${data.contributor_count}</li>
                </ul>
                <h3>Language Stats</h3>
                <ul>`;
            for (const [lang, bytes] of Object.entries(data.language_stats)) {
                html += `<li>${lang}: ${bytes} bytes</li>`;
            }
            html += "</ul>";
            resultsDiv.innerHTML = html;
        })
        .catch((err) => {
            resultsDiv.innerHTML = `<p class="error">Error: ${err.message}</p>`;
        });
});

