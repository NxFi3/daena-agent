# RepoPulse - GitHub Repository Analyzer

A simple web application built to analyze public GitHub repositories using the GitHub REST API.

## 🚀 Getting Started

This project requires Python 3.x and the `Flask` and `requests` libraries.

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd EvanaEval/repopulse
   ```

2. **Install dependencies:**
   ```bash
   pip install Flask requests
   ```

### Running the Application

To start the web server, run the application module:
```bash
python -m repopulse.app
```
The application will be accessible at `http://127.0.0.1:5000/`.

## 🧪 Running Tests

The automated tests ensure that the API data processing logic is correct and isolated from live network calls.

To run the complete test suite:
```bash
python -m unittest repopulse.test_github_api
```

## 🌐 GitHub API Endpoints Used

The application interacts with the following GitHub REST API endpoints:

1. **Repository Details:** `GET /repos/{owner}/{repo}`
   - Used to fetch basic metadata (name, description, stars, forks).
2. **Repository Languages:** `GET /repos/{owner}/{repo}/languages`
   - Used to fetch language statistics (size/character count).
3. **Repository Contributors:** `GET /repos/{owner}/{repo}/contributors?per_page=1`
   - Used to fetch the count of unique contributors.

## ⚠️ Limitations

1. **Rate Limiting:** The application is subject to GitHub's API rate limits. Excessive requests may fail.
2. **API Structure:** The application relies on the current structure of the GitHub API. Changes to these endpoints may require code updates.
3. **Dependencies:** While standard libraries are preferred, `requests` is used for robust HTTP handling, and `Flask` is required for the web framework.

## 📁 Project Structure

```
repopulse/
├── __init__.py
├── app.py               # Flask web application entry point
├── github_api.py        # Core API interaction logic
├── templates/
│   └── index.html       # Frontend HTML template
├── test_github_api.py   # Unit tests for API logic
└── README.md            # Project documentation
```

## 💡 How it Works

The `app.py` file handles the web request lifecycle. Upon submission, it calls `analyze_repository` in `github_api.py`. This function constructs the correct API URLs, makes sequential requests to GitHub, handles potential API errors (like 404 or network failures), and aggregates the data into a clean dictionary format, which is then passed to the `index.html` template for rendering.


