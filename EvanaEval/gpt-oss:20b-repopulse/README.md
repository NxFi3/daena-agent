# RepoPulse

RepoPulse is a tiny web application that lets you paste a public GitHub
repository URL and get a quick overview of the repository.  It shows:

* Repository name
* Description
* Stars
* Forks
* Primary language
* Language statistics (bytes per language)
* Number of contributors

The application talks to the official GitHub REST API:

* `GET /repos/{owner}/{repo}` – repository information
* `GET /repos/{owner}/{repo}/languages` – language statistics
* `GET /repos/{owner}/{repo}/contributors` – list of contributors

## Running the application

The project is pure Python and uses only the standard library.

```bash
cd EvanaEval/repopulse
python server.py
```

The server will listen on `http://localhost:8000`.  Open that URL in a
browser and enter a GitHub repository URL such as
`https://github.com/NxFi3/SimpleDL`.

## Running the tests

The test suite uses the standard `unittest` framework and mocks the
network calls, so no real HTTP requests are made.

```bash
python -m unittest discover tests
```

## Limitations

* Only public repositories are supported.
* The contributors endpoint is limited to the first 100 contributors.
* No authentication is performed; the GitHub API rate limit for unauthenticated
  requests applies.
* The UI is intentionally minimal and has no styling.
```

