from flask import Flask, render_template, request
from repopulse.github_api import analyze_repository
import os

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    """Handles the main page logic: displaying the form and processing the analysis."""
    repo_url = None
    analysis_result = None
    error_message = None

    if request.method == "POST":
        repo_url = request.form.get("repo_url")
        if repo_url:
            # Call the core API logic
            analysis_result = analyze_repository(repo_url)
            
            if analysis_result["error"]:
                error_message = analysis_result["error"]
            else:
                data = analysis_result["data"]
                # Prepare data for template rendering
                context = {
                    "repo_name": data["name"],
                    "description": data["description"],
                    "stars": f"{data['stars']:,}",
                    "forks": f"{data['forks']:,}",
                    "primary_language": data["primary_language"],
                    "language_stats": data["language_stats"],
                    "contributors_count": data["contributors_count"]
                }
                return render_template("index.html", context=context, repo_url=repo_url)
        else:
            error_message = "Please enter a GitHub repository URL."
    
    # Initial GET request or failed POST request
    return render_template("index.html", error=error_message, repo_url=repo_url)

if __name__ == "__main__":
    # Ensure the directory structure is set up for Flask
    # We run this from the parent directory /home/itsnxfi3/Desktop/daena-agent/EvanaEval
    # and the app is inside repopulse. We need to modify the run command later.
    # For now, we just set up the entry point.
    print("--- RepoPulse Web App Initialized ---")
    print("To run: python repopulse/app.py")
    # We will run it from the parent directory later.
    # app.run(debug=True)
