from flask import Flask, render_template, request, jsonify
import threading
import time
import os
import re
import sqlite3

from comments.parse_comments import process_video_comments, init_db
from models.classify_comments import classify_comment
from visualization.visualization_functions import (
    create_sentiment_barplot,
    create_word_frequency_barplots,
    create_word_cloud,
    create_phrase_frequency_barplots
)

app = Flask(__name__)
DB_FILE = os.path.abspath("comments/comments.db")

# Initialize the database
init_db()

# A simple global dictionary to store progress
# In production, store this in a database or cache for multi-user concurrency
progress_status = {
    "state": "idle",     # can be 'idle', 'running', 'done', 'error'
    "current": 0,        # current processed comments
    "total": 0,          # total comments to process
    "message": "",       # any status message
    "results": None      # to store final results (counts, etc.)
}


def extract_video_id(url):
    """Extracts YouTube video ID from various URL formats."""
    match = re.search(
        r"(?:youtu\.be/|youtube\.com/.*(?:v=|\/))([\w-]{11})", url)
    return match.group(1) if match else None


@app.route("/", methods=["GET"])
def index():
    """
    Render the home page where user can input the YouTube link.
    If we already have results in progress_status, we can display them too.
    """
    # If the job is done and we have final results, display them
    if progress_status["state"] == "done" and progress_status["results"]:
        return render_template("index.html", **progress_status["results"])

    # Otherwise, just render the blank form
    return render_template("index.html")


@app.route("/start_process", methods=["POST"])
def start_process():
    """
    This route is called via AJAX when the user submits the form.
    It spawns a background thread to fetch & classify comments.
    Returns a simple JSON response to confirm the process started.
    """
    video_link = request.form.get("video_link", "")
    video_id = extract_video_id(video_link)

    if not video_id:
        return jsonify({"error": "Invalid YouTube URL."}), 400

    # Reset global progress
    progress_status["state"] = "running"
    progress_status["current"] = 0
    progress_status["total"] = 0
    progress_status["message"] = "Starting..."
    progress_status["results"] = None

    # Start the background thread with all the heavy lifting
    thread = threading.Thread(target=background_task, args=(video_id,))
    thread.start()

    return jsonify({"status": "Processing started."})


@app.route("/progress", methods=["GET"])
def get_progress():
    """
    The frontend polls this endpoint to get the latest progress status.
    Returns JSON with current progress info.
    """
    return jsonify({
        "state": progress_status["state"],
        "current": progress_status["current"],
        "total": progress_status["total"],
        "message": progress_status["message"]
    })


def background_task(video_id):
    """
    The heavy-lifting function that:
      1) Fetches & saves comments,
      2) Classifies them,
      3) Generates plots,
      4) Stores final results in progress_status['results'].
    We'll periodically update progress_status so the frontend can see progress.
    """
    try:
        # 1) Fetch comments from YouTube
        progress_status["message"] = "Fetching comments..."
        num_new_comments = process_video_comments(video_id, DB_FILE)

        # Check total comments after fetching
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM comments WHERE video_id=?", (video_id,))
        total_comments = cursor.fetchone()[0]

        if total_comments == 0:
            # no comments at all
            conn.close()
            progress_status["state"] = "done"
            progress_status["results"] = {
                "message": "No comments available for this video.",
                "comments": None
            }
            return

        # 2) Classify comments
        progress_status["message"] = "Classifying comments..."
        cursor.execute(
            "SELECT comment_text FROM comments WHERE video_id=?", (video_id,))
        all_comments = cursor.fetchall()
        conn.close()

        # Prepare counters
        pos_count = 0
        neu_count = 0
        neg_count = 0
        spam_count = 0
        pos_comments = []
        neu_comments = []
        neg_comments = []

        # We know how many comments we have to classify
        progress_status["total"] = len(all_comments)
        progress_status["current"] = 0

        for c in all_comments:
            # For each comment, classify it
            sentiment = classify_comment(c[0])
            if sentiment == 0:
                neg_count += 1
                neg_comments.append(c[0])
            elif sentiment == 1:
                neu_count += 1
                neu_comments.append(c[0])
            elif sentiment == "Spam":
                spam_count += 1
            else:
                pos_count += 1
                pos_comments.append(c[0])

            # Update current progress
            progress_status["current"] += 1

            # (Optional) Sleep briefly to simulate “long” work, so we can see progress
            # time.sleep(0.01)

        # 3) Create visualizations
        progress_status["message"] = "Generating plots..."
        plot_data = create_sentiment_barplot(pos_count, neu_count, neg_count)
        word_freq_plot = create_word_frequency_barplots(
            pos_comments, neu_comments, neg_comments
        )
        phrase_freq_plot = create_phrase_frequency_barplots(
            pos_comments, neu_comments, neg_comments
        )

        warning_message_plot = None
        if word_freq_plot is None or phrase_freq_plot is None:
            warning_message_plot = (
                "Warning: Small number of comments detected. "
                "Word & phrase frequency plots were not generated."
            )

        # Create word cloud
        all_comments_text = [c[0] for c in all_comments]
        word_cloud_data = create_word_cloud(all_comments_text)

        # 4) Fetch only the last 10 comments to display
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT comment_text, published_at FROM comments WHERE video_id=? "
            "ORDER BY published_at DESC LIMIT 10;",
            (video_id,)
        )
        recent_comments = cursor.fetchall()
        conn.close()

        # Save final results into progress_status
        progress_status["state"] = "done"
        progress_status["message"] = "Complete!"
        progress_status["results"] = {
            "message": (
                f"Added {num_new_comments} new comments to database!"
                if num_new_comments > 0 else
                "Comments already exist in the database. Analysis performed."
            ),
            "total_comments": total_comments,
            "warning_plot": warning_message_plot,
            "video_id": video_id,
            "comments": recent_comments,
            "positive_count": pos_count,
            "neutral_count": neu_count,
            "negative_count": neg_count,
            "spam_count": spam_count,
            "plot_data": plot_data,
            "word_freq_plot": word_freq_plot,
            "word_cloud_data": word_cloud_data,
            "phrase_freq_plot": phrase_freq_plot
        }

    except Exception as e:
        # If something goes wrong, set error state
        progress_status["state"] = "error"
        progress_status["message"] = str(e)


if __name__ == "__main__":
    app.run(debug=True)
