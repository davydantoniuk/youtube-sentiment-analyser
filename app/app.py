from flask import Flask, render_template, request, jsonify, session
import threading
import time
import os
import re
import sqlite3
from uuid import uuid4

from comments.parse_comments import process_video_comments, init_db
from models.classify_comments import classify_comment
from visualization.visualization_functions import (
    create_sentiment_barplot,
    create_word_frequency_barplots,
    create_word_cloud,
    create_phrase_frequency_barplots
)

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Set a secret key for session management
DB_FILE = os.path.abspath("comments/comments.db")

# Initialize the database
init_db()

# Global dictionary to track progress for each task and a lock for thread safety
tasks = {}
tasks_lock = threading.Lock()


def extract_video_id(url):
    """Extracts YouTube video ID from various URL formats."""
    match = re.search(
        r"(?:youtu\.be/|youtube\.com/.*(?:v=|\/)([\w-]{11}))", url)
    return match.group(1) if match else None


@app.route("/", methods=["GET"])
def index():
    """Render the home page with results if available."""
    task_id = session.get('task_id')
    results = None

    if task_id:
        with tasks_lock:
            task = tasks.get(task_id)
            if task and task['state'] == 'done' and task['results']:
                results = task['results']

    return render_template("index.html", **results) if results else render_template("index.html")


@app.route("/start_process", methods=["POST"])
def start_process():
    """Start a background task to process comments and return the task ID."""
    video_link = request.form.get("video_link", "")
    video_id = extract_video_id(video_link)

    if not video_id:
        return jsonify({"error": "Invalid YouTube URL."}), 400

    # Generate a unique task ID and set up the progress entry
    task_id = str(uuid4())
    new_progress = {
        "state": "running",
        "current": 0,
        "total": 0,
        "message": "Starting...",
        "results": None
    }

    with tasks_lock:
        # Clean up any existing task for this session
        old_task_id = session.get('task_id')
        if old_task_id in tasks:
            del tasks[old_task_id]
        # Add the new task
        tasks[task_id] = new_progress
        session['task_id'] = task_id

    # Start the background task
    thread = threading.Thread(target=background_task, args=(task_id, video_id))
    thread.start()

    return jsonify({"status": "Processing started.", "task_id": task_id})


@app.route("/progress", methods=["GET"])
def get_progress():
    """Retrieve progress for the current user's task."""
    task_id = session.get('task_id')
    if not task_id:
        return jsonify({"error": "No task ID found."}), 400

    with tasks_lock:
        task = tasks.get(task_id, {
            "state": "error",
            "message": "Task not found.",
            "current": 0,
            "total": 0
        })

    return jsonify({
        "state": task["state"],
        "current": task["current"],
        "total": task["total"],
        "message": task["message"]
    })


def background_task(task_id, video_id):
    """Process comments and update progress for the given task ID."""
    try:
        with tasks_lock:
            tasks[task_id]["message"] = "Fetching comments..."
            tasks[task_id]["state"] = "running"

        # Fetch comments and update progress
        num_new_comments = process_video_comments(video_id, DB_FILE)

        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM comments WHERE video_id=?", (video_id,))
        total_comments = cursor.fetchone()[0]

        if total_comments == 0:
            conn.close()
            with tasks_lock:
                tasks[task_id].update({
                    "state": "done",
                    "results": {"message": "No comments available for this video."}
                })
            return

        with tasks_lock:
            tasks[task_id]["message"] = "Classifying comments..."

        cursor.execute(
            "SELECT comment_text FROM comments WHERE video_id=?", (video_id,))
        all_comments = cursor.fetchall()
        conn.close()

        # Initialize counters
        pos_count = neu_count = neg_count = spam_count = english_count = non_english_count = 0
        pos_comments = []
        neu_comments = []
        neg_comments = []

        with tasks_lock:
            tasks[task_id]["total"] = len(all_comments)
            tasks[task_id]["current"] = 0

        for i, (comment_text,) in enumerate(all_comments):
            sentiment, language = classify_comment(comment_text)
            english_count += 1 if language == "en" else 0
            non_english_count += 0 if language == "en" else 1

            if sentiment == 0:
                neg_count += 1
                neg_comments.append(comment_text)
            elif sentiment == 1:
                neu_count += 1
                neu_comments.append(comment_text)
            elif sentiment == "Spam":
                spam_count += 1
            else:
                pos_count += 1
                pos_comments.append(comment_text)

            with tasks_lock:
                tasks[task_id]["current"] = i + 1

        # Generate visualizations
        with tasks_lock:
            tasks[task_id]["message"] = "Generating plots..."

        plot_data = create_sentiment_barplot(pos_count, neu_count, neg_count)
        word_freq_plot = create_word_frequency_barplots(
            pos_comments, neu_comments, neg_comments)
        phrase_freq_plot = create_phrase_frequency_barplots(
            pos_comments, neu_comments, neg_comments)
        word_cloud_data = create_word_cloud([c[0] for c in all_comments])

        # Fetch recent comments
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT comment_text, published_at 
            FROM comments 
            WHERE video_id=? 
            ORDER BY published_at DESC 
            LIMIT 10;
        """, (video_id,))
        recent_comments = cursor.fetchall()
        conn.close()

        # Update task results
        with tasks_lock:
            tasks[task_id].update({
                "state": "done",
                "message": "Complete!",
                "results": {
                    "message": f"Added {num_new_comments} new comments!" if num_new_comments > 0 else "Analysis complete.",
                    "total_comments": total_comments,
                    "english_comments": english_count,
                    "non_english_comments": non_english_count,
                    "positive_count": pos_count,
                    "neutral_count": neu_count,
                    "negative_count": neg_count,
                    "spam_count": spam_count,
                    "plot_data": plot_data,
                    "word_freq_plot": word_freq_plot,
                    "phrase_freq_plot": phrase_freq_plot,
                    "word_cloud_data": word_cloud_data,
                    "comments": recent_comments,
                    "video_id": video_id
                }
            })

    except Exception as e:
        with tasks_lock:
            tasks[task_id].update({
                "state": "error",
                "message": str(e)
            })


@app.route("/show_comments/<video_id>", methods=["GET"])
def show_comments(video_id):
    return render_template("show_comments.html", video_id=video_id)


@app.route("/fetch_comments/<video_id>", methods=["GET"])
def fetch_comments(video_id):
    try:
        page = int(request.args.get("page", 1))
        limit = 50
        offset = (page - 1) * limit

        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT comment_text 
            FROM comments 
            WHERE video_id=? 
            ORDER BY published_at DESC 
            LIMIT ? OFFSET ?;
        """, (video_id, limit, offset))
        comments = [row[0] for row in cursor.fetchall()]
        conn.close()

        return jsonify({"comments": comments, "has_more": len(comments) == limit})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
