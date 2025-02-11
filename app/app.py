from flask import Flask, render_template, request, jsonify
from comments.parse_comments import process_video_comments, init_db
from models.classify_comments import classify_comment
from visualization.visualization_functions import create_sentiment_barplot
import sqlite3
import os
import re

app = Flask(__name__)
DB_FILE = os.path.abspath("comments/comments.db")

# Initialize the database
init_db()


def extract_video_id(url):
    """Extracts YouTube video ID from various URL formats."""
    match = re.search(
        r"(?:youtu\.be/|youtube\.com/.*(?:v=|\/))([\w-]{11})", url)
    return match.group(1) if match else None


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        video_link = request.form["video_link"]
        video_id = extract_video_id(video_link)

        if not video_id:
            return render_template("index.html", message="Invalid YouTube URL. Please enter a valid link.")

        num_new_comments = process_video_comments(video_id, DB_FILE)
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        # Analyze all comments for overall sentiment counts
        cursor.execute(
            "SELECT comment_text FROM comments WHERE video_id=?", (video_id,))
        all_comments = cursor.fetchall()
        pos_count = 0
        neu_count = 0
        neg_count = 0
        for c in all_comments:
            sentiment = classify_comment(c[0])
            if sentiment == 0:
                neg_count += 1
            elif sentiment == 1:
                neu_count += 1
            else:
                pos_count += 1

        plot_data = create_sentiment_barplot(pos_count, neu_count, neg_count)

        # Fetch only the last 10 comments for display
        cursor.execute(
            "SELECT comment_text, published_at FROM comments WHERE video_id=? ORDER BY published_at DESC LIMIT 10;",
            (video_id,)
        )
        recent_comments = cursor.fetchall()

        conn.close()

        return render_template(
            "index.html",
            message=f"Added {num_new_comments} new comments to database!",
            video_id=video_id,
            comments=recent_comments,
            positive_count=pos_count,
            neutral_count=neu_count,
            negative_count=neg_count,
            plot_data=plot_data
        )

    return render_template("index.html")


@app.route("/comments", methods=["GET"])
def get_comments():
    video_id = request.args.get("video_id")
    if not video_id:
        return jsonify({"error": "Missing video_id"}), 400

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT comment_text, published_at FROM comments WHERE video_id = ? ORDER BY published_at DESC LIMIT 10;",
        (video_id,)
    )
    comments = cursor.fetchall()
    conn.close()

    return jsonify(comments)


if __name__ == "__main__":
    app.run(debug=True)
