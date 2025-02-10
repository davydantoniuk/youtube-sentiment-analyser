import sqlite3
from googleapiclient.discovery import build
import yaml
import os

# === Load API key from config.yml ===


def load_config():
    config_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "../config.yml"))
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


config = load_config()
API_KEY = config["youtube_api_key"]

# === Database Configuration ===
DB_FILE = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "../comments.db"))


def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS comments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id TEXT NOT NULL,
            video_link TEXT NOT NULL,
            comment_text TEXT NOT NULL,
            published_at DATETIME
        );
    """)
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_video_id ON comments (video_id);")
    conn.commit()
    conn.close()


# === Fetch Comments ===
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"


def fetch_comments(video_link, max_comments=1000):
    video_id = video_link.split("v=")[-1]
    youtube = build(YOUTUBE_API_SERVICE_NAME,
                    YOUTUBE_API_VERSION, developerKey=API_KEY)
    comments = []
    next_page_token = None

    while len(comments) < max_comments:
        response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            pageToken=next_page_token
        ).execute()

        for item in response["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "video_id": video_id,
                "video_link": video_link,
                "comment_text": comment["textDisplay"],
                "published_at": comment["publishedAt"]
            })

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return comments[:max_comments]

# === Save Comments to Database ===


def save_comments_to_db(comments):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    for comment in comments:
        cursor.execute("""
            INSERT INTO comments (video_id, video_link, comment_text, published_at)
            SELECT ?, ?, ?, ?
            WHERE NOT EXISTS (
                SELECT 1 FROM comments WHERE video_id = ? AND comment_text = ?
            )
        """, (
            comment["video_id"],
            comment["video_link"],
            comment["comment_text"],
            comment["published_at"],
            comment["video_id"],
            comment["comment_text"]
        ))

    conn.commit()
    conn.close()


def process_video_comments(video_link):
    comments = fetch_comments(video_link)
    save_comments_to_db(comments)
    return len(comments)
