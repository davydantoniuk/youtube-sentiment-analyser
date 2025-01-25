import sqlite3
from googleapiclient.discovery import build
import yaml
import datetime
import os

# === 1. Load API key from config.yml ===


def load_config():
    """
    Load configuration from a YAML file located two directories above the script.
    """
    # Get the absolute path to config.yml
    config_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "../../config.yml"))
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


# Load the API key from config.yml
config = load_config()
API_KEY = config["youtube_api_key"]

# === 2. Initialize the database ===
DB_FILE = "comments.db"


def init_db():
    """
    Create the SQLite database and the comments table if they do not already exist.
    """
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


# === 3. Fetch comments from YouTube Data API ===
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"


def fetch_comments(video_link, max_comments=1000):
    """
    Fetch comments for a YouTube video using the YouTube Data API.

    Args:
        video_link (str): The YouTube video link.
        max_comments (int): The maximum number of comments to fetch.

    Returns:
        list: A list of comment dictionaries containing video ID, link, text, and timestamp.
    """
    # Extract the video ID from the link
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

# === 4. Save comments to the database ===


def save_comments_to_db(comments):
    """
    Save a list of comments to the SQLite database, avoiding duplicates.

    Args:
        comments (list): A list of comment dictionaries.
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    for comment in comments:
        # Avoid duplicates by checking video_id and comment_text
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

# === 5. Main function ===


def main():
    """
    Main script function to fetch and save YouTube comments for a given video link.
    """
    video_link = input("Enter the YouTube video link: ")
    print("Fetching comments...")

    comments = fetch_comments(video_link)
    print(f"Fetched {len(comments)} comments.")

    print("Saving comments to the database...")
    save_comments_to_db(comments)
    print("Comments saved successfully!")


# === Entry point ===
if __name__ == "__main__":
    init_db()  # Ensure the database is initialized
    main()  # Run the main function
