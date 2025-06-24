import os
import csv
import praw
from datetime import datetime, timezone

# Initialize with your real credentials
reddit = praw.Reddit(
    client_id="IZ-EhbjZECcYYTfMi2nBDg",
    client_secret="K8pO4mW4cUO1HY63IMZeIwgrNbRliQ",
    user_agent="stock-sentiment/0.1 by Unhappy-One145"
)

TICKERS    = ["AAPL", "TSLA", "NVDA"]
SUBREDDITS = ["stocks", "wallstreetbets"]
OUT_DIR    = "data/raw/sentiment_text"
os.makedirs(OUT_DIR, exist_ok=True)

# Fetch top posts in the last 24h for each ticker + subreddit
for ticker in TICKERS:
    rows = []
    for sub in SUBREDDITS:
        subreddit = reddit.subreddit(sub)
        for post in subreddit.search(f"${ticker}", limit=200, time_filter="day"):
            ts = datetime.fromtimestamp(post.created_utc, timezone.utc).isoformat()
            rows.append((ts, ticker, post.title + "\n\n" + post.selftext))

    out_path = os.path.join(OUT_DIR, f"{ticker}.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "ticker", "text"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} posts for {ticker} → {out_path}")

