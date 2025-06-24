<<<<<<< HEAD
=======

>>>>>>> 3411695 (Merge origin/main into main – resolved conflicts)
import os
import csv
import praw
from datetime import datetime, timezone

# 1) Set up PRAW (you’ll need to register an app at https://www.reddit.com/prefs/apps)
reddit = praw.Reddit(
    client_id="YOUR_CLIENT_ID",
    client_secret="YOUR_SECRET",
    user_agent="ucsd_stock_sentiment/0.1 by YOUR_USERNAME"
)

# 2) Configuration
<<<<<<< HEAD
=======

>>>>>>> 3411695 (Merge origin/main into main – resolved conflicts)
import praw
from datetime import datetime, timezone
import os, csv

# ← Replace these with your actual credentials
reddit = praw.Reddit(
    client_id="IZ-EhbjZECcYYTfMi2nBDg",
    client_secret="K8pO4mW4cUO1HY63IMZeIwgrNbRliQ",
    user_agent="stock-sentiment/0.1 by Unhappy-One145"
)
TICKERS = ["AAPL", "TSLA", "NVDA"]
SUBREDDITS = ["stocks", "wallstreetbets"]
OUT_DIR = "data/raw/sentiment_text"
os.makedirs(OUT_DIR, exist_ok=True)

# 3) Fech top posts in last 24h for each ticker + subreddit
for ticker in TICKERS:
    rows = []
    for sub in SUBREDDITS:
        subreddit = reddit.subreddit(sub)
        for post in subreddit.search(f"${ticker}", limit=200, time_filter="day"):
            ts = datetime.fromtimestamp(post.created_utc, timezone.utc).isoformat()
            rows.append((ts, ticker, post.title + "\n\n" + post.selftext))
<<<<<<< Updated upstream

    # 4) Write CSV

=======
<<<<<<< HEAD
    # 4) Write CSV
=======
>>>>>>> 12d142cf135db447f6bd6b1a4917ac86d6a95e75
>>>>>>> Stashed changes
    out_path = os.path.join(OUT_DIR, f"{ticker}.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "ticker", "text"])
        writer.writerows(rows)
<<<<<<< HEAD
    print(f"Wrote {len(rows)} Reddit posts for {ticker} to {out_path}")
<<<<<<< Updated upstream
    print(f"Wrote {len(rows)} posts for {ticker} → {out_path}")

=======
=======
    print(f"Wrote {len(rows)} posts for {ticker} → {out_path}")
>>>>>>> 12d142cf135db447f6bd6b1a4917ac86d6a95e75
>>>>>>> Stashed changes
