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
TICKERS = ["AAPL", "TSLA", "NVDA"]
SUBREDDITS = ["stocks", "wallstreetbets"]
OUT_DIR = "data/raw/sentiment_text"
os.makedirs(OUT_DIR, exist_ok=True)

# 3) Fetch top posts in last 24h for each ticker + subreddit
for ticker in TICKERS:
    rows = []
    for sub in SUBREDDITS:
        subreddit = reddit.subreddit(sub)
        for post in subreddit.search(f"${ticker}", limit=200, time_filter="day"):
            ts = datetime.fromtimestamp(post.created_utc, timezone.utc).isoformat()
            rows.append((ts, ticker, post.title + "\n\n" + post.selftext))
    # 4) Write CSV
    out_path = os.path.join(OUT_DIR, f"{ticker}.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "ticker", "text"])
        writer.writerows(rows)
    print(f"Wrote {len(rows)} Reddit posts for {ticker} to {out_path}")
