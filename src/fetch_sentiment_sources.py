import os, yaml, pandas as pd
from src.fetch_news import fetch_news_df       # returns DataFrame
from src.fetch_twitter import fetch_twitter_df # you’ll write similarly
from src.fetch_reddit import fetch_reddit_df
# … import other fetchers

def load_config():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    with open(os.path.join(root, "configs", "config.yaml")) as f:
        return yaml.safe_load(f)

def main():
    cfg = load_config()
    out_dir = os.path.join("data", "raw", "sentiment_text")
    os.makedirs(out_dir, exist_ok=True)

    # 1) Fetch from each source
    dfs = []
    dfs.append(fetch_news_df(cfg))       # adds column source='newsapi'
    dfs.append(fetch_twitter_df(cfg))    # source='twitter'
    dfs.append(fetch_reddit_df(cfg))     # source='reddit'
    # … dfs.append(...) other sources

    # 2) Concatenate & clean
    full = pd.concat(dfs, ignore_index=True)
    full = full.dropna(subset=["text"])
    full["timestamp"] = pd.to_datetime(full["timestamp"])
    full = full.sort_values("timestamp")

    # 3) Write a single master CSV (or per-ticker CSVs)
    full.to_csv(os.path.join(out_dir, "all_sources.csv"), index=False)
    # Or split by ticker:
    for ticker, grp in full.groupby("ticker"):
        fn = os.path.join(out_dir, f"{ticker}.csv")
        grp.to_csv(fn, index=False)
        print(f"Wrote {len(grp)} rows for {ticker}")

if __name__ == "__main__":
    main()
