import os
import time
import sqlite3

import yaml
import pandas as pd
import yfinance as yf


class DataLoader:
    def __init__(self, config_path="config/config.yaml"):
        # 1) Load config
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        self.tickers    = cfg["tickers"]
        self.start      = cfg["start_date"]
        self.end        = cfg["end_date"]

        s_cfg           = cfg["sentiment"]
        dl_cfg          = cfg["data_loader"]
        self.sent_out   = s_cfg["output_dir"]
        self.proc_path  = dl_cfg["processed_data_path"]

        # 2) Ensure output folder exists (cleanup if it was a file)
        out_dir = os.path.dirname(self.proc_path)
        if os.path.exists(out_dir) and not os.path.isdir(out_dir):
            os.remove(out_dir)
        os.makedirs(out_dir, exist_ok=True)

    def download_price_data(self):
        """Download OHLCV per‐ticker, retrying on SQLite locked errors."""
        all_dfs = []
        failed = []

        for ticker in self.tickers:
            df = None
            for attempt in range(1, 4):
                try:
                    df = yf.download(
                        ticker,
                        start=self.start,
                        end=self.end,
                        threads=False  # avoid cache‐locking
                    )
                    break
                except sqlite3.OperationalError:
                    print(f"⚠️  database locked for {ticker}, retry {attempt}/3")
                    time.sleep(2)

            if df is None or df.empty:
                print(f"❌  Giving up on {ticker} after 3 attempts")
                failed.append(ticker)
                continue

            # Reset the index (so Date becomes a column) and tag ticker
            df = df.reset_index()
            df["Ticker"] = ticker
            all_dfs.append(df)

        if failed:
            print(f"⚠️  Failed downloads: {failed}")
        return pd.concat(all_dfs, ignore_index=True)

    def load_sentiment_data(self):
        """Read the per‐ticker sentiment CSVs, rename + parse their timestamps."""
        frames = []
        for fn in os.listdir(self.sent_out):
            if not fn.lower().endswith(".csv"):
                continue
            path = os.path.join(self.sent_out, fn)
            df = pd.read_csv(path)

            # rename our columns to match the price DataFrame
            if "timestamp" in df.columns:
                df = df.rename(columns={
                    "timestamp": "Date",
                    "ticker": "Ticker"
                })

            # parse Date, strip any tz so it's naive
            df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
            frames.append(df)

        return pd.concat(frames, ignore_index=True)

    def merge_data(self, price_df, sentiment_df):
        """Left‐merge on Date & Ticker, fill missing sentiment."""
        merged = price_df.merge(
            sentiment_df,
            on=["Date", "Ticker"],
            how="left"
        )
        merged["sentiment_score"] = merged.get("sentiment_score", 0.0).fillna(0.0)
        return merged

    def add_indicators_and_labels(self, df):
        """Compute pct‐return, moving averages, and next‐day binary label."""
        df = df.sort_values(["Ticker", "Date"])
        df["return"] = df.groupby("Ticker")["Close"].pct_change()
        df["ma10"]   = (
            df.groupby("Ticker")["Close"]
              .rolling(10).mean()
              .reset_index(0, drop=True)
        )
        df["ma50"]   = (
            df.groupby("Ticker")["Close"]
              .rolling(50).mean()
              .reset_index(0, drop=True)
        )
        df["label"]  = (df.groupby("Ticker")["return"].shift(-1) > 0).astype(int)

        return df.dropna(subset=["return", "ma10", "ma50", "label"])

    def process(self):
        # 1) Price fetch
        price_df = self.download_price_data()

        # 2) Flatten any MultiIndex columns from yfinance
        if isinstance(price_df.columns, pd.MultiIndex):
            # keep only the inner level (e.g. 'Date', 'Open', 'Close', etc.)
            price_df.columns = price_df.columns.get_level_values(-1)

        # 3) Sentiment fetch
        sent_df = self.load_sentiment_data()

        # 4) Merge price + sentiment
        merged = self.merge_data(price_df, sent_df)

        # 5) Indicators & labels
        final_df = self.add_indicators_and_labels(merged)

        # 6) Write out
        final_df.to_csv(self.proc_path, index=False)
        print(f"✅ Wrote processed data to {self.proc_path}")
        return final_df


if __name__ == "__main__":
    DataLoader().process()
