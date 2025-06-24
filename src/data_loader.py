import os
import time
import sqlite3

import yaml
import pandas as pd
import yfinance as yf


class DataLoader:
    def __init__(self, config_path="config/config.yaml"):
        # Load config
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        self.tickers    = cfg["tickers"]
        self.start      = cfg["start_date"]
        self.end        = cfg["end_date"]
        s_cfg           = cfg["sentiment"]
        dl_cfg          = cfg["data_loader"]

        self.sent_raw_dir  = s_cfg["raw_data_dir"]
        self.sent_out_dir  = s_cfg["output_dir"]
        self.proc_path     = dl_cfg["processed_data_path"]

        # Make sure output folders exist
        os.makedirs(os.path.dirname(self.proc_path), exist_ok=True)

    def download_price_data(self):
        """Download OHLCV for each ticker, retrying on SQLite lock."""
        price_frames = []
        failed = []

        for ticker in self.tickers:
            attempts = 0
            df = None
            while attempts < 3:
                try:
                    df = yf.download(
                        ticker,
                        start=self.start,
                        end=self.end,
                        threads=False  # disable threaded caching
                    )
                    break
                except sqlite3.OperationalError:
                    attempts += 1
                    print(f"⚠️  database locked for {ticker}, retry {attempts}/3")
                    time.sleep(2)

            if df is None or df.empty:
                print(f"❌  Giving up on {ticker} after {attempts} retries")
                failed.append(ticker)
                continue

            df = df.reset_index()
            df["Ticker"] = ticker
            price_frames.append(df)

        if failed:
            print(f"⚠️  Failed downloads for: {failed}")
        return pd.concat(price_frames, ignore_index=True)

    def load_sentiment_data(self):
        """Read per‐ticker sentiment CSVs and combine into one DataFrame."""
        frames = []
        for fn in os.listdir(self.sent_out_dir):
            if not fn.lower().endswith(".csv"):
                continue
            df = pd.read_csv(os.path.join(self.sent_out_dir, fn))
            # Ensure Date is timezone-naive to match price data
            df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
            frames.append(df)
        return pd.concat(frames, ignore_index=True)

    def merge_data(self, price_df, sentiment_df):
        """Left‐merge price + sentiment on Date+Ticker."""
        merged = price_df.merge(
            sentiment_df,
            on=["Date", "Ticker"],
            how="left"
        )
        # Fill any missing sentiment with zero or neutral value
        merged["sentiment_score"] = merged["sentiment_score"].fillna(0.0)
        return merged

    def add_indicators_and_labels(self, df):
        """Compute simple moving averages, returns & binary labels."""
        df = df.sort_values(["Ticker", "Date"])
        # Example indicators
        df["return"] = df.groupby("Ticker")["Close"].pct_change()
        df["ma10"]   = df.groupby("Ticker")["Close"].rolling(10).mean().reset_index(0,drop=True)
        df["ma50"]   = df.groupby("Ticker")["Close"].rolling(50).mean().reset_index(0,drop=True)
        # Label = 1 if next-day return > 0, else 0
        df["label"]  = (df.groupby("Ticker")["return"].shift(-1) > 0).astype(int)
        return df.dropna(subset=["return", "ma10", "ma50", "label"])

    def process(self):
        # 1) Download price data
        price_df = self.download_price_data()

        # 2) Load sentiment scores
        sent_df  = self.load_sentiment_data()

        # 3) Merge
        merged   = self.merge_data(price_df, sent_df)

        # 4) Indicators + labels
        final_df = self.add_indicators_and_labels(merged)

        # 5) Write out
        final_df.to_csv(self.proc_path, index=False)
        print(f"✅ Wrote processed data to {self.proc_path}")
        return final_df


if __name__ == "__main__":
    loader = DataLoader()
    loader.process()

