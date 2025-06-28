import os
import yaml
import yfinance as yf
import pandas as pd
from datetime import timedelta


def load_config(path=None):
    """
    Load config.yaml. If no path is passed, look for <project-root>/configs/config.yaml.
    """
    if path:
        config_path = path
    else:
        here = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(here)
        config_path = os.path.join(project_root, "configs", "config.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found at {config_path!r}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def fetch_and_save_prices(conf):
    """
    Downloads OHLCV data for all tickers using bulk download, handles incremental updates,
    and saves CSVs under the absolute raw_price_dir.
    """
    # Build absolute output directory
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(here)
    raw_dir = os.path.join(project_root, conf["raw_price_dir"])
    os.makedirs(raw_dir, exist_ok=True)

    tickers = conf["tickers"]
    start_date = conf["start_date"]
    end_date = conf["end_date"]

    # Determine per-ticker start dates for incremental updates
    desired_start = {}
    for ticker in tickers:
        csv_path = os.path.join(raw_dir, f"{ticker}.csv")
        if os.path.exists(csv_path):
            df_existing = pd.read_csv(csv_path, parse_dates=["Date"] )
            last_date = df_existing["Date"].max()
            next_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
            desired_start[ticker] = next_date
        else:
            desired_start[ticker] = start_date

    # Bulk download all tickers
    print(f"Downloading data for {tickers} from {start_date} to {end_date}...")
    raw_data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        threads=True,
        group_by="ticker",
        progress=False,
    )

    # Process each ticker separately
    for ticker in tickers:
        df_t = raw_data[ticker].copy()
        df_t.index.name = "Date"
        # Filter new rows only
        df_t = df_t[df_t.index >= desired_start[ticker]]

        csv_path = os.path.join(raw_dir, f"{ticker}.csv")
        if os.path.exists(csv_path):
            # Append and remove duplicates
            df_existing = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")
            combined = pd.concat([df_existing, df_t])
            combined = combined[~combined.index.duplicated(keep="first")]
            combined.to_csv(csv_path)
            print(f"Updated {ticker}: appended data from {desired_start[ticker]}, saved to {csv_path}")
        else:
            df_t.to_csv(csv_path)
            print(f"Saved {ticker}: full data to {csv_path}")


if __name__ == "__main__":
    config = load_config()
    fetch_and_save_prices(config)
