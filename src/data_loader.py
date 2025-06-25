import os
import yaml
import yfinance as yf

def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

def fetch_and_save_prices(conf):
    out_dir = conf["raw_price_dir"]
    os.makedirs(out_dir, exist_ok=True)

    for ticker in conf["tickers"]:
        print(f"Fetching {ticker} from {conf['start_date']} to {conf['end_date']}…")
        df = yf.download(
            ticker,
            start=conf["start_date"],
            end=conf["end_date"],
            progress=False,
        )
        csv_path = os.path.join(out_dir, f"{ticker}.csv")
        df.to_csv(csv_path)
        print(f"  → saved to {csv_path}")

if __name__ == "__main__":
    config = load_config()
    fetch_and_save_prices(config)
