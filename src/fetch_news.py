import os
import yaml
import requests
import pandas as pd
import logging
from datetime import datetime

# Configure logging
tlogging = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')


def load_config(config_path=None):
    """
    Load configuration from configs/config.yaml by default.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(here)
    cfg_path = config_path or os.path.join(project_root, 'configs', 'config.yaml')
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config not found at {cfg_path}")
    with open(cfg_path, 'r') as f:
        return yaml.safe_load(f)


def fetch_news_df(cfg):
    """
    Fetch news articles for each ticker using NewsAPI.org.
    Returns a DataFrame with columns: timestamp, ticker, text, source.
    """
    api_key = cfg.get('newsapi_key')
    if not api_key:
        raise KeyError("`newsapi_key` must be set in config.yaml")

    tickers = cfg.get('tickers', [])
    start = cfg.get('start_date')
    end = cfg.get('end_date')
    if not tickers or not start or not end:
        raise KeyError("`tickers`, `start_date`, and `end_date` must be set in config.yaml")

    base_url = 'https://newsapi.org/v2/everything'
    page_size = 100  # max allowed by NewsAPI
    max_pages = 5    # fetch up to 500 results per ticker

    records = []
    headers = {'Authorization': api_key}

    for ticker in tickers:
        logging.info(f"Fetching news for {ticker} from {start} to {end}")
        for page in range(1, max_pages + 1):
            params = {
                'q': ticker,
                'from': start,
                'to': end,
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': page_size,
                'page': page,
            }
            try:
                resp = requests.get(base_url, params=params, timeout=10)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                logging.error(f"Error fetching page {page} for {ticker}: {e}")
                break

            articles = data.get('articles', [])
            if not articles:
                break

            for art in articles:
                ts = art.get('publishedAt') or art.get('published_at')
                title = art.get('title', '')
                descr = art.get('description', '')
                content = art.get('content', '')
                text = content or descr or title
                if not text:
                    continue
                try:
                    timestamp = pd.to_datetime(ts)
                except Exception:
                    timestamp = ts
                records.append({
                    'timestamp': timestamp,
                    'ticker': ticker,
                    'text': text,
                    'source': 'newsapi'
                })

            total_results = data.get('totalResults', 0)
            if page * page_size >= total_results:
                # no more pages
                break

    if not records:
        logging.warning("No news articles fetched; check your API key and query parameters.")
        return pd.DataFrame(columns=['timestamp', 'ticker', 'text', 'source'])

    df = pd.DataFrame(records)
    # Ensure timestamp is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


if __name__ == '__main__':
    cfg = load_config()
    df_news = fetch_news_df(cfg)
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'raw', 'sentiment_text')
    os.makedirs(out_dir, exist_ok=True)
    # Save a combined CSV per run
    output_path = os.path.join(out_dir, 'newsapi_all.csv')
    df_news.to_csv(output_path, index=False)
    logging.info(f"Saved {len(df_news)} articles to {output_path}")
