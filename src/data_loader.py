import os
import yaml
import pandas as pd
import yfinance as yf


class DataLoader:
    """
    DataLoader handles downloading price data, loading sentiment data,
    merging features, computing technical indicators, and generating labels.
    """
    def __init__(self, config_path="config/config.yaml"):
        # Load configuration
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.tickers = self.config.get("tickers", [])
        self.pred_horizon = self.config.get("prediction_horizon", 3)
        self.sentiment_window = self.config.get("sentiment_window_days", 1)
        self.start_date = self.config.get("start_date")
        self.end_date = self.config.get("end_date")
        self.sentiment_dir = self.config.get("sentiment_output_dir", "data/raw/sentiment")

    def download_price_data(self):
        """
        Download OHLCV price data for configured tickers.
        Returns a DataFrame with columns: Date, Ticker, Open, High, Low, Close, Volume.
        """
        raw = yf.download(
            tickers=self.tickers,
            start=self.start_date,
            end=self.end_date,
            group_by='ticker',
            auto_adjust=True,
            progress=False
        )
        df_list = []
        for ticker in self.tickers:
            temp = raw[ticker].reset_index()
            temp['Ticker'] = ticker
            df_list.append(temp)
        price_df = pd.concat(df_list, ignore_index=True)
        price_df = price_df.rename(columns={
            'Date': 'Date', 'Open': 'Open', 'High': 'High',
            'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume'
        })
        return price_df

    def load_sentiment_data(self):
        """
        Load all sentiment CSV files from the configured sentiment_output_dir.
        Assumes each CSV has columns: timestamp, ticker, sentiment_score.
        Returns a concatenated DataFrame.
        """
        dir_path = self.sentiment_dir
        if not os.path.isdir(dir_path):
            raise FileNotFoundError(f"Sentiment directory not found: {dir_path}")
        files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.csv')]
        if not files:
            raise FileNotFoundError(f"No sentiment CSV files found in {dir_path}")
        dfs = []
        for fp in files:
            df = pd.read_csv(fp, parse_dates=['timestamp'])
            dfs.append(df)
        sentiment_df = pd.concat(dfs, ignore_index=True)
        return sentiment_df

    def merge_data(self, price_df, sentiment_df):
        """
        Merge price and sentiment data on Date and Ticker.
        Computes daily average sentiment and fills missing values with 0.
        """
        sentiment_df['Date'] = sentiment_df['timestamp'].dt.floor('D')
        daily_sent = (
            sentiment_df
            .groupby(['Date', 'ticker'])['sentiment_score']
            .mean()
            .reset_index()
            .rename(columns={'ticker': 'Ticker'})
        )
        merged = price_df.merge(daily_sent, how='left', on=['Date', 'Ticker'])
        merged['sentiment_score'] = merged['sentiment_score'].fillna(0.0)
        merged = merged.sort_values(['Ticker', 'Date']).reset_index(drop=True)
        return merged

    def add_technical_indicators(self, df):
        """
        Add basic technical indicators to DataFrame:
        - Daily returns
        - Moving averages (10-day, 50-day)
        """
        df['return'] = df.groupby('Ticker')['Close'].pct_change()
        df['ma10'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=10).mean())
        df['ma50'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=50).mean())
        return df

    def add_labels(self, df):
        """
        Create target labels based on future price movement:
        - future_close: Close price shifted by -prediction_horizon days
        - label: 1 if future_return > 0 else 0
        """
        df['future_close'] = df.groupby('Ticker')['Close'].shift(-self.pred_horizon)
        df['future_return'] = (df['future_close'] - df['Close']) / df['Close']
        df['label'] = (df['future_return'] > 0).astype(int)
        df = df.drop(columns=['future_close', 'future_return'])
        return df

    def process(self):
        """
        Orchestrate the full data loading and processing pipeline:
        1. Download price data
        2. Load sentiment data
        3. Merge datasets
        4. Add technical indicators
        5. Generate labels
        6. Save processed data to CSV
        Returns final DataFrame.
        """
        price_df = self.download_price_data()
        sentiment_df = self.load_sentiment_data()
        merged_df = self.merge_data(price_df, sentiment_df)
        feats_df = self.add_technical_indicators(merged_df)
        final_df = self.add_labels(feats_df)
        os.makedirs('data/processed', exist_ok=True)
        final_df.to_csv('data/processed/processed_data.csv', index=False)
        return final_df


if __name__ == '__main__':
    loader = DataLoader()
    df = loader.process()
    print(f"Processed data saved with shape: {df.shape}")

