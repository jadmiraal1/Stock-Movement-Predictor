import os
import glob
import yaml
import pandas as pd
import logging
from datetime import datetime

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline

# Ensure VADER lexicon is available
nltk.download('vader_lexicon')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

class SentimentProcessor:
    def __init__(self, config_path=None):
        # Load configuration
        here = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(here)
        cfg_path = config_path or os.path.join(project_root, 'configs', 'config.yaml')
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Config not found at {cfg_path}")
        with open(cfg_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Directories for raw text and processed sentiment
        self.raw_dir = os.path.join(project_root, self.config.get('raw_sentiment_text_dir', 'data/raw/sentiment_text'))
        self.output_dir = os.path.join(project_root, self.config.get('sentiment_output_dir', 'data/raw/sentiment'))

        # Sentiment analysis method: 'vader' or 'transformer'
        method_cfg = self.config.get('sentiment_method', {})
        self.method = method_cfg.get('method', 'vader')
        self.model_name = method_cfg.get('model_name')

        # Initialize sentiment model
        if self.method == 'vader':
            self.analyzer = SentimentIntensityAnalyzer()
            logging.info('Using VADER for sentiment analysis')
        elif self.method == 'transformer':
            if not self.model_name:
                raise ValueError("Transformer method requires a 'model_name' in config.yaml under sentiment_method.model_name")
            self.nlp = pipeline('sentiment-analysis', model=self.model_name)
            logging.info(f"Using transformer model {self.model_name} for sentiment analysis")
        else:
            raise ValueError("Unsupported sentiment method. Use 'vader' or 'transformer'.")

    def load_raw(self):
        """
        Load all raw text CSVs from the raw_dir.
        Expects each CSV to have: timestamp (ISO string), ticker, text.
        """
        pattern = os.path.join(self.raw_dir, '*.csv')
        files = glob.glob(pattern)
        if not files:
            raise FileNotFoundError(f"No CSV files found in {self.raw_dir}")

        df_list = []
        for fp in files:
            try:
                df = pd.read_csv(fp, parse_dates=['timestamp'])
                df_list.append(df)
            except Exception as e:
                logging.error(f"Failed to read {fp}: {e}")
        if not df_list:
            raise ValueError("No valid raw files loaded")
        return pd.concat(df_list, ignore_index=True)

    def compute_vader(self, text: str) -> float:
        """
        Compute VADER compound sentiment score for given text.
        """
        vs = self.analyzer.polarity_scores(text)
        return vs['compound']

    def compute_transformer(self, texts: list) -> list:
        """
        Compute sentiment scores using a Transformer pipeline for a list of texts.
        Returns a list of floats (pos_score minus neg_score).
        """
        # Batch inference
        results = self.nlp(texts, truncation=True, padding=True, max_length=512, batch_size=16)
        scores = []
        for res in results:
            label = res.get('label', '').upper()
            score = res.get('score', 0.0)
            if label.startswith('POS'):
                scores.append(score)
            elif label.startswith('NEG'):
                scores.append(-score)
            else:
                scores.append(0.0)
        return scores

    def process(self):
        """
        Full pipeline:
          1. Load raw text
          2. Compute sentiment scores
          3. Aggregate daily mean per ticker and save CSVs
        """
        df = self.load_raw()
        # Verify required columns
        for col in ['timestamp', 'ticker', 'text']:
            if col not in df.columns:
                raise KeyError(f"Raw sentiment data must include '{col}' column")

        # Compute sentiment
        if self.method == 'vader':
            df['sentiment_score'] = df['text'].apply(self.compute_vader)
        else:
            texts = df['text'].tolist()
            df['sentiment_score'] = self.compute_transformer(texts)

        # Round timestamp to date only
        df['Date'] = df['timestamp'].dt.floor('D')

        # Aggregate daily mean sentiment per ticker
        df_daily = (
            df.groupby(['ticker', 'Date'])['sentiment_score']
            .mean()
            .reset_index()
        )

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Save one CSV per ticker
        for ticker, grp in df_daily.groupby('ticker'):
            out_path = os.path.join(self.output_dir, f"{ticker}_sentiment.csv")
            grp.to_csv(out_path, index=False)
            logging.info(f"Wrote {out_path} with {len(grp)} daily records")

        return df_daily

if __name__ == '__main__':
    sp = SentimentProcessor()
    daily_df = sp.process()
    print(f"Processed {len(daily_df)} daily sentiment records")
