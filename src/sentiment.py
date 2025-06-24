import os
import glob
import yaml
import pandas as pd
from datetime import datetime

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

nltk.download('vader_lexicon')


class SentimentProcessor:
     
    def __init__(self, config_path="config/config.yaml"):
        # Load configuration
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Directories for raw text and processed sentiment
        self.raw_dir = self.config.get("sentiment_raw_dir", "data/raw/sentiment_text")
        self.output_dir = self.config.get("sentiment_output_dir", "data/raw/sentiment")

        # Sentiment analysis method: 'vader' or 'transformer'
        method_cfg = self.config.get("sentiment_method", {})
        self.method = method_cfg.get("method", "vader")  # default to Vader
        self.model_name = method_cfg.get("model_name", None)

        if self.method == "vader":
            self.analyzer = SentimentIntensityAnalyzer()
        elif self.method == "transformer":
            if not self.model_name:
                raise ValueError("Transformer method requires a 'model_name' in config.yaml under sentiment_method.model_name")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.eval()
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
            df = pd.read_csv(fp, parse_dates=['timestamp'])
            df_list.append(df)
        return pd.concat(df_list, ignore_index=True)

    def compute_vader(self, text: str) -> float:
        """
        Compute VADER compound sentiment score for given text.
        """
        vs = self.analyzer.polarity_scores(text)
        return vs['compound']

    def compute_transformer(self, text: str) -> float:
        """
        Compute sentiment score using a Transformer model (e.g., FinBERT).
        Returns (pos_prob - neg_prob) as a single float.
        """
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
            # Label ordering depends on model; common: 0=neg,1=neu,2=pos
            neg, neu, pos = probs.tolist()
            return pos - neg

    def process(self):
        """
        Full pipeline:
          1. Load raw text
          2. Compute sentiment scores
          3. Save per-ticker CSVs to output_dir
        """
        df = self.load_raw()
        # Verify required columns
        for col in ['timestamp', 'ticker', 'text']:
            if col not in df.columns:
                raise KeyError(f"Raw sentiment data must include '{col}' column")

        # Compute sentiment
        if self.method == "vader":
            df['sentiment_score'] = df['text'].apply(self.compute_vader)
        else:
            df['sentiment_score'] = df['text'].apply(self.compute_transformer)

        # Prepare output
        os.makedirs(self.output_dir, exist_ok=True)
        # Round timestamp to date only
        df['Date'] = df['timestamp'].dt.floor('D')

        # Save one CSV per ticker
        for ticker, grp in df.groupby('ticker'):
            out_path = os.path.join(self.output_dir, f"{ticker}_sentiment.csv")
            grp[['timestamp', 'ticker', 'sentiment_score']].to_csv(out_path, index=False)

        # Also return full DataFrame
        return df


if __name__ == '__main__':
    sp = SentimentProcessor()
    full_df = sp.process()
    print(f"Processed sentiment entries: {full_df.shape[0]}")

