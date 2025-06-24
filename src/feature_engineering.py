import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class FeatureEngineer:
    """
    FeatureEngineer loads processed data, computes additional features,
    scales them, and splits into train/validation/test sets.
    """
    def __init__(self, config_path="config/config.yaml"):
        self.config = load_config(config_path)
        self.data_path = self.config.get("processed_data_path", "data/processed/processed_data.csv")
        # Feature columns (basic defaults)
        self.price_feats = self.config.get("price_features", ["return", "ma10", "ma50"])
        self.sent_feat = self.config.get("sentiment_feature", "sentiment_score")
        self.feature_cols = self.price_feats + [self.sent_feat]
        # Split ratios
        self.test_size = self.config.get("test_size", 0.2)
        self.val_size = self.config.get("val_size", 0.1)
        # Random seed for reproducibility
        self.random_state = self.config.get("random_state", 42)

    def load_data(self):
        df = pd.read_csv(self.data_path, parse_dates=["Date"] )
        # Drop rows with NaNs in features or label
        df = df.dropna(subset=self.feature_cols + ['label'])
        return df

    def scale_features(self, X_train, X_val, X_test):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_val_scaled, X_test_scaled, scaler

    def split_data(self, df):
        # Extract features and labels
        X = df[self.feature_cols].values
        y = df['label'].values

        # Initial train+val vs test split
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=self.test_size, shuffle=False
        )
        # Further split train into train and val
        val_ratio = self.val_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_ratio, shuffle=False
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    def process(self):
        # Load
        df = self.load_data()
        # Split
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(df)
        # Scale
        X_train_s, X_val_s, X_test_s, scaler = self.scale_features(X_train, X_val, X_test)

        # Create output directory
        out_dir = os.path.dirname(self.data_path)
        # Save splits as numpy files
        os.makedirs(out_dir, exist_ok=True)
        # Using pandas for easy load later
        pd.DataFrame(X_train_s, columns=self.feature_cols).to_csv(os.path.join(out_dir, 'X_train.csv'), index=False)
        pd.DataFrame(X_val_s, columns=self.feature_cols).to_csv(os.path.join(out_dir, 'X_val.csv'), index=False)
        pd.DataFrame(X_test_s, columns=self.feature_cols).to_csv(os.path.join(out_dir, 'X_test.csv'), index=False)
        pd.DataFrame(y_train, columns=['label']).to_csv(os.path.join(out_dir, 'y_train.csv'), index=False)
        pd.DataFrame(y_val, columns=['label']).to_csv(os.path.join(out_dir, 'y_val.csv'), index=False)
        pd.DataFrame(y_test, columns=['label']).to_csv(os.path.join(out_dir, 'y_test.csv'), index=False)

        # Return as dict
        return {
            'X_train': X_train_s,
            'X_val': X_val_s,
            'X_test': X_test_s,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'scaler': scaler
        }


if __name__ == '__main__':
    fe = FeatureEngineer()
    splits = fe.process()
    print("Feature engineering complete.")
    print("Shapes:", {k: v.shape for k, v in splits.items() if hasattr(v, 'shape')})

