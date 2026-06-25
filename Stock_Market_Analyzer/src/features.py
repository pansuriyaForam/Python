"""
Feature engineering pipeline for Online News Popularity prediction.
Matches the feature engineering from notebooks 02 and 04.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib


class FeatureEngineer:
    """
    Handles all feature engineering for the Online News Popularity dataset.
    Matches the feature engineering from notebooks 02 and 04.
    """

    def __init__(self):
        self.feature_names = None
        self.scaler = None

    def create_features(self, df):
        """
        Create all engineered features from raw data.
        Matches feature engineering from notebook 02 and 04.
        """
        data = df.copy()

        # 1. Media Features
        data['media_count'] = data['num_imgs'] + data['num_videos']
        data['has_media'] = (data['media_count'] > 0).astype(int)
        data['media_density'] = data['media_count'] / (data['n_tokens_content'] + 1)

        # 2. Content Features
        data['content_length_per_href'] = data['n_tokens_content'] / (data['num_hrefs'] + 1)
        data['href_density'] = data['num_hrefs'] / (data['n_tokens_content'] + 1)
        data['self_reference_ratio'] = data['num_self_hrefs'] / (data['num_hrefs'] + 1)

        # 3. Keyword Features
        data['keyword_score'] = data['kw_avg_avg'] * (data['kw_min_avg'] + data['kw_max_avg']) / 2

        # 4. Topic Features
        topic_cols = ['LDA_00', 'LDA_01', 'LDA_02', 'LDA_03', 'LDA_04']
        data['topic_diversity'] = data[topic_cols].var(axis=1)
        data['dominant_topic'] = data[topic_cols].idxmax(axis=1).str.replace('LDA_', '').astype(int)

        # 5. Sentiment Features
        data['sentiment_volatility'] = data['max_positive_polarity'] - abs(data['min_negative_polarity'])

        return data

    def clean_features(self, df):
        """Remove redundant features - matches notebook 02."""
        data = df.copy()

        features_to_drop = [
            # Identical content features (keep n_non_stop_words)
            'n_unique_tokens', 'n_non_stop_unique_tokens',

            # Original media features (replaced by media_count)
            'num_imgs', 'num_videos',

            # Redundant keyword features (keep keyword_score)
            'kw_avg_avg', 'kw_min_avg', 'kw_max_avg',
            'kw_max_min', 'kw_min_min', 'kw_max_max',

            # Redundant LDA topics (keep LDA_03 which correlates best)
            'LDA_02',

            # Redundant self-reference (keep only average)
            'self_reference_min_shares', 'self_reference_max_shares',

            # Redundant temporal (is_weekend captures this)
            'weekday_is_sunday',

            # Redundant channel features (LDA topics capture this better)
            'data_channel_is_world', 'data_channel_is_bus', 'data_channel_is_tech',

            # Redundant sentiment features
            'rate_positive_words', 'global_rate_positive_words',
            'sentiment_balance', 'sentiment_consistency',
            'abs_title_sentiment_polarity',
        ]

        features_to_drop = [col for col in features_to_drop if col in data.columns]
        data = data.drop(columns=features_to_drop)

        return data

    def fit_scaler(self, X):
        """Fit StandardScaler on training data."""
        self.scaler = StandardScaler()
        self.scaler.fit(X)
        self.feature_names = X.columns.tolist() if hasattr(X, 'columns') else None
        return self.scaler

    def transform(self, X):
        """Scale features using fitted scaler."""
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call fit_scaler first.")
        return self.scaler.transform(X)

    def fit_transform(self, X):
        """Fit scaler and transform features."""
        self.fit_scaler(X)
        return self.transform(X)

    def save(self, path_prefix='models/feature_engineer'):
        """Save the feature engineer."""
        if self.scaler is not None:
            joblib.dump(self.scaler, f'{path_prefix}_scaler.pkl')
        print(f"✅ Feature engineer saved to {path_prefix}_scaler.pkl")

    def load(self, path_prefix='models/feature_engineer'):
        """Load the feature engineer."""
        self.scaler = joblib.load(f'{path_prefix}_scaler.pkl')
        print(f"✅ Feature engineer loaded from {path_prefix}_scaler.pkl")
        return self


def engineer_features(df, fit_scaler=False):
    """Full feature engineering pipeline."""
    engineer = FeatureEngineer()
    df_engineered = engineer.create_features(df)
    df_cleaned = engineer.clean_features(df_engineered)

    if 'shares' in df_cleaned.columns:
        X = df_cleaned.drop(columns=['shares'])
    else:
        X = df_cleaned

    if fit_scaler:
        X_scaled = engineer.fit_transform(X)
    else:
        if engineer.scaler is None:
            raise ValueError("Scaler not loaded. Call engineer.load() first.")
        X_scaled = engineer.transform(X)

    return X_scaled, engineer