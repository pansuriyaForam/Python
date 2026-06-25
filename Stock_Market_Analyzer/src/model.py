"""
Model training and prediction pipeline for Online News Popularity.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import warnings

warnings.filterwarnings('ignore')


class NewsPopularityModel:
    """
    Complete model pipeline for predicting article popularity.
    Handles training, evaluation, and prediction.
    """

    def __init__(self, model_type='xgb', threshold=0.51):
        """
        Initialize the model pipeline.

        Args:
            model_type: 'lr', 'rf', or 'xgb'
            threshold: Classification threshold (default 0.51)
        """
        self.model_type = model_type
        self.threshold = threshold
        self.model = None
        self.scaler = None
        self.feature_engineer = None
        self.X_train = None
        self.y_train = None

    def _create_model(self):
        """Create the appropriate model instance."""
        if self.model_type == 'lr':
            return LogisticRegression(max_iter=1000, random_state=42)
        elif self.model_type == 'rf':
            return RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=30,
                min_samples_leaf=15,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'xgb':
            return XGBClassifier(
                n_estimators=500,
                max_depth=5,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.7,
                min_child_weight=5,
                reg_alpha=0.5,
                reg_lambda=2.0,
                gamma=0.3,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train(self, X, y, test_size=0.2):
        """
        Train the model on provided data.

        Args:
            X: Feature matrix (DataFrame or numpy array)
            y: Target labels
            test_size: Proportion for validation split

        Returns:
            Dictionary with training metrics
        """
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Store training data
        self.X_train = X_train
        self.y_train = y_train

        # Create and train model
        self.model = self._create_model()
        self.model.fit(X_train, y_train)

        # Evaluate
        train_pred = self.model.predict(X_train)
        train_proba = self.model.predict_proba(X_train)[:, 1]
        val_pred = self.model.predict(X_val)
        val_proba = self.model.predict_proba(X_val)[:, 1]

        metrics = {
            'train_accuracy': accuracy_score(y_train, train_pred),
            'val_accuracy': accuracy_score(y_val, val_pred),
            'train_auc': roc_auc_score(y_train, train_proba),
            'val_auc': roc_auc_score(y_val, val_proba),
            'val_report': classification_report(y_val, val_pred,
                                                target_names=['Not Popular', 'Popular'])
        }

        return metrics

    def predict(self, X):
        """
        Predict labels for new data.

        Args:
            X: Feature matrix

        Returns:
            Predictions (binary labels)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        proba = self.model.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)

    def predict_proba(self, X):
        """
        Predict probabilities for new data.

        Args:
            X: Feature matrix

        Returns:
            Probability scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test data.

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary with evaluation metrics
        """
        pred = self.predict(X_test)
        proba = self.predict_proba(X_test)

        metrics = {
            'accuracy': accuracy_score(y_test, pred),
            'auc_roc': roc_auc_score(y_test, proba),
            'classification_report': classification_report(y_test, pred,
                                                           target_names=['Not Popular', 'Popular'])
        }

        return metrics

    def save(self, model_path='models/best_model.pkl'):
        """Save the trained model and configuration."""
        if self.model is None:
            raise ValueError("No model to save. Train first.")

        config = {
            'model': self.model,
            'model_type': self.model_type,
            'threshold': self.threshold
        }
        joblib.dump(config, model_path)
        print(f"✅ Model saved to {model_path}")

    def load(self, model_path='models/best_model.pkl'):
        """Load a trained model."""
        config = joblib.load(model_path)
        self.model = config['model']
        self.model_type = config['model_type']
        self.threshold = config['threshold']
        print(f"✅ Model loaded from {model_path}")
        return self


# === Convenience Functions ===

def train_model(X_train, y_train, model_type='xgb'):
    """Train a model quickly."""
    model = NewsPopularityModel(model_type=model_type)
    metrics = model.train(X_train, y_train)
    return model, metrics


def load_best_model():
    """Load the best trained model."""
    model = NewsPopularityModel()
    model.load('models/best_model_advanced.pkl')
    return model


def predict_popularity(df, model=None):
    """
    End-to-end prediction function for new articles.

    Args:
        df: Raw DataFrame with article features
        model: Pre-loaded model (if None, loads best model)

    Returns:
        predictions: Array of 0/1 predictions
        probabilities: Array of probability scores
    """
    from features import FeatureEngineer

    # Load model if not provided
    if model is None:
        model = load_best_model()

    # Initialize feature engineer and load scaler
    engineer = FeatureEngineer()
    engineer.load('models/feature_engineer_scaler')

    # Engineer features
    X_scaled = engineer.transform(df)

    # Predict
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)

    return predictions, probabilities