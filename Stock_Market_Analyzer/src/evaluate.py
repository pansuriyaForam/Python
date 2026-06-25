"""
Model evaluation utilities for Online News Popularity.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             roc_curve, classification_report)
import warnings

warnings.filterwarnings('ignore')


class ModelEvaluator:
    """
    Comprehensive model evaluation with visualizations.
    """

    def __init__(self, model, X_test, y_test, feature_names=None):
        """
        Initialize evaluator.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            feature_names: List of feature names
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_names
        self.predictions = None
        self.probabilities = None
        self.results = None

    def evaluate(self):
        """Run full evaluation and store results."""
        # Get predictions
        if hasattr(self.model, 'predict'):
            self.predictions = self.model.predict(self.X_test)
        else:
            self.predictions = self.model.predict(self.X_test)

        if hasattr(self.model, 'predict_proba'):
            self.probabilities = self.model.predict_proba(self.X_test)[:, 1]
        else:
            self.probabilities = self.predictions

        # Calculate metrics
        self.results = {
            'accuracy': accuracy_score(self.y_test, self.predictions),
            'precision': precision_score(self.y_test, self.predictions),
            'recall': recall_score(self.y_test, self.predictions),
            'f1_score': f1_score(self.y_test, self.predictions),
            'auc_roc': roc_auc_score(self.y_test, self.probabilities),
            'confusion_matrix': confusion_matrix(self.y_test, self.predictions),
            'classification_report': classification_report(
                self.y_test, self.predictions,
                target_names=['Not Popular', 'Popular']
            )
        }

        return self.results

    def plot_confusion_matrix(self, title='Confusion Matrix'):
        """Plot confusion matrix."""
        if self.results is None:
            self.evaluate()

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(self.results['confusion_matrix'],
                    annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Not Popular', 'Popular'],
                    yticklabels=['Not Popular', 'Popular'],
                    ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(title)
        plt.tight_layout()
        return fig

    def plot_roc_curve(self, title='ROC Curve'):
        """Plot ROC curve."""
        if self.results is None:
            self.evaluate()

        fpr, tpr, _ = roc_curve(self.y_test, self.probabilities)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, label=f'AUC = {self.results["auc_roc"]:.3f}')
        ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)
        ax.legend()
        plt.tight_layout()
        return fig

    def plot_feature_importance(self, top_n=15):
        """Plot feature importance if model supports it."""
        if not hasattr(self.model, 'feature_importances_'):
            print("⚠️ Model does not support feature importance")
            return None

        importances = self.model.feature_importances_
        if self.feature_names is None:
            features = [f'Feature_{i}' for i in range(len(importances))]
        else:
            features = self.feature_names

        importance_df = pd.DataFrame({
            'feature': features,
            'importance': importances
        }).sort_values('importance', ascending=False)

        fig, ax = plt.subplots(figsize=(10, 8))
        top_features = importance_df.head(top_n)
        ax.barh(top_features['feature'], top_features['importance'])
        ax.set_xlabel('Importance')
        ax.set_title(f'Top {top_n} Feature Importances')
        ax.invert_yaxis()
        plt.tight_layout()
        return fig

    def get_summary_table(self):
        """Get summary metrics as DataFrame."""
        if self.results is None:
            self.evaluate()

        summary = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'],
            'Score': [
                self.results['accuracy'],
                self.results['precision'],
                self.results['recall'],
                self.results['f1_score'],
                self.results['auc_roc']
            ]
        })
        return summary

    def print_report(self):
        """Print full evaluation report."""
        if self.results is None:
            self.evaluate()

        print("=" * 60)
        print("MODEL EVALUATION REPORT")
        print("=" * 60)
        print(f"Accuracy:  {self.results['accuracy']:.4f}")
        print(f"Precision: {self.results['precision']:.4f}")
        print(f"Recall:    {self.results['recall']:.4f}")
        print(f"F1-Score:  {self.results['f1_score']:.4f}")
        print(f"AUC-ROC:   {self.results['auc_roc']:.4f}")
        print("\n" + "=" * 60)
        print("Classification Report:")
        print("=" * 60)
        print(self.results['classification_report'])

        print("\nConfusion Matrix:")
        print(self.results['confusion_matrix'])
        print("=" * 60)


# === Convenience Functions ===

def evaluate_model(model, X_test, y_test, feature_names=None):
    """
    Quick model evaluation.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        feature_names: List of feature names

    Returns:
        evaluator: ModelEvaluator object with results
    """
    evaluator = ModelEvaluator(model, X_test, y_test, feature_names)
    evaluator.evaluate()
    return evaluator


def compare_models(models, X_test, y_test, model_names=None):
    """
    Compare multiple models.

    Args:
        models: List of trained models
        X_test: Test features
        y_test: Test labels
        model_names: List of model names

    Returns:
        DataFrame with comparison results
    """
    if model_names is None:
        model_names = [f'Model_{i}' for i in range(len(models))]

    results = []
    for name, model in zip(model_names, models):
        pred = model.predict(X_test) if hasattr(model, 'predict') else model
        proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else pred

        results.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, pred),
            'Precision': precision_score(y_test, pred),
            'Recall': recall_score(y_test, pred),
            'F1-Score': f1_score(y_test, pred),
            'AUC-ROC': roc_auc_score(y_test, proba)
        })

    return pd.DataFrame(results).sort_values('Accuracy', ascending=False)