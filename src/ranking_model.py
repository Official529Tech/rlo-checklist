"""
Stage 2: Ranking Model Training

This module trains a learning-to-rank model to predict item priority
using LambdaMART (via LightGBM) or pairwise logistic regression.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
import pickle
from collections import defaultdict

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from scipy.stats import kendalltau, spearmanr

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("Warning: LightGBM not installed. Using pairwise logistic regression instead.")

import warnings
warnings.filterwarnings('ignore')


def ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = None) -> float:
    """
    Compute Normalized Discounted Cumulative Gain.

    Args:
        y_true: True relevance scores (higher = better)
        y_pred: Predicted scores
        k: Cutoff (None = all items)

    Returns:
        NDCG score
    """
    if len(y_true) == 0:
        return 0.0

    if k is None:
        k = len(y_true)

    # Limit k to actual number of items
    k = min(k, len(y_true))

    # Sort by predicted scores
    order = np.argsort(y_pred)[::-1]
    y_true_sorted = y_true[order]

    # DCG
    gains = 2 ** y_true_sorted[:k] - 1
    discounts = np.log2(np.arange(k) + 2)
    dcg = np.sum(gains / discounts)

    # Ideal DCG
    ideal_order = np.argsort(y_true)[::-1]
    ideal_gains = 2 ** y_true[ideal_order][:k] - 1
    idcg = np.sum(ideal_gains / discounts)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def mean_average_precision(y_true: np.ndarray, y_pred: np.ndarray,
                           threshold: float = 0.5) -> float:
    """
    Compute Mean Average Precision.

    Args:
        y_true: True relevance scores
        y_pred: Predicted scores
        threshold: Relevance threshold for "relevant"

    Returns:
        MAP score
    """
    order = np.argsort(y_pred)[::-1]
    y_true_sorted = y_true[order]

    relevant = y_true_sorted >= threshold
    if not relevant.any():
        return 0.0

    precision_at_k = np.cumsum(relevant) / (np.arange(len(relevant)) + 1)
    return np.sum(precision_at_k * relevant) / relevant.sum()


def pairwise_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute pairwise ranking accuracy.

    Args:
        y_true: True relevance scores
        y_pred: Predicted scores

    Returns:
        Fraction of correctly ordered pairs
    """
    n = len(y_true)
    if n < 2:
        return 1.0

    correct = 0
    total = 0

    for i in range(n):
        for j in range(i + 1, n):
            if y_true[i] != y_true[j]:
                total += 1
                if (y_true[i] > y_true[j]) == (y_pred[i] > y_pred[j]):
                    correct += 1

    return correct / total if total > 0 else 1.0


class RankingModel:
    """
    Stage 2: Ranking Model for predicting item priority.

    Uses LambdaMART (LightGBM) for learning to rank, or falls back
    to pairwise logistic regression if LightGBM unavailable.
    """

    def __init__(self, model_type: str = 'lambdamart'):
        """
        Initialize ranking model.

        Args:
            model_type: 'lambdamart' (LightGBM) or 'pairwise' (logistic)
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.is_fitted = False
        self.metrics: Dict = {}

        if model_type == 'lambdamart' and not HAS_LIGHTGBM:
            print("LightGBM not available, falling back to pairwise logistic regression")
            self.model_type = 'pairwise'

    def fit(self, X: np.ndarray, relevance: np.ndarray, groups: np.ndarray,
            feature_names: List[str] = None,
            X_pairwise: np.ndarray = None,
            y_pairwise: np.ndarray = None) -> 'RankingModel':
        """
        Fit the ranking model.

        Args:
            X: Feature matrix (n_samples, n_features)
            relevance: Relevance scores (n_samples,)
            groups: Group sizes for each query (n_queries,)
            feature_names: Feature names
            X_pairwise: Pairwise difference features (for pairwise model)
            y_pairwise: Pairwise labels (for pairwise model)

        Returns:
            self
        """
        self.feature_names = feature_names or [f'f{i}' for i in range(X.shape[1])]

        print(f"\n{'='*60}")
        print("Training Ranking Model (Stage 2)")
        print(f"{'='*60}")
        print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")
        print(f"Groups: {len(groups)}, Avg group size: {groups.mean():.1f}")
        print(f"Model type: {self.model_type}")

        if self.model_type == 'lambdamart':
            self._fit_lambdamart(X, relevance, groups)
        else:
            if X_pairwise is None or y_pairwise is None:
                raise ValueError("Pairwise model requires X_pairwise and y_pairwise")
            self._fit_pairwise(X_pairwise, y_pairwise, X, relevance, groups)

        self.is_fitted = True
        return self

    def _fit_lambdamart(self, X: np.ndarray, relevance: np.ndarray,
                        groups: np.ndarray):
        """Fit LambdaMART model using LightGBM."""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Convert relevance to integer grades for LightGBM
        # Scale to 0-4 range
        relevance_grades = np.round(relevance * 4).astype(int)

        # Create dataset
        train_data = lgb.Dataset(X_scaled, label=relevance_grades, group=groups)

        # Parameters
        params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'ndcg_eval_at': [5, 10, 20],
            'num_leaves': 31,
            'max_depth': 4,
            'learning_rate': 0.05,
            'lambda_l1': 0.1,
            'lambda_l2': 1.0,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }

        # Cross-validation
        print("\nRunning cross-validation...")
        n_folds = min(5, len(set(groups))) if groups is not None else 5
        cv_results = lgb.cv(
            params,
            train_data,
            num_boost_round=200,
            nfold=n_folds,
            callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)],
            return_cvbooster=True
        )

        # Get best iteration
        best_iteration = len(cv_results['valid ndcg@5-mean'])

        print(f"Best iteration: {best_iteration}")
        print(f"CV NDCG@5: {cv_results['valid ndcg@5-mean'][-1]:.4f} ± {cv_results['valid ndcg@5-stdv'][-1]:.4f}")
        print(f"CV NDCG@10: {cv_results['valid ndcg@10-mean'][-1]:.4f} ± {cv_results['valid ndcg@10-stdv'][-1]:.4f}")

        # Train final model
        print("\nTraining final model...")
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=best_iteration
        )

        # Store CV metrics
        self.metrics['cv_ndcg5'] = cv_results['valid ndcg@5-mean'][-1]
        self.metrics['cv_ndcg10'] = cv_results['valid ndcg@10-mean'][-1]
        self.metrics['cv_ndcg20'] = cv_results['valid ndcg@20-mean'][-1]

        # Compute additional metrics on full data
        self._compute_metrics(X_scaled, relevance, groups)

    def _fit_pairwise(self, X_pairwise: np.ndarray, y_pairwise: np.ndarray,
                      X: np.ndarray, relevance: np.ndarray, groups: np.ndarray):
        """Fit pairwise logistic regression model."""
        print(f"\nTraining on {len(y_pairwise)} pairs...")

        # Scale pairwise features
        X_pairwise_scaled = self.scaler.fit_transform(X_pairwise)

        # Train logistic regression
        self.model = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42
        )
        self.model.fit(X_pairwise_scaled, y_pairwise)

        # Compute training accuracy
        train_acc = self.model.score(X_pairwise_scaled, y_pairwise)
        print(f"Pairwise training accuracy: {train_acc:.4f}")

        self.metrics['pairwise_accuracy'] = train_acc

        # Compute ranking metrics
        self._compute_metrics(X, relevance, groups)

    def _compute_metrics(self, X: np.ndarray, relevance: np.ndarray,
                         groups: np.ndarray):
        """Compute evaluation metrics."""
        # Get predictions
        if self.model_type == 'lambdamart':
            X_scaled = self.scaler.transform(X)
            y_pred = self.model.predict(X_scaled)
        else:
            # For pairwise model, compute scores differently
            # Use dot product with coefficients as proxy for relevance
            y_pred = self.predict(X)

        # Compute metrics per group
        ndcg5_scores = []
        ndcg10_scores = []
        ndcg20_scores = []
        map_scores = []
        pairwise_scores = []
        kendall_scores = []

        idx = 0
        for group_size in groups:
            if group_size < 2:
                idx += group_size
                continue

            group_rel = relevance[idx:idx + group_size]
            group_pred = y_pred[idx:idx + group_size]

            ndcg5_scores.append(ndcg_at_k(group_rel, group_pred, k=5))
            ndcg10_scores.append(ndcg_at_k(group_rel, group_pred, k=10))
            ndcg20_scores.append(ndcg_at_k(group_rel, group_pred, k=20))
            map_scores.append(mean_average_precision(group_rel, group_pred))
            pairwise_scores.append(pairwise_accuracy(group_rel, group_pred))

            tau, _ = kendalltau(group_rel, group_pred)
            if not np.isnan(tau):
                kendall_scores.append(tau)

            idx += group_size

        self.metrics['ndcg5'] = np.mean(ndcg5_scores)
        self.metrics['ndcg10'] = np.mean(ndcg10_scores)
        self.metrics['ndcg20'] = np.mean(ndcg20_scores)
        self.metrics['map'] = np.mean(map_scores)
        self.metrics['pairwise_accuracy_eval'] = np.mean(pairwise_scores)
        self.metrics['kendall_tau'] = np.mean(kendall_scores)

        print(f"\n=== Evaluation Metrics ===")
        print(f"NDCG@5:  {self.metrics['ndcg5']:.4f}")
        print(f"NDCG@10: {self.metrics['ndcg10']:.4f}")
        print(f"NDCG@20: {self.metrics['ndcg20']:.4f}")
        print(f"MAP:     {self.metrics['map']:.4f}")
        print(f"Pairwise Accuracy: {self.metrics['pairwise_accuracy_eval']:.4f}")
        print(f"Kendall's τ: {self.metrics['kendall_tau']:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict relevance scores.

        Args:
            X: Feature matrix

        Returns:
            Predicted scores (higher = more relevant/higher priority)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X_scaled = self.scaler.transform(X)

        if self.model_type == 'lambdamart':
            return self.model.predict(X_scaled)
        else:
            # For pairwise model, use logistic regression coefficients
            # to score items directly
            return self.model.decision_function(X_scaled)

    def rank_items(self, X: np.ndarray, item_ids: List[str] = None) -> List[Tuple[str, float]]:
        """
        Rank items by predicted score.

        Args:
            X: Feature matrix
            item_ids: Optional item IDs

        Returns:
            List of (item_id, score) tuples sorted by score descending
        """
        scores = self.predict(X)

        if item_ids is None:
            item_ids = [f'item_{i}' for i in range(len(scores))]

        ranked = sorted(zip(item_ids, scores), key=lambda x: -x[1])
        return ranked

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance."""
        if not self.is_fitted:
            raise ValueError("Model not fitted.")

        if self.model_type == 'lambdamart':
            importance = self.model.feature_importance(importance_type='gain')
        else:
            importance = np.abs(self.model.coef_[0])

        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        return df.head(top_n)

    def save(self, path: str):
        """Save model to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save model
        if self.model_type == 'lambdamart':
            self.model.save_model(str(path / 'ranking_model.lgb'))
        else:
            with open(path / 'ranking_model.pkl', 'wb') as f:
                pickle.dump(self.model, f)

        # Save scaler
        with open(path / 'ranking_scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)

        # Save metadata
        metadata = {
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'metrics': self.metrics,
            'is_fitted': self.is_fitted
        }
        with open(path / 'ranking_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\nModel saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'RankingModel':
        """Load model from disk."""
        path = Path(path)

        with open(path / 'ranking_metadata.json', 'r') as f:
            metadata = json.load(f)

        instance = cls(model_type=metadata['model_type'])
        instance.feature_names = metadata['feature_names']
        instance.metrics = metadata['metrics']
        instance.is_fitted = metadata['is_fitted']

        if instance.model_type == 'lambdamart':
            instance.model = lgb.Booster(model_file=str(path / 'ranking_model.lgb'))
        else:
            with open(path / 'ranking_model.pkl', 'rb') as f:
                instance.model = pickle.load(f)

        with open(path / 'ranking_scaler.pkl', 'rb') as f:
            instance.scaler = pickle.load(f)

        return instance


def train_ranking_model(X: np.ndarray, relevance: np.ndarray, groups: np.ndarray,
                        feature_names: List[str],
                        X_pairwise: np.ndarray = None,
                        y_pairwise: np.ndarray = None,
                        output_dir: str = None,
                        model_type: str = 'lambdamart') -> RankingModel:
    """
    Train and evaluate ranking model.

    Args:
        X: Feature matrix
        relevance: Relevance scores
        groups: Group sizes
        feature_names: Feature names
        X_pairwise: Pairwise features (for pairwise model)
        y_pairwise: Pairwise labels (for pairwise model)
        output_dir: Directory to save model
        model_type: Model type

    Returns:
        Trained RankingModel
    """
    model = RankingModel(model_type=model_type)

    model.fit(X, relevance, groups, feature_names,
              X_pairwise=X_pairwise, y_pairwise=y_pairwise)

    # Feature importance
    print("\n=== Top Feature Importance ===")
    importance_df = model.get_feature_importance(top_n=20)
    print(importance_df.to_string(index=False))

    # Save if output_dir specified
    if output_dir:
        model.save(output_dir)
        importance_df.to_csv(Path(output_dir) / 'ranking_feature_importance.csv', index=False)

    return model


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    from data_loader import load_data
    from feature_engineering import create_feature_datasets

    base_dir = Path(__file__).parent.parent
    data_dir = str(base_dir / 'Data')
    output_dir = str(base_dir / 'Models')

    # Load data
    loader = load_data(data_dir)

    # Create features
    datasets = create_feature_datasets(loader)
    X, relevance, groups, examples = datasets['ranking']
    X_pairwise, y_pairwise, _ = datasets['pairwise']
    feature_names = datasets['feature_names']['ranking']

    # Train model
    model_type = 'lambdamart' if HAS_LIGHTGBM else 'pairwise'
    model = train_ranking_model(
        X, relevance, groups, feature_names,
        X_pairwise=X_pairwise, y_pairwise=y_pairwise,
        output_dir=output_dir,
        model_type=model_type
    )
