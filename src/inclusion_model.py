"""
Stage 1: Inclusion Model Training

This module trains a model to predict P(item included | context, provenance)
using regularized logistic regression.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
import pickle
from collections import defaultdict

from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import (
    cross_val_score, cross_val_predict,
    StratifiedKFold, GroupKFold, LeaveOneGroupOut
)
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, average_precision_score,
    classification_report, confusion_matrix, f1_score,
    precision_score, recall_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

import warnings
warnings.filterwarnings('ignore')


class InclusionModel:
    """
    Stage 1: Inclusion Model for predicting item inclusion probability.

    Predicts P(include | item, context, provenance) using regularized
    logistic regression for interpretability.
    """

    def __init__(self, model_type: str = 'logistic'):
        """
        Initialize inclusion model.

        Args:
            model_type: 'logistic', 'rf' (random forest), or 'gbm'
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.is_fitted = False
        self.metrics: Dict = {}

    def fit(self, X: np.ndarray, y: np.ndarray,
            feature_names: List[str] = None,
            groups: np.ndarray = None,
            cv_type: str = 'stratified') -> 'InclusionModel':
        """
        Fit the inclusion model with cross-validation.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Binary labels (n_samples,)
            feature_names: Names for each feature
            groups: Group labels for GroupKFold (e.g., user_ids)
            cv_type: 'stratified', 'group', or 'logo' (leave-one-group-out)

        Returns:
            self
        """
        self.feature_names = feature_names or [f'f{i}' for i in range(X.shape[1])]

        print(f"\n{'='*60}")
        print("Training Inclusion Model (Stage 1)")
        print(f"{'='*60}")
        print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")
        print(f"Positive rate: {y.mean():.1%}")
        print(f"Model type: {self.model_type}")

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Create model
        if self.model_type == 'logistic':
            self.model = LogisticRegressionCV(
                Cs=10,
                cv=5,
                penalty='l2',
                scoring='roc_auc',
                class_weight='balanced',
                max_iter=1000,
                random_state=42
            )
        elif self.model_type == 'rf':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'gbm':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Cross-validation setup — cap folds at number of unique groups
        n_folds = 5
        if groups is not None:
            n_folds = min(5, len(set(groups)))
        if cv_type == 'stratified':
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        elif cv_type == 'group' and groups is not None:
            cv = GroupKFold(n_splits=n_folds)
        elif cv_type == 'logo' and groups is not None:
            cv = LeaveOneGroupOut()
        else:
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        # Perform cross-validation
        print("\nCross-validation...")

        if groups is not None and cv_type in ['group', 'logo']:
            cv_scores = cross_val_score(self.model, X_scaled, y, cv=cv, groups=groups, scoring='roc_auc')
            y_pred_proba = cross_val_predict(self.model, X_scaled, y, cv=cv, groups=groups, method='predict_proba')[:, 1]
        else:
            cv_scores = cross_val_score(self.model, X_scaled, y, cv=cv, scoring='roc_auc')
            y_pred_proba = cross_val_predict(self.model, X_scaled, y, cv=cv, method='predict_proba')[:, 1]

        print(f"CV AUC-ROC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # Fit on full data
        print("\nFitting on full dataset...")
        self.model.fit(X_scaled, y)
        self.is_fitted = True

        # Compute metrics
        self._compute_metrics(y, y_pred_proba, cv_scores)

        return self

    def _compute_metrics(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                         cv_scores: np.ndarray):
        """Compute and store evaluation metrics."""
        # Optimal threshold
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5

        y_pred = (y_pred_proba >= optimal_threshold).astype(int)

        self.metrics = {
            'cv_auc_mean': cv_scores.mean(),
            'cv_auc_std': cv_scores.std(),
            'cv_auc_scores': cv_scores.tolist(),
            'auc_roc': roc_auc_score(y_true, y_pred_proba),
            'avg_precision': average_precision_score(y_true, y_pred_proba),
            'optimal_threshold': optimal_threshold,
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }

        print(f"\n=== Evaluation Metrics ===")
        print(f"AUC-ROC: {self.metrics['auc_roc']:.4f}")
        print(f"Average Precision: {self.metrics['avg_precision']:.4f}")
        print(f"Optimal Threshold: {self.metrics['optimal_threshold']:.3f}")
        print(f"Precision: {self.metrics['precision']:.4f}")
        print(f"Recall: {self.metrics['recall']:.4f}")
        print(f"F1 Score: {self.metrics['f1']:.4f}")

        cm = self.metrics['confusion_matrix']
        print(f"\nConfusion Matrix:")
        print(f"  TN: {cm[0][0]:5d}  FP: {cm[0][1]:5d}")
        print(f"  FN: {cm[1][0]:5d}  TP: {cm[1][1]:5d}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict inclusion probability."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = None) -> np.ndarray:
        """Predict inclusion (binary)."""
        if threshold is None:
            threshold = self.metrics.get('optimal_threshold', 0.5)
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance/coefficients."""
        if not self.is_fitted:
            raise ValueError("Model not fitted.")

        if self.model_type == 'logistic':
            importance = self.model.coef_[0]
        elif self.model_type in ['rf', 'gbm']:
            importance = self.model.feature_importances_
        else:
            importance = np.zeros(len(self.feature_names))

        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance,
            'abs_importance': np.abs(importance)
        }).sort_values('abs_importance', ascending=False)

        return df.head(top_n)

    def analyze_by_criticality(self, X: np.ndarray, y: np.ndarray,
                               criticality_scores: np.ndarray) -> Dict:
        """Analyze performance by item criticality."""
        y_pred_proba = self.predict_proba(X)

        results = {}
        for crit_name, crit_val in [('mandatory', 3), ('recommended', 2), ('optional', 1)]:
            mask = criticality_scores == crit_val / 3.0  # Normalized
            if mask.sum() > 0:
                results[crit_name] = {
                    'n_samples': int(mask.sum()),
                    'acceptance_rate': float(y[mask].mean()),
                    'auc_roc': float(roc_auc_score(y[mask], y_pred_proba[mask])) if y[mask].std() > 0 else None,
                    'avg_pred_proba': float(y_pred_proba[mask].mean())
                }

        print("\n=== Performance by Criticality ===")
        for crit, stats in results.items():
            auc_str = f"{stats['auc_roc']:.3f}" if stats['auc_roc'] is not None else 'N/A'
            print(f"{crit}: n={stats['n_samples']}, "
                  f"acceptance={stats['acceptance_rate']:.1%}, "
                  f"AUC={auc_str}")

        return results

    def analyze_by_rule_family(self, X: np.ndarray, y: np.ndarray,
                               feature_names: List[str]) -> Dict:
        """Analyze acceptance rates by rule family."""
        y_pred_proba = self.predict_proba(X)

        results = {}
        for i, fname in enumerate(feature_names):
            if fname.startswith('rule_family_'):
                family = fname.replace('rule_family_', '')
                mask = X[:, i] > 0
                if mask.sum() > 0:
                    results[family] = {
                        'n_samples': int(mask.sum()),
                        'acceptance_rate': float(y[mask].mean()),
                        'avg_pred_proba': float(y_pred_proba[mask].mean())
                    }

        print("\n=== Acceptance by Rule Family ===")
        for family, stats in sorted(results.items(), key=lambda x: -x[1]['acceptance_rate']):
            print(f"{family}: n={stats['n_samples']}, "
                  f"acceptance={stats['acceptance_rate']:.1%}")

        return results

    def save(self, path: str):
        """Save model to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save model
        with open(path / 'inclusion_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)

        # Save scaler
        with open(path / 'inclusion_scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)

        # Save metadata
        metadata = {
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'metrics': self.metrics,
            'is_fitted': self.is_fitted
        }
        with open(path / 'inclusion_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\nModel saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'InclusionModel':
        """Load model from disk."""
        path = Path(path)

        with open(path / 'inclusion_metadata.json', 'r') as f:
            metadata = json.load(f)

        instance = cls(model_type=metadata['model_type'])
        instance.feature_names = metadata['feature_names']
        instance.metrics = metadata['metrics']
        instance.is_fitted = metadata['is_fitted']

        with open(path / 'inclusion_model.pkl', 'rb') as f:
            instance.model = pickle.load(f)

        with open(path / 'inclusion_scaler.pkl', 'rb') as f:
            instance.scaler = pickle.load(f)

        return instance


def train_inclusion_model(X: np.ndarray, y: np.ndarray,
                          feature_names: List[str],
                          user_ids: np.ndarray = None,
                          output_dir: str = None,
                          model_type: str = 'logistic') -> InclusionModel:
    """
    Train and evaluate inclusion model.

    Args:
        X: Feature matrix
        y: Binary labels
        feature_names: Feature names
        user_ids: User IDs for group-based CV
        output_dir: Directory to save model and results
        model_type: Model type ('logistic', 'rf', 'gbm')

    Returns:
        Trained InclusionModel
    """
    model = InclusionModel(model_type=model_type)

    # Fit with group-based CV if user_ids provided
    model.fit(X, y, feature_names=feature_names,
              groups=user_ids,
              cv_type='group' if user_ids is not None else 'stratified')

    # Feature importance
    print("\n=== Top Feature Importance ===")
    importance_df = model.get_feature_importance(top_n=20)
    print(importance_df.to_string(index=False))

    # Analyze by criticality (feature index 2 is criticality_score)
    if 'criticality_score' in feature_names:
        crit_idx = feature_names.index('criticality_score')
        model.analyze_by_criticality(X, y, X[:, crit_idx])

    # Analyze by rule family
    model.analyze_by_rule_family(X, y, feature_names)

    # Save if output_dir specified
    if output_dir:
        model.save(output_dir)

        # Save feature importance
        importance_df.to_csv(Path(output_dir) / 'feature_importance.csv', index=False)

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
    X, y, examples = datasets['inclusion']
    feature_names = datasets['feature_names']['inclusion']

    # Get user IDs for group CV
    user_ids = np.array([ex.user_id for ex in examples])

    # Train model
    model = train_inclusion_model(
        X, y, feature_names,
        user_ids=user_ids,
        output_dir=output_dir,
        model_type='logistic'
    )
