"""
Combined Utility Model for Preference-Aware Optimization

This module combines:
1. Inclusion model predictions (Stage 1)
2. Ranking model predictions (Stage 2)
3. Symbolic priors (safety/regulatory boosts)

Into a unified utility function for constraint-aware optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
from dataclasses import dataclass
from collections import defaultdict

from data_loader import DataLoader, TripData
from feature_engineering import FeatureExtractor
from inclusion_model import InclusionModel
from ranking_model import RankingModel, ndcg_at_k, mean_average_precision, pairwise_accuracy

from scipy.stats import kendalltau


@dataclass
class ScoredItem:
    """Item with combined utility score."""
    item_id: str
    name: str
    category: str
    criticality: str

    # Component scores
    p_include: float  # Inclusion probability
    s_rank: float  # Ranking score (normalized)
    u_symbolic: float  # Symbolic prior

    # Combined utility
    utility: float

    # Metadata
    is_regulatory: bool = False
    is_safety_critical: bool = False
    reasons: List[str] = None


class CombinedPreferenceModel:
    """
    Combined preference model for packing list optimization.

    Computes: u_i(c_t) = P_include(i) × s_rank(i) × u_symbolic(i)
    """

    def __init__(self,
                 inclusion_model: InclusionModel,
                 ranking_model: RankingModel,
                 data_loader: DataLoader):
        """
        Initialize combined model.

        Args:
            inclusion_model: Trained Stage 1 model
            ranking_model: Trained Stage 2 model
            data_loader: Data loader with item/rule info
        """
        self.inclusion_model = inclusion_model
        self.ranking_model = ranking_model
        self.loader = data_loader
        self.feature_extractor = FeatureExtractor(data_loader)

        # Symbolic prior weights
        self.regulatory_boost = 10.0
        self.safety_boost = 5.0
        self.default_prior = 1.0

    def compute_symbolic_prior(self, item: Dict, trip: TripData) -> Tuple[float, bool, bool]:
        """
        Compute symbolic prior for an item.

        Args:
            item: Item data
            trip: Trip context

        Returns:
            (u_symbolic, is_regulatory, is_safety_critical)
        """
        item_id = item['item_id']
        criticality = item.get('criticality', 'recommended')

        # Check if regulatory-required
        is_regulatory = False
        is_safety = False

        # Check rule activations for this item
        for rule_id in trip.rule_activations.keys():
            family = self.loader.get_rule_family(rule_id)
            rule = self.loader.rules.get(rule_id, {})

            # Check if this rule emits the item
            then_clause = rule.get('then', {})
            emit_items = then_clause.get('emit_items', [])

            for emitted in emit_items:
                emitted_id = emitted.get('item_id') if isinstance(emitted, dict) else emitted
                if emitted_id == item_id:
                    # Check flags
                    flags = emitted.get('flags', []) if isinstance(emitted, dict) else []
                    if 'safety_critical' in flags or 'regulatory' in flags:
                        is_regulatory = True
                    if 'safety' in flags:
                        is_safety = True

            if family == 'regulatory':
                is_regulatory = True

        # Also check criticality
        if criticality == 'mandatory':
            is_safety = True

        # Compute prior
        if is_regulatory and criticality == 'mandatory':
            return self.regulatory_boost, True, True
        elif is_safety or is_regulatory:
            return self.safety_boost, is_regulatory, is_safety
        else:
            return self.default_prior, False, False

    def score_items(self, trip: TripData) -> List[ScoredItem]:
        """
        Score all seed items for a trip.

        Args:
            trip: Trip with seed items

        Returns:
            List of ScoredItem sorted by utility (descending)
        """
        scored_items = []

        # Get context features
        context_features = self.feature_extractor._extract_context_features(trip)

        # Score each seed item
        for item in trip.seed_items:
            item_id = item['item_id']

            # Extract features
            item_features = self.feature_extractor._extract_item_features(item, trip)
            features = np.concatenate([item_features, context_features]).reshape(1, -1)

            # Inclusion probability
            p_include = float(self.inclusion_model.predict_proba(features)[0])

            # Ranking score (normalize to [0, 1])
            s_rank_raw = float(self.ranking_model.predict(features)[0])
            # Use sigmoid to normalize
            s_rank = 1.0 / (1.0 + np.exp(-s_rank_raw / 10.0))

            # Symbolic prior
            u_symbolic, is_regulatory, is_safety = self.compute_symbolic_prior(item, trip)

            # Combined utility
            utility = p_include * s_rank * u_symbolic

            # Get item info
            catalog_info = self.loader.get_item_info(item_id)

            scored_items.append(ScoredItem(
                item_id=item_id,
                name=item.get('name', catalog_info.get('item', item_id)),
                category=catalog_info.get('category_code', item_id.split('-')[0]),
                criticality=item.get('criticality', 'recommended'),
                p_include=p_include,
                s_rank=s_rank,
                u_symbolic=u_symbolic,
                utility=utility,
                is_regulatory=is_regulatory,
                is_safety_critical=is_safety,
                reasons=item.get('reasons', [])
            ))

        # Sort by utility
        scored_items.sort(key=lambda x: -x.utility)

        return scored_items

    def generate_list(self, trip: TripData,
                      max_items: int = None,
                      include_threshold: float = 0.3) -> List[ScoredItem]:
        """
        Generate optimized packing list.

        Args:
            trip: Trip context
            max_items: Maximum items to include
            include_threshold: Minimum inclusion probability

        Returns:
            Filtered and ranked list of items
        """
        scored = self.score_items(trip)

        # Filter by inclusion probability (but always keep regulatory)
        filtered = [
            item for item in scored
            if item.p_include >= include_threshold or item.is_regulatory
        ]

        # Limit to max_items if specified
        if max_items is not None:
            filtered = filtered[:max_items]

        return filtered

    def evaluate_on_trip(self, trip: TripData) -> Dict:
        """
        Evaluate model predictions against user's actual selections.

        Args:
            trip: Trip with both seed and final items

        Returns:
            Evaluation metrics
        """
        # Get predictions
        scored = self.score_items(trip)
        pred_scores = {item.item_id: item.utility for item in scored}
        pred_include_proba = {item.item_id: item.p_include for item in scored}

        # Get ground truth
        def get_priority(item, default=999):
            p = item.get('priority', default)
            try:
                return int(p)
            except (TypeError, ValueError):
                return default

        final_ids = {item['item_id'] for item in trip.final_items}
        final_ranks = {item['item_id']: get_priority(item, 999) for item in trip.final_items}

        # Inclusion metrics
        y_true_include = np.array([1 if item.item_id in final_ids else 0 for item in scored])
        y_pred_include = np.array([pred_include_proba[item.item_id] for item in scored])

        # Ranking metrics (only for kept items)
        kept_items = [item for item in scored if item.item_id in final_ids]
        if len(kept_items) >= 2:
            # True relevance (inverse of rank)
            max_rank = max(final_ranks[item.item_id] for item in kept_items)
            y_true_rank = np.array([
                (max_rank - final_ranks[item.item_id] + 1) / max_rank
                for item in kept_items
            ])
            y_pred_rank = np.array([item.utility for item in kept_items])

            ndcg5 = ndcg_at_k(y_true_rank, y_pred_rank, k=5)
            ndcg10 = ndcg_at_k(y_true_rank, y_pred_rank, k=10)
            pairwise_acc = pairwise_accuracy(y_true_rank, y_pred_rank)
            tau, _ = kendalltau(y_true_rank, y_pred_rank)
        else:
            ndcg5 = ndcg10 = pairwise_acc = tau = None

        # Recall@K
        pred_sorted = sorted(scored, key=lambda x: -x.utility)
        recalls = {}
        for k in [10, 20, 30, 50]:
            top_k_ids = {item.item_id for item in pred_sorted[:k]}
            recall = len(top_k_ids & final_ids) / len(final_ids) if final_ids else 0
            recalls[f'recall@{k}'] = recall

        return {
            'trip_id': trip.checklist_id,
            'user_id': trip.user_id,
            'seed_size': len(trip.seed_items),
            'final_size': len(trip.final_items),
            'acceptance_rate': len(final_ids) / len(trip.seed_items) if trip.seed_items else 0,
            'ndcg@5': ndcg5,
            'ndcg@10': ndcg10,
            'pairwise_accuracy': pairwise_acc,
            'kendall_tau': tau,
            **recalls
        }

    def evaluate(self, trips: List[TripData] = None) -> Dict:
        """
        Evaluate model on multiple trips.

        Args:
            trips: List of trips (default: all from loader)

        Returns:
            Aggregated metrics
        """
        if trips is None:
            trips = self.loader.trips

        print(f"\n{'='*60}")
        print("Combined Model Evaluation")
        print(f"{'='*60}")
        print(f"Evaluating on {len(trips)} trips...")

        results = []
        for trip in trips:
            if trip.final_items:  # Only evaluate trips with selections
                result = self.evaluate_on_trip(trip)
                results.append(result)

        # Aggregate
        df = pd.DataFrame(results)

        metrics = {
            'n_trips': len(results),
            'avg_seed_size': df['seed_size'].mean(),
            'avg_final_size': df['final_size'].mean(),
            'avg_acceptance_rate': df['acceptance_rate'].mean(),
            'ndcg@5': df['ndcg@5'].dropna().mean(),
            'ndcg@10': df['ndcg@10'].dropna().mean(),
            'pairwise_accuracy': df['pairwise_accuracy'].dropna().mean(),
            'kendall_tau': df['kendall_tau'].dropna().mean(),
            'recall@10': df['recall@10'].mean(),
            'recall@20': df['recall@20'].mean(),
            'recall@30': df['recall@30'].mean(),
            'recall@50': df['recall@50'].mean(),
        }

        print(f"\n=== Aggregated Metrics ===")
        print(f"Trips evaluated: {metrics['n_trips']}")
        print(f"Avg seed size: {metrics['avg_seed_size']:.1f}")
        print(f"Avg final size: {metrics['avg_final_size']:.1f}")
        print(f"Avg acceptance rate: {metrics['avg_acceptance_rate']:.1%}")
        print(f"\nRanking Quality:")
        print(f"  NDCG@5:  {metrics['ndcg@5']:.4f}")
        print(f"  NDCG@10: {metrics['ndcg@10']:.4f}")
        print(f"  Pairwise Accuracy: {metrics['pairwise_accuracy']:.4f}")
        print(f"  Kendall's τ: {metrics['kendall_tau']:.4f}")
        print(f"\nRecall (items in user's list):")
        print(f"  Recall@10: {metrics['recall@10']:.4f}")
        print(f"  Recall@20: {metrics['recall@20']:.4f}")
        print(f"  Recall@30: {metrics['recall@30']:.4f}")
        print(f"  Recall@50: {metrics['recall@50']:.4f}")

        return metrics, df

    def analyze_user_patterns(self) -> pd.DataFrame:
        """Analyze patterns by user."""
        user_results = defaultdict(list)

        for trip in self.loader.trips:
            if trip.final_items:
                result = self.evaluate_on_trip(trip)
                user_results[trip.user_id].append(result)

        # Aggregate by user
        user_stats = []
        for user_id, results in user_results.items():
            df = pd.DataFrame(results)
            user_stats.append({
                'user_id': user_id,
                'n_trips': len(results),
                'avg_final_size': df['final_size'].mean(),
                'avg_acceptance_rate': df['acceptance_rate'].mean(),
                'avg_ndcg10': df['ndcg@10'].dropna().mean(),
                'avg_recall20': df['recall@20'].mean()
            })

        user_df = pd.DataFrame(user_stats).sort_values('n_trips', ascending=False)

        print("\n=== User Pattern Analysis ===")
        print(f"Total users: {len(user_df)}")
        print(f"\nTop users by trip count:")
        print(user_df.head(10).to_string(index=False))

        # Categorize users
        user_df['packer_type'] = pd.cut(
            user_df['avg_acceptance_rate'],
            bins=[0, 0.3, 0.5, 0.7, 1.0],
            labels=['Minimalist', 'Selective', 'Moderate', 'Comprehensive']
        )

        print(f"\nPacker type distribution:")
        print(user_df['packer_type'].value_counts())

        return user_df

    def save(self, path: str):
        """Save combined model configuration."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        config = {
            'regulatory_boost': self.regulatory_boost,
            'safety_boost': self.safety_boost,
            'default_prior': self.default_prior
        }

        with open(path / 'combined_config.json', 'w') as f:
            json.dump(config, f, indent=2)

        print(f"Combined model config saved to {path}")


def create_combined_model(data_dir: str, model_dir: str) -> CombinedPreferenceModel:
    """
    Load trained models and create combined model.

    Args:
        data_dir: Data directory
        model_dir: Directory with saved models

    Returns:
        CombinedPreferenceModel
    """
    from data_loader import load_data

    # Load data
    loader = load_data(data_dir)

    # Load models
    inclusion_model = InclusionModel.load(model_dir)
    ranking_model = RankingModel.load(model_dir)

    # Create combined model
    combined = CombinedPreferenceModel(
        inclusion_model=inclusion_model,
        ranking_model=ranking_model,
        data_loader=loader
    )

    return combined


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    base_dir = Path(__file__).parent.parent
    data_dir = str(base_dir / 'Data')
    model_dir = str(base_dir / 'Models')

    # Try to load existing models, otherwise train new ones
    try:
        combined = create_combined_model(data_dir, model_dir)
    except Exception as e:
        print(f"Could not load models: {e}")
        print("Please run train_pipeline.py first to train the models.")
        sys.exit(1)

    # Evaluate
    metrics, results_df = combined.evaluate()

    # Analyze users
    user_df = combined.analyze_user_patterns()

    # Save results
    results_dir = base_dir / 'Results'
    results_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(results_dir / 'evaluation_results.csv', index=False)
    user_df.to_csv(results_dir / 'user_analysis.csv', index=False)

    with open(results_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nResults saved to {results_dir}")
