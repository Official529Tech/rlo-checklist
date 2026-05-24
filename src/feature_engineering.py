"""
Feature Engineering Module for Preference Learning

This module creates features for:
1. Stage 1: Inclusion Model (binary classification)
2. Stage 2: Ranking Model (learning to rank)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass

from data_loader import DataLoader, TripData


# Constants
CRITICALITY_MAP = {'mandatory': 3, 'recommended': 2, 'optional': 1}
RULE_FAMILIES = ['regulatory', 'wellbeing', 'essential', 'contextual', 'activities', 'traveler', 'dependencies']
CATEGORIES = ['DOC', 'CLO', 'PER', 'ELE', 'MED', 'TRV', 'ACT']
PURPOSES = ['Business Trip', 'Leisure Trip', 'Family Visit', 'Adventure Trip']
LUGGAGE_TYPES = ['Personal Item', 'Carry-On', 'Checked Baggage']
TRAVEL_TYPES = ['Domestic', 'International']
SPECIAL_NEEDS = ['None', 'Medical Needs', 'With Child', 'With Infant', 'With Pet',
                 'Accessibility Needs', 'Athlete Equipment', 'Pregnancy Needs', 'Senior Traveler', 'Service Animal']
ACTIVITIES = ['Hiking', 'Photography', 'Wildlife & Nature', 'Camping', 'Business Events',
              'Weddings or Ceremonies', 'Fitness Activities', 'Snow Sports', 'Water Sports']


@dataclass
class InclusionExample:
    """Single example for inclusion model training."""
    trip_id: str
    user_id: str
    item_id: str
    features: np.ndarray
    label: int  # 1 if kept, 0 if removed
    feature_names: List[str] = None


@dataclass
class RankingExample:
    """Single example for ranking model training."""
    trip_id: str
    user_id: str
    item_id: str
    features: np.ndarray
    relevance: float  # Higher = more important (inverse of rank)
    rank: int  # Original rank (1 = highest priority)
    feature_names: List[str] = None


class FeatureExtractor:
    """Extracts features for preference learning models."""

    def __init__(self, data_loader: DataLoader):
        self.loader = data_loader
        self.feature_names_inclusion: List[str] = []
        self.feature_names_ranking: List[str] = []
        self._build_feature_names()

    def _build_feature_names(self):
        """Build list of feature names."""
        # Inclusion model features
        names = []

        # Symbolic provenance (7 features)
        names.append('max_rule_priority')
        names.append('num_rules_supporting')
        names.append('criticality_score')
        names.append('confidence_score')
        names.append('is_safety_critical')
        names.append('is_regulatory')
        names.append('num_reasons')

        # Rule family indicators (7 features)
        for family in RULE_FAMILIES:
            names.append(f'rule_family_{family}')

        # Item category (7 features)
        for cat in CATEGORIES:
            names.append(f'category_{cat}')

        # Item attributes (3 features)
        names.append('weight_grams_normalized')
        names.append('volume_ml_normalized')
        names.append('price_normalized')

        # Context features
        # Travel type (2 features)
        for tt in TRAVEL_TYPES:
            names.append(f'travel_type_{tt.replace(" ", "_")}')

        # Duration bins (4 features)
        names.append('duration_1_3')
        names.append('duration_4_7')
        names.append('duration_8_14')
        names.append('duration_15_plus')

        # Purpose (4 features)
        for p in PURPOSES:
            names.append(f'purpose_{p.replace(" ", "_")}')

        # Luggage type (3 features)
        for lt in LUGGAGE_TYPES:
            names.append(f'luggage_{lt.replace(" ", "_").replace("-", "_")}')

        # Special needs (10 features)
        for sn in SPECIAL_NEEDS:
            names.append(f'special_{sn.replace(" ", "_")}')

        # Activity indicators (9 features)
        for act in ACTIVITIES:
            names.append(f'activity_{act.replace(" ", "_").replace("&", "and")}')

        # Numeric context (2 features)
        names.append('num_activities')
        names.append('duration_days_normalized')

        # Item-context interactions (key ones)
        names.append('category_matches_purpose')  # e.g., DOC for business
        names.append('category_matches_activity')  # e.g., ACT for hiking
        names.append('weight_vs_luggage_capacity')  # pressure on space

        self.feature_names_inclusion = names

        # Ranking model uses same base features plus some ranking-specific ones
        self.feature_names_ranking = names.copy()

    def extract_inclusion_features(self) -> Tuple[np.ndarray, np.ndarray, List[InclusionExample]]:
        """
        Extract features for inclusion model.

        Returns:
            X: Feature matrix (n_examples, n_features)
            y: Labels (n_examples,)
            examples: List of InclusionExample objects with metadata
        """
        examples = []

        for trip in self.loader.trips:
            # Get context features (shared for all items in trip)
            context_features = self._extract_context_features(trip)

            # Process each seed item
            for item in trip.seed_items:
                item_id = item['item_id']

                # Item-specific features
                item_features = self._extract_item_features(item, trip)

                # Combine features
                features = np.concatenate([item_features, context_features])

                # Label: 1 if kept, 0 if removed
                label = 1 if item_id in trip.kept_ids else 0

                examples.append(InclusionExample(
                    trip_id=trip.checklist_id,
                    user_id=trip.user_id,
                    item_id=item_id,
                    features=features,
                    label=label,
                    feature_names=self.feature_names_inclusion
                ))

        # Convert to arrays
        X = np.array([ex.features for ex in examples])
        y = np.array([ex.label for ex in examples])

        return X, y, examples

    def extract_ranking_features(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[RankingExample]]:
        """
        Extract features for ranking model.

        Returns:
            X: Feature matrix (n_examples, n_features)
            relevance: Relevance scores (n_examples,)
            groups: Group sizes for each trip (n_trips,)
            examples: List of RankingExample objects with metadata
        """
        examples = []
        groups = []

        for trip in self.loader.trips:
            if not trip.final_items:
                continue

            # Get context features
            context_features = self._extract_context_features(trip)

            # Get max rank for relevance calculation
            # Handle both string and int priority values
            def get_priority(item, default=1):
                p = item.get('priority', default)
                try:
                    return int(p)
                except (TypeError, ValueError):
                    return default

            max_rank = max(get_priority(item, 1) for item in trip.final_items)
            if max_rank == 0:
                max_rank = 1

            trip_examples = []
            for item in trip.final_items:
                item_id = item['item_id']
                rank = get_priority(item, max_rank)

                # Find original item in seed list for provenance features
                seed_item = next((s for s in trip.seed_items if s['item_id'] == item_id), item)

                # Item-specific features
                item_features = self._extract_item_features(seed_item, trip)

                # Combine features
                features = np.concatenate([item_features, context_features])

                # Relevance: higher is better (inverse of rank, normalized)
                relevance = (max_rank - rank + 1) / max_rank

                trip_examples.append(RankingExample(
                    trip_id=trip.checklist_id,
                    user_id=trip.user_id,
                    item_id=item_id,
                    features=features,
                    relevance=relevance,
                    rank=rank,
                    feature_names=self.feature_names_ranking
                ))

            examples.extend(trip_examples)
            groups.append(len(trip_examples))

        # Convert to arrays
        X = np.array([ex.features for ex in examples])
        relevance = np.array([ex.relevance for ex in examples])
        groups = np.array(groups)

        return X, relevance, groups, examples

    def extract_pairwise_data(self) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Extract pairwise preference data for ranking.

        Returns:
            X: Pairwise difference features (n_pairs, n_features)
            y: Labels (1 if first item preferred, 0 otherwise)
            pairs: List of pair metadata
        """
        pairs_X = []
        pairs_y = []
        pairs_meta = []

        for trip in self.loader.trips:
            if len(trip.final_items) < 2:
                continue

            # Sort by priority (1 = highest)
            sorted_items = sorted(trip.final_items, key=lambda x: x.get('priority', 999))
            context_features = self._extract_context_features(trip)

            # Generate pairs
            for i, item_i in enumerate(sorted_items):
                for j, item_j in enumerate(sorted_items[i+1:], start=i+1):
                    # item_i is preferred over item_j
                    seed_i = next((s for s in trip.seed_items if s['item_id'] == item_i['item_id']), item_i)
                    seed_j = next((s for s in trip.seed_items if s['item_id'] == item_j['item_id']), item_j)

                    feat_i = self._extract_item_features(seed_i, trip)
                    feat_j = self._extract_item_features(seed_j, trip)

                    # Difference features
                    diff_features = feat_i - feat_j

                    pairs_X.append(np.concatenate([diff_features, context_features]))
                    pairs_y.append(1)  # item_i preferred

                    pairs_meta.append({
                        'trip_id': trip.checklist_id,
                        'item_i': item_i['item_id'],
                        'item_j': item_j['item_id'],
                        'rank_i': item_i.get('priority', 0),
                        'rank_j': item_j.get('priority', 0)
                    })

        return np.array(pairs_X), np.array(pairs_y), pairs_meta

    def _extract_item_features(self, item: Dict, trip: TripData) -> np.ndarray:
        """Extract features for a single item."""
        features = []
        item_id = item['item_id']

        # Get catalog info
        catalog_info = self.loader.get_item_info(item_id)
        attributes = self.loader.get_item_attributes(item_id)

        # === Symbolic Provenance Features ===

        # Max rule priority that recommended this item
        item_rules = [rule_id for rule_id, count in trip.rule_activations.items()
                      if self._rule_emits_item(rule_id, item_id)]
        max_priority = max([self.loader.get_rule_priority(r) for r in item_rules], default=100)
        features.append(max_priority / 200.0)  # Normalize

        # Number of rules supporting
        features.append(len(item_rules) / 10.0)  # Normalize

        # Criticality score
        crit = item.get('criticality', catalog_info.get('criticality', 'recommended'))
        features.append(CRITICALITY_MAP.get(crit, 2) / 3.0)

        # Confidence score
        features.append(item.get('confidence', 0.5))

        # Is safety critical (check flags and rules)
        is_safety = any('safety' in str(self.loader.rules.get(r, {})).lower() for r in item_rules)
        features.append(float(is_safety))

        # Is regulatory
        is_regulatory = any(self.loader.get_rule_family(r) == 'regulatory' for r in item_rules)
        features.append(float(is_regulatory))

        # Number of reasons
        reasons = item.get('reasons', [])
        features.append(len(reasons) / 5.0)

        # Rule family indicators
        item_families = set(self.loader.get_rule_family(r) for r in item_rules)
        for family in RULE_FAMILIES:
            features.append(float(family in item_families))

        # === Item Category ===
        category = catalog_info.get('category_code', item_id.split('-')[0] if '-' in item_id else 'UNK')
        for cat in CATEGORIES:
            features.append(float(category == cat))

        # === Item Attributes ===
        features.append(attributes.get('weight_grams', 100) / 1000.0)  # Normalize to kg
        features.append(attributes.get('volume_ml', 200) / 1000.0)  # Normalize to liters
        features.append(attributes.get('price', 20) / 100.0)  # Normalize

        return np.array(features)

    def _extract_context_features(self, trip: TripData) -> np.ndarray:
        """Extract context features for a trip."""
        features = []

        # Travel type
        for tt in TRAVEL_TYPES:
            features.append(float(trip.travel_type == tt))

        # Duration bins
        d = trip.duration_days
        features.append(float(1 <= d <= 3))
        features.append(float(4 <= d <= 7))
        features.append(float(8 <= d <= 14))
        features.append(float(d >= 15))

        # Purpose
        for p in PURPOSES:
            features.append(float(trip.purpose == p))

        # Luggage type
        for lt in LUGGAGE_TYPES:
            features.append(float(trip.luggage_type == lt))

        # Special needs
        for sn in SPECIAL_NEEDS:
            features.append(float(trip.special_needs == sn))

        # Activities
        trip_activities = set(trip.activities)
        for act in ACTIVITIES:
            features.append(float(act in trip_activities))

        # Numeric features
        features.append(len(trip.activities) / 5.0)  # Normalize
        features.append(min(trip.duration_days, 30) / 30.0)  # Normalize, cap at 30

        # Item-context interaction placeholders (computed per-item in practice)
        features.append(0.0)  # category_matches_purpose
        features.append(0.0)  # category_matches_activity
        features.append(0.0)  # weight_vs_luggage_capacity

        return np.array(features)

    def _rule_emits_item(self, rule_id: str, item_id: str) -> bool:
        """Check if a rule emits a specific item (heuristic)."""
        rule = self.loader.rules.get(rule_id, {})
        then_clause = rule.get('then', {})
        emit_items = then_clause.get('emit_items', [])

        for emitted in emit_items:
            if isinstance(emitted, dict) and emitted.get('item_id') == item_id:
                return True
            elif isinstance(emitted, str) and emitted == item_id:
                return True

        # Fallback: check if item category matches rule pattern
        item_category = item_id.split('-')[0] if '-' in item_id else ''
        rule_lower = rule_id.lower()

        category_rule_map = {
            'DOC': ['docs', 'passport', 'visa', 'booking'],
            'CLO': ['clothing', 'weather', 'cold', 'warm', 'attire'],
            'ELE': ['electronics', 'phone', 'laptop', 'camera', 'power'],
            'MED': ['medical', 'health', 'prescription', 'otc'],
            'TRV': ['travel', 'luggage', 'comfort'],
            'ACT': ['activity', 'hiking', 'camping', 'fitness'],
            'PER': ['toiletries', 'grooming', 'personal']
        }

        for cat, keywords in category_rule_map.items():
            if item_category == cat and any(kw in rule_lower for kw in keywords):
                return True

        return False


def create_feature_datasets(data_loader: DataLoader, output_dir: str = None):
    """Create and optionally save feature datasets."""
    extractor = FeatureExtractor(data_loader)

    print("\n=== Extracting Inclusion Features ===")
    X_inc, y_inc, examples_inc = extractor.extract_inclusion_features()
    print(f"Inclusion dataset: {X_inc.shape[0]} examples, {X_inc.shape[1]} features")
    print(f"  Positive (kept): {y_inc.sum()} ({y_inc.mean():.1%})")
    print(f"  Negative (removed): {(1-y_inc).sum()} ({(1-y_inc).mean():.1%})")

    print("\n=== Extracting Ranking Features ===")
    X_rank, relevance, groups, examples_rank = extractor.extract_ranking_features()
    print(f"Ranking dataset: {X_rank.shape[0]} examples, {X_rank.shape[1]} features")
    print(f"  Groups (trips): {len(groups)}")
    print(f"  Avg group size: {groups.mean():.1f}")

    print("\n=== Extracting Pairwise Features ===")
    X_pairs, y_pairs, pairs_meta = extractor.extract_pairwise_data()
    print(f"Pairwise dataset: {X_pairs.shape[0]} pairs")

    if output_dir:
        from pathlib import Path
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save as numpy arrays
        np.save(output_path / 'X_inclusion.npy', X_inc)
        np.save(output_path / 'y_inclusion.npy', y_inc)
        np.save(output_path / 'X_ranking.npy', X_rank)
        np.save(output_path / 'relevance_ranking.npy', relevance)
        np.save(output_path / 'groups_ranking.npy', groups)
        np.save(output_path / 'X_pairwise.npy', X_pairs)
        np.save(output_path / 'y_pairwise.npy', y_pairs)

        # Save feature names
        import json
        with open(output_path / 'feature_names.json', 'w') as f:
            json.dump({
                'inclusion': extractor.feature_names_inclusion,
                'ranking': extractor.feature_names_ranking
            }, f, indent=2)

        print(f"\nSaved feature datasets to {output_path}")

    return {
        'inclusion': (X_inc, y_inc, examples_inc),
        'ranking': (X_rank, relevance, groups, examples_rank),
        'pairwise': (X_pairs, y_pairs, pairs_meta),
        'feature_names': {
            'inclusion': extractor.feature_names_inclusion,
            'ranking': extractor.feature_names_ranking
        }
    }


if __name__ == '__main__':
    from data_loader import load_data
    from pathlib import Path
    import sys

    base_dir = Path(__file__).parent.parent
    data_dir = sys.argv[1] if len(sys.argv) > 1 else str(base_dir / 'Data')
    output_dir = sys.argv[2] if len(sys.argv) > 2 else str(base_dir / 'Results')

    loader = load_data(data_dir)
    datasets = create_feature_datasets(loader, output_dir)
