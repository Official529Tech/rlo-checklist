"""
Data Loading and Preprocessing Module for Preference Learning

This module handles:
1. Loading raw JSON/JSONL data files
2. Parsing nested JSON structures
3. Joining user inputs with selections
4. Deriving preference sets (K_t, D_t, A_t)
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import yaml


@dataclass
class TripData:
    """Represents a single trip with all associated data."""
    checklist_id: str
    user_id: str
    timestamp: str

    # Context
    travel_type: str
    destination: str
    duration_days: int
    purpose: str
    luggage_type: str
    activities: List[str]
    special_needs: str

    # Items
    seed_items: List[Dict]  # S_t - system generated
    final_items: List[Dict]  # F_t - user selected with rankings

    # Rule activations
    rule_activations: Dict[str, int]

    # Derived sets
    kept_ids: set = field(default_factory=set)
    removed_ids: set = field(default_factory=set)
    added_ids: set = field(default_factory=set)

    # Metadata
    items_count: int = 0
    rules_activated: int = 0

    def compute_derived_sets(self):
        """Compute K_t, D_t, A_t from seed and final items."""
        seed_ids = {item['item_id'] for item in self.seed_items}
        final_ids = {item['item_id'] for item in self.final_items}

        self.kept_ids = seed_ids & final_ids
        self.removed_ids = seed_ids - final_ids
        self.added_ids = final_ids - seed_ids


class DataLoader:
    """Loads and preprocesses all data for preference learning."""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.trips: List[TripData] = []
        self.item_catalog: Dict = {}
        self.item_attributes: Dict = {}
        self.rules: Dict[str, Dict] = {}
        self.rule_families: Dict[str, str] = {}  # rule_id -> family

    def load_all(self) -> 'DataLoader':
        """Load all data sources."""
        print("Loading data...")
        self._load_item_catalog()
        self._load_item_attributes()
        self._load_rules()
        self._load_trips()
        print(f"Loaded {len(self.trips)} trips with selections")
        return self

    def _load_item_catalog(self):
        """Load item catalog with definitions."""
        catalog_path = self.data_dir / 'items' / 'item_catalog.json'
        with open(catalog_path, 'r') as f:
            data = json.load(f)

        # Flatten categories into single dict
        for cat_code, cat_data in data.get('categories', {}).items():
            for item_id, item_info in cat_data.get('items', {}).items():
                self.item_catalog[item_id] = {
                    **item_info,
                    'category_code': cat_code,
                    'category_name': cat_data.get('category', cat_code)
                }

        print(f"  Loaded {len(self.item_catalog)} items from catalog")

    def _load_item_attributes(self):
        """Load physical attributes for items."""
        attr_path = self.data_dir / 'items' / 'item_attributes.json'
        with open(attr_path, 'r') as f:
            data = json.load(f)

        self.item_attributes = data.get('attributes', {})
        self.luggage_capacities = data.get('luggage_capacities', {})
        print(f"  Loaded attributes for {len(self.item_attributes)} items")

    def _load_rules(self):
        """Load all rule catalogs and build family mapping."""
        rules_dir = self.data_dir / 'rules'

        family_mapping = {
            'regulatory_catalog.yaml': 'regulatory',
            'wellbeing_catalog.yaml': 'wellbeing',
            'essential_catalog.yaml': 'essential',
            'contextual_catalog.yaml': 'contextual',
            'activities_catalog.yaml': 'activities',
            'traveler_catalog.yaml': 'traveler',
            'dependencies_catalog.yaml': 'dependencies'
        }

        for filename, family in family_mapping.items():
            filepath = rules_dir / filename
            if filepath.exists():
                with open(filepath, 'r') as f:
                    data = yaml.safe_load(f)

                for rule in data.get('rules', []):
                    rule_id = rule.get('id', '')
                    self.rules[rule_id] = rule
                    self.rule_families[rule_id] = family

        print(f"  Loaded {len(self.rules)} rules across {len(family_mapping)} families")

    def _load_trips(self):
        """Load and join user contexts with evaluations."""
        # Release: sample files live in data/samples/; full dataset uses data/users/
        samples_dir = self.data_dir / 'samples'
        users_dir = self.data_dir / 'users'
        if (samples_dir / 'user_context_sample.json').exists():
            context_path = samples_dir / 'user_context_sample.json'
            evaluation_path = samples_dir / 'user_evaluation_sample.json'
        else:
            context_path = users_dir / 'user_context.json'
            evaluation_path = users_dir / 'user_evaluation.json'

        # Load user contexts (checklist generation requests)
        inputs_by_id = {}
        with open(context_path, 'r') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    inputs_by_id[record['checklist_id']] = record

        print(f"  Loaded {len(inputs_by_id)} checklist generation requests")

        # Load evaluations and join with contexts
        with open(evaluation_path, 'r') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    checklist_id = record['checklist_id']

                    # Parse nested JSON
                    checklist_details = json.loads(record.get('checklist_details', '{}'))

                    # Get input context
                    input_record = inputs_by_id.get(checklist_id, {})
                    context = json.loads(input_record.get('context_details', '{}'))

                    # Extract data
                    trip = TripData(
                        checklist_id=checklist_id,
                        user_id=record.get('user_id', ''),
                        timestamp=record.get('timestamp', ''),
                        travel_type=context.get('travel_type', 'Unknown'),
                        destination=context.get('destination', ''),
                        duration_days=int(context.get('duration_days', 1)),
                        purpose=context.get('purpose', 'Unknown'),
                        luggage_type=context.get('luggage_type', 'Unknown'),
                        activities=context.get('activities', []),
                        special_needs=context.get('special_needs', 'None'),
                        seed_items=checklist_details.get('items', []),
                        final_items=checklist_details.get('user_selections', []),
                        rule_activations=checklist_details.get('rule_activations', {}),
                        items_count=int(input_record.get('items_count', 0)),
                        rules_activated=int(input_record.get('rules_activated', 0))
                    )

                    # Compute derived sets
                    trip.compute_derived_sets()

                    # Only include trips with actual selections
                    if trip.final_items:
                        self.trips.append(trip)

        print(f"  Loaded {len(self.trips)} trips with user selections")

    def get_rule_family(self, rule_id: str) -> str:
        """Get the family for a rule ID."""
        # Direct lookup
        if rule_id in self.rule_families:
            return self.rule_families[rule_id]

        # Pattern-based fallback
        if rule_id.startswith('rule.docs.') or rule_id.startswith('rule.tsa.'):
            return 'regulatory'
        elif rule_id.startswith('rule.medical.') or rule_id.startswith('rule.accessibility.') or rule_id.startswith('rule.special.'):
            return 'wellbeing'
        elif rule_id.startswith('rule.essentials.'):
            return 'essential'
        elif rule_id.startswith('rule.destination.') or rule_id.startswith('rule.weather.') or rule_id.startswith('rule.duration.') or rule_id.startswith('rule.accommodation.'):
            return 'contextual'
        elif rule_id.startswith('rule.activity.') or rule_id.startswith('rule.athlete.'):
            return 'activities'
        elif rule_id.startswith('rule.demographic.') or rule_id.startswith('rule.comfort.'):
            return 'traveler'
        elif rule_id.startswith('rule.dependency.') or rule_id.startswith('rule.bundle.') or rule_id.startswith('rule.intent.') or rule_id.startswith('rule.regulatory.'):
            return 'dependencies'

        return 'unknown'

    def get_rule_priority(self, rule_id: str) -> int:
        """Get priority for a rule."""
        if rule_id in self.rules:
            return self.rules[rule_id].get('priority', 100)
        return 100

    def get_item_info(self, item_id: str) -> Dict:
        """Get catalog info for an item."""
        return self.item_catalog.get(item_id, {})

    def get_item_attributes(self, item_id: str) -> Dict:
        """Get physical attributes for an item."""
        return self.item_attributes.get(item_id, {
            'weight_grams': 100,
            'volume_ml': 200,
            'price': 20
        })

    def get_statistics(self) -> Dict:
        """Compute summary statistics."""
        stats = {
            'num_trips': len(self.trips),
            'num_users': len(set(t.user_id for t in self.trips)),
            'num_items_catalog': len(self.item_catalog),
            'num_rules': len(self.rules),
        }

        # Trip-level stats
        seed_sizes = [len(t.seed_items) for t in self.trips]
        final_sizes = [len(t.final_items) for t in self.trips]
        kept_sizes = [len(t.kept_ids) for t in self.trips]
        removed_sizes = [len(t.removed_ids) for t in self.trips]
        added_sizes = [len(t.added_ids) for t in self.trips]

        stats['seed_size'] = {
            'mean': np.mean(seed_sizes),
            'std': np.std(seed_sizes),
            'min': np.min(seed_sizes),
            'max': np.max(seed_sizes)
        }
        stats['final_size'] = {
            'mean': np.mean(final_sizes),
            'std': np.std(final_sizes),
            'min': np.min(final_sizes),
            'max': np.max(final_sizes)
        }
        stats['kept_size'] = {'mean': np.mean(kept_sizes), 'std': np.std(kept_sizes)}
        stats['removed_size'] = {'mean': np.mean(removed_sizes), 'std': np.std(removed_sizes)}
        stats['added_size'] = {'mean': np.mean(added_sizes), 'std': np.std(added_sizes)}

        # Acceptance rate
        stats['overall_acceptance_rate'] = np.mean([
            len(t.kept_ids) / len(t.seed_items) if t.seed_items else 0
            for t in self.trips
        ])

        # Context distributions
        stats['travel_types'] = defaultdict(int)
        stats['purposes'] = defaultdict(int)
        stats['luggage_types'] = defaultdict(int)

        for t in self.trips:
            stats['travel_types'][t.travel_type] += 1
            stats['purposes'][t.purpose] += 1
            stats['luggage_types'][t.luggage_type] += 1

        stats['travel_types'] = dict(stats['travel_types'])
        stats['purposes'] = dict(stats['purposes'])
        stats['luggage_types'] = dict(stats['luggage_types'])

        return stats


def load_data(data_dir: str) -> DataLoader:
    """Convenience function to load all data."""
    loader = DataLoader(data_dir)
    return loader.load_all()


if __name__ == '__main__':
    # Test loading
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parent.parent / 'data')

    loader = load_data(data_dir)
    stats = loader.get_statistics()

    print("\n=== Data Statistics ===")
    print(f"Trips: {stats['num_trips']}")
    print(f"Users: {stats['num_users']}")
    print(f"Items in catalog: {stats['num_items_catalog']}")
    print(f"Rules: {stats['num_rules']}")
    print(f"\nSeed list size: {stats['seed_size']['mean']:.1f} ± {stats['seed_size']['std']:.1f}")
    print(f"Final list size: {stats['final_size']['mean']:.1f} ± {stats['final_size']['std']:.1f}")
    print(f"Overall acceptance rate: {stats['overall_acceptance_rate']:.1%}")
    print(f"\nTravel types: {stats['travel_types']}")
    print(f"Purposes: {stats['purposes']}")
