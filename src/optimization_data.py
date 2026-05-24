"""
Optimization Data Loader

Loads and preprocesses data from the optimization folder for CP-SAT solver.
"""

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional
from enum import Enum


class ConstraintType(Enum):
    HARD = "hard"
    SOFT = "soft"


@dataclass
class Bag:
    """Bag specification for optimization."""
    bag_id: str
    name: str
    max_weight_kg: float
    max_volume_liters: float
    dimensions_cm: Tuple[float, float, float]  # L, W, H
    packing_efficiency: float
    loss_probability: float
    accessibility: str  # immediate, limited, none
    temperature_controlled: bool
    pressure_controlled: bool

    @property
    def max_weight_grams(self) -> float:
        return self.max_weight_kg * 1000

    @property
    def max_volume_ml(self) -> float:
        return self.max_volume_liters * 1000

    @property
    def effective_volume_ml(self) -> float:
        return self.max_volume_ml * self.packing_efficiency


@dataclass
class Item:
    """Item specification for optimization."""
    item_id: str
    weight_grams: float
    volume_ml: float
    dimensions_mm: Tuple[float, float, float]  # L, W, H
    price_usd: float
    replacement_cost: float
    fragility: str  # low, medium, high, very_high
    spillage_risk: bool
    temperature_sensitive: bool
    pressure_sensitive: bool
    value_tier: str  # low, medium, high, critical
    accessibility_preference: str  # immediate, limited, none
    category: str

    # Computed/assigned fields
    utility: float = 1.0  # Set from preference model

    def fits_in_bag(self, bag: Bag) -> bool:
        """Check if item can physically fit in bag (any orientation)."""
        item_dims = sorted(self.dimensions_mm)
        bag_dims_mm = sorted([d * 10 for d in bag.dimensions_cm])  # Convert to mm
        return all(i <= b for i, b in zip(item_dims, bag_dims_mm))


@dataclass
class MustTogetherGroup:
    """Items that must be in the same bag."""
    group_id: str
    name: str
    items: List[str]
    reason: str
    condition: Optional[str] = None


@dataclass
class MustSeparateGroup:
    """Items that should not be in the same bag."""
    group_id: str
    name: str
    sets: List[List[str]]  # At most one item from each set per bag
    reason: str
    penalty: float
    constraint_type: ConstraintType = ConstraintType.SOFT


@dataclass
class OptimizationData:
    """Container for all optimization data."""
    bags: Dict[str, Bag]
    items: Dict[str, Item]

    # Regulatory constraints
    carry_on_mandatory: Set[str]  # Items that must be in carry-on/personal
    checked_only: Set[str]  # Items that must be in checked
    prohibited_assignments: List[Tuple[str, str]]  # (item_id, bag_id) pairs

    # Environmental constraints
    strict_temp_sensitive: Set[str]  # Hard constraint - no checked
    moderate_temp_sensitive: Dict[str, float]  # Item -> penalty for checked
    pressure_sensitive: Dict[str, float]  # Item -> penalty for checked

    # Dependencies
    must_together_groups: List[MustTogetherGroup]
    must_separate_groups: List[MustSeparateGroup]

    # Soft preferences
    accessibility_penalties: Dict[str, Dict[str, float]]  # item -> bag -> penalty
    fragility_penalties: Dict[str, Dict[str, float]]  # item -> bag -> penalty
    value_tier_penalties: Dict[str, Dict[str, float]]  # item -> bag -> penalty
    spillage_pairs: List[Tuple[str, str, float]]  # (liquid_item, sensitive_item, penalty)

    # Parameters
    risk_aversion_alpha: float = 1.0
    packing_efficiency_eta: float = 0.8


class OptimizationDataLoader:
    """Loads optimization data from JSON files."""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir) / 'optimization'
        self._raw_data = {}

    def load_all(self) -> OptimizationData:
        """Load all optimization data files."""
        self._load_raw_files()
        return self._build_optimization_data()

    def _load_raw_files(self):
        """Load all JSON files."""
        files = [
            'bags.json',
            'items_optimization.json',
            'regulatory_constraints.json',
            'environmental_constraints.json',
            'dependency_constraints.json',
            'soft_preferences.json'
        ]

        for filename in files:
            filepath = self.data_dir / filename
            if filepath.exists():
                with open(filepath, 'r') as f:
                    key = filename.replace('.json', '')
                    self._raw_data[key] = json.load(f)

    def _build_optimization_data(self) -> OptimizationData:
        """Build OptimizationData from raw JSON."""
        bags = self._parse_bags()
        items = self._parse_items()

        # Regulatory constraints
        carry_on_mandatory, checked_only, prohibited = self._parse_regulatory()

        # Environmental constraints
        strict_temp, moderate_temp, pressure = self._parse_environmental()

        # Dependencies
        must_together, must_separate = self._parse_dependencies()

        # Soft preferences
        accessibility_pen, fragility_pen, value_pen, spillage = self._parse_soft_preferences(bags, items)

        return OptimizationData(
            bags=bags,
            items=items,
            carry_on_mandatory=carry_on_mandatory,
            checked_only=checked_only,
            prohibited_assignments=prohibited,
            strict_temp_sensitive=strict_temp,
            moderate_temp_sensitive=moderate_temp,
            pressure_sensitive=pressure,
            must_together_groups=must_together,
            must_separate_groups=must_separate,
            accessibility_penalties=accessibility_pen,
            fragility_penalties=fragility_pen,
            value_tier_penalties=value_pen,
            spillage_pairs=spillage
        )

    def _parse_bags(self) -> Dict[str, Bag]:
        """Parse bag specifications."""
        bags = {}
        raw_bags = self._raw_data.get('bags', {}).get('bags', {})

        for bag_id, data in raw_bags.items():
            dims = data.get('dimensions_cm', {})
            env = data.get('environment', {})

            bags[bag_id] = Bag(
                bag_id=bag_id,
                name=data.get('name', bag_id),
                max_weight_kg=data.get('max_weight_kg', 10),
                max_volume_liters=data.get('max_volume_liters', 40),
                dimensions_cm=(
                    dims.get('length', 50),
                    dims.get('width', 35),
                    dims.get('height', 25)
                ),
                packing_efficiency=data.get('packing_efficiency', 0.8),
                loss_probability=data.get('loss_probability', 0.05),
                accessibility=data.get('accessibility', 'limited'),
                temperature_controlled=env.get('temperature_controlled', True),
                pressure_controlled=env.get('pressure_controlled', True)
            )

        return bags

    def _parse_items(self) -> Dict[str, Item]:
        """Parse item specifications."""
        items = {}
        raw_items = self._raw_data.get('items_optimization', {}).get('items', {})

        for item_id, data in raw_items.items():
            dims = data.get('dimensions_mm', [100, 50, 30])

            items[item_id] = Item(
                item_id=item_id,
                weight_grams=data.get('weight_grams', 100),
                volume_ml=data.get('volume_ml', 200),
                dimensions_mm=tuple(dims) if isinstance(dims, list) else dims,
                price_usd=data.get('price_usd', 20),
                replacement_cost=data.get('replacement_cost', data.get('price_usd', 20)),
                fragility=data.get('fragility', 'low'),
                spillage_risk=data.get('spillage_risk', False),
                temperature_sensitive=data.get('temperature_sensitive', False),
                pressure_sensitive=data.get('pressure_sensitive', False),
                value_tier=data.get('value_tier', 'low'),
                accessibility_preference=data.get('accessibility_preference', 'none'),
                category=data.get('category', 'OTHER')
            )

        return items

    def _parse_regulatory(self) -> Tuple[Set[str], Set[str], List[Tuple[str, str]]]:
        """Parse regulatory constraints."""
        raw = self._raw_data.get('regulatory_constraints', {}).get('constraints', {})

        # Carry-on mandatory (lithium batteries, etc.)
        carry_on_mandatory = set(raw.get('carry_on_mandatory', {}).get('items', []))

        # Checked only
        checked_only = set(raw.get('checked_only', {}).get('items', []))

        # Prohibited assignments
        prohibited = []
        matrix = self._raw_data.get('regulatory_constraints', {}).get('bag_eligibility_matrix', {})
        for entry in matrix.get('prohibited_assignments', []):
            prohibited.append((entry['item'], entry['bag']))

        return carry_on_mandatory, checked_only, prohibited

    def _parse_environmental(self) -> Tuple[Set[str], Dict[str, float], Dict[str, float]]:
        """Parse environmental constraints."""
        raw = self._raw_data.get('environmental_constraints', {}).get('constraints', {})

        # Strict temperature sensitive (hard constraint)
        strict_temp = set(raw.get('strict_temperature_sensitive', {}).get('items', []))

        # Moderate temperature sensitive (soft penalty)
        moderate_temp = {}
        for item in raw.get('moderate_temperature_sensitive', {}).get('items', []):
            moderate_temp[item] = raw.get('moderate_temperature_sensitive', {}).get('penalty_for_checked', 20)

        # Pressure sensitive
        pressure = {}
        for item in raw.get('pressure_sensitive', {}).get('items', []):
            pressure[item] = raw.get('pressure_sensitive', {}).get('penalty_for_checked', 15)

        return strict_temp, moderate_temp, pressure

    def _parse_dependencies(self) -> Tuple[List[MustTogetherGroup], List[MustSeparateGroup]]:
        """Parse dependency constraints."""
        raw = self._raw_data.get('dependency_constraints', {})

        # Must-together groups
        must_together = []
        for group in raw.get('must_together_groups', {}).get('groups', []):
            must_together.append(MustTogetherGroup(
                group_id=group.get('group_id', ''),
                name=group.get('name', ''),
                items=group.get('items', []),
                reason=group.get('reason', ''),
                condition=group.get('condition')
            ))

        # Must-separate groups
        must_separate = []
        for group in raw.get('must_separate_groups', {}).get('groups', []):
            must_separate.append(MustSeparateGroup(
                group_id=group.get('group_id', ''),
                name=group.get('name', ''),
                sets=group.get('sets', []),
                reason=group.get('reason', ''),
                penalty=group.get('penalty', 20)
            ))

        return must_together, must_separate

    def _parse_soft_preferences(self, bags: Dict[str, Bag], items: Dict[str, Item]) -> Tuple[
        Dict[str, Dict[str, float]],
        Dict[str, Dict[str, float]],
        Dict[str, Dict[str, float]],
        List[Tuple[str, str, float]]
    ]:
        """Parse soft preference penalties."""
        raw = self._raw_data.get('soft_preferences', {})

        # Accessibility penalties
        accessibility_pen = {}
        acc_prefs = raw.get('accessibility_preferences', {})
        item_acc = acc_prefs.get('item_accessibility', {})
        levels = acc_prefs.get('levels', {})

        for item_id, level in item_acc.items():
            if item_id not in items:
                continue
            accessibility_pen[item_id] = {}
            level_data = levels.get(level, {})
            penalties = level_data.get('penalty_if_wrong', {})
            for bag_id, penalty in penalties.items():
                accessibility_pen[item_id][bag_id] = penalty

        # Fragility penalties
        fragility_pen = {}
        frag_prefs = raw.get('fragility_preferences', {})
        frag_levels = frag_prefs.get('fragility_levels', {})
        bag_risk = frag_prefs.get('bag_risk_scores', {})
        base_penalty = frag_prefs.get('base_penalty', 10)

        for level_name, level_data in frag_levels.items():
            multiplier = level_data.get('penalty_multiplier', 1.0)
            for item_id in level_data.get('items', []):
                if item_id not in items:
                    continue
                fragility_pen[item_id] = {}
                for bag_id, risk in bag_risk.items():
                    # Higher penalty for fragile items in high-risk bags
                    penalty = base_penalty * multiplier * (risk - 1.0)
                    if penalty > 0:
                        fragility_pen[item_id][bag_id] = penalty

        # Value tier penalties
        value_pen = {}
        risk_prefs = raw.get('risk_preferences', {}).get('value_tiers', {})
        for tier_name, tier_data in risk_prefs.items():
            checked_penalty = tier_data.get('checked_penalty', 0)
            for item_id in tier_data.get('items', []):
                if item_id not in items:
                    continue
                value_pen[item_id] = {'checked': checked_penalty}

        # Spillage pairs
        spillage = []
        spillage_data = raw.get('special_handling', {}).get('spillage_containment', {})
        liquid_items = spillage_data.get('items', [])
        penalty = spillage_data.get('penalty_if_with_sensitive', 25)

        # Define sensitive items (electronics)
        sensitive_items = [i for i in items if items[i].category == 'ELE']

        for liquid in liquid_items:
            for sensitive in sensitive_items:
                if liquid in items and sensitive in items:
                    spillage.append((liquid, sensitive, penalty))

        return accessibility_pen, fragility_pen, value_pen, spillage


def load_optimization_data(data_dir: str) -> OptimizationData:
    """Convenience function to load optimization data."""
    loader = OptimizationDataLoader(data_dir)
    return loader.load_all()


if __name__ == '__main__':
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else '/Users/529tech/Desktop/Research/PackingChecklistExperiments/Data'

    data = load_optimization_data(data_dir)

    print(f"Loaded {len(data.bags)} bags:")
    for bag_id, bag in data.bags.items():
        print(f"  {bag_id}: {bag.max_weight_kg}kg, {bag.max_volume_liters}L, loss_prob={bag.loss_probability}")

    print(f"\nLoaded {len(data.items)} items")
    print(f"Carry-on mandatory: {len(data.carry_on_mandatory)} items")
    print(f"Strict temp sensitive: {len(data.strict_temp_sensitive)} items")
    print(f"Must-together groups: {len(data.must_together_groups)}")
    print(f"Must-separate groups: {len(data.must_separate_groups)}")
