"""
CP-SAT Packing Optimizer

Solves the constraint-aware packing optimization problem using Google OR-Tools CP-SAT solver.
"""

from ortools.sat.python import cp_model
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
import time

from optimization_data import OptimizationData, Bag, Item, MustTogetherGroup, MustSeparateGroup


@dataclass
class PackingAssignment:
    """Result of optimization: which items go in which bags."""
    item_id: str
    bag_id: Optional[str]  # None if item not packed
    item: Item
    bag: Optional[Bag]


@dataclass
class PackingSolution:
    """Complete solution from optimizer."""
    assignments: List[PackingAssignment]
    objective_value: float
    status: str  # OPTIMAL, FEASIBLE, INFEASIBLE, etc.
    solve_time_ms: float

    # Objective components
    total_utility: float = 0.0
    expected_retained_utility: float = 0.0
    expected_monetary_loss: float = 0.0
    environmental_penalty: float = 0.0
    soft_penalty: float = 0.0

    # Constraint stats
    items_packed: int = 0
    items_excluded: int = 0

    # Bag utilization
    bag_weights: Dict[str, float] = field(default_factory=dict)
    bag_volumes: Dict[str, float] = field(default_factory=dict)
    bag_weight_utilization: Dict[str, float] = field(default_factory=dict)
    bag_volume_utilization: Dict[str, float] = field(default_factory=dict)

    def get_bag_contents(self, bag_id: str) -> List[PackingAssignment]:
        """Get all items assigned to a specific bag."""
        return [a for a in self.assignments if a.bag_id == bag_id]

    def get_unpacked_items(self) -> List[PackingAssignment]:
        """Get items that were not packed."""
        return [a for a in self.assignments if a.bag_id is None]

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"=== Packing Solution ===",
            f"Status: {self.status}",
            f"Solve time: {self.solve_time_ms:.1f}ms",
            f"",
            f"Items: {self.items_packed} packed, {self.items_excluded} excluded",
            f"",
            f"Objective breakdown:",
            f"  Total utility: {self.total_utility:.2f}",
            f"  Expected retained utility: {self.expected_retained_utility:.2f}",
            f"  Expected monetary loss: {self.expected_monetary_loss:.2f}",
            f"  Environmental penalty: {self.environmental_penalty:.2f}",
            f"  Soft penalty: {self.soft_penalty:.2f}",
            f"  Final objective: {self.objective_value:.2f}",
            f"",
            f"Bag utilization:"
        ]

        for bag_id in sorted(self.bag_weights.keys()):
            w_util = self.bag_weight_utilization.get(bag_id, 0) * 100
            v_util = self.bag_volume_utilization.get(bag_id, 0) * 100
            lines.append(f"  {bag_id}: weight={w_util:.1f}%, volume={v_util:.1f}%")

        return "\n".join(lines)


class PackingOptimizer:
    """
    CP-SAT optimizer for packing problem.

    Maximizes expected retained utility subject to:
    - Physical constraints (weight, volume, dimensions)
    - Regulatory constraints (bag eligibility)
    - Environmental constraints (temperature, pressure)
    - Dependency constraints (must-together, must-separate)
    """

    # Scaling factor for converting floats to integers (CP-SAT requires integers)
    SCALE = 1000

    def __init__(self, data: OptimizationData, alpha: float = 1.0):
        """
        Initialize optimizer.

        Args:
            data: Optimization data including bags, items, constraints
            alpha: Risk aversion parameter (higher = more conservative)
        """
        self.data = data
        self.alpha = alpha
        self.model = None
        self.solver = None

        # Decision variables
        self.x = {}  # x[item_id, bag_id] = 1 if item in bag
        self.y = {}  # y[item_id] = 1 if item packed anywhere

    def solve(self,
              items_to_pack: List[str],
              available_bags: List[str],
              item_utilities: Optional[Dict[str, float]] = None,
              time_limit_ms: int = 5000,
              log_search: bool = False) -> PackingSolution:
        """
        Solve the packing optimization problem.

        Args:
            items_to_pack: List of item IDs to consider packing
            available_bags: List of bag IDs available
            item_utilities: Optional utility scores (default: use replacement cost)
            time_limit_ms: Maximum solve time in milliseconds
            log_search: Whether to log solver progress

        Returns:
            PackingSolution with optimal assignments
        """
        start_time = time.time()

        # Filter to items we have data for
        items = {i: self.data.items[i] for i in items_to_pack if i in self.data.items}
        bags = {b: self.data.bags[b] for b in available_bags if b in self.data.bags}

        if not items:
            return self._empty_solution("No valid items", time.time() - start_time)
        if not bags:
            return self._empty_solution("No valid bags", time.time() - start_time)

        # Set utilities
        if item_utilities:
            for item_id, utility in item_utilities.items():
                if item_id in items:
                    items[item_id].utility = utility
        else:
            # Default: use replacement cost as proxy for utility
            for item_id, item in items.items():
                items[item_id].utility = max(1, item.replacement_cost / 10)

        # Build and solve model
        self.model = cp_model.CpModel()
        self._create_variables(items, bags)
        self._add_physical_constraints(items, bags)
        self._add_regulatory_constraints(items, bags)
        self._add_environmental_constraints(items, bags)
        self._add_dependency_constraints(items, bags)
        self._set_objective(items, bags)

        # Solve
        self.solver = cp_model.CpSolver()
        self.solver.parameters.max_time_in_seconds = time_limit_ms / 1000.0
        if log_search:
            self.solver.parameters.log_search_progress = True

        status = self.solver.Solve(self.model)
        solve_time = (time.time() - start_time) * 1000

        # Extract solution
        return self._extract_solution(items, bags, status, solve_time)

    def _create_variables(self, items: Dict[str, Item], bags: Dict[str, Bag]):
        """Create decision variables."""
        self.x = {}
        self.y = {}

        for item_id in items:
            # y[i] = 1 if item i is packed
            self.y[item_id] = self.model.NewBoolVar(f'y_{item_id}')

            for bag_id in bags:
                # x[i,b] = 1 if item i is in bag b
                self.x[item_id, bag_id] = self.model.NewBoolVar(f'x_{item_id}_{bag_id}')

        # Link y to x: y[i] = sum_b x[i,b]
        for item_id in items:
            self.model.Add(
                self.y[item_id] == sum(self.x[item_id, bag_id] for bag_id in bags)
            )

    def _add_physical_constraints(self, items: Dict[str, Item], bags: Dict[str, Bag]):
        """Add weight, volume, and dimensional constraints."""

        for bag_id, bag in bags.items():
            # Weight constraint
            self.model.Add(
                sum(int(items[i].weight_grams) * self.x[i, bag_id] for i in items)
                <= int(bag.max_weight_grams)
            )

            # Volume constraint (with packing efficiency)
            self.model.Add(
                sum(int(items[i].volume_ml) * self.x[i, bag_id] for i in items)
                <= int(bag.effective_volume_ml)
            )

        # Dimensional compatibility (precompute and forbid incompatible)
        for item_id, item in items.items():
            for bag_id, bag in bags.items():
                if not item.fits_in_bag(bag):
                    self.model.Add(self.x[item_id, bag_id] == 0)

    def _add_regulatory_constraints(self, items: Dict[str, Item], bags: Dict[str, Bag]):
        """Add regulatory eligibility constraints (hard)."""

        # Carry-on mandatory items cannot go in checked
        if 'checked' in bags:
            for item_id in items:
                if item_id in self.data.carry_on_mandatory:
                    self.model.Add(self.x[item_id, 'checked'] == 0)

        # Checked-only items cannot go in carry-on or personal
        for bag_id in ['carry_on', 'personal']:
            if bag_id in bags:
                for item_id in items:
                    if item_id in self.data.checked_only:
                        self.model.Add(self.x[item_id, bag_id] == 0)

        # Explicit prohibited assignments
        for item_id, bag_id in self.data.prohibited_assignments:
            if item_id in items and bag_id in bags:
                self.model.Add(self.x[item_id, bag_id] == 0)

    def _add_environmental_constraints(self, items: Dict[str, Item], bags: Dict[str, Bag]):
        """Add environmental sensitivity constraints."""

        # Strict temperature sensitive: hard constraint - no checked bag
        if 'checked' in bags:
            for item_id in items:
                if item_id in self.data.strict_temp_sensitive:
                    self.model.Add(self.x[item_id, 'checked'] == 0)

        # Moderate constraints are handled as soft penalties in objective

    def _add_dependency_constraints(self, items: Dict[str, Item], bags: Dict[str, Bag]):
        """Add must-together and must-separate constraints."""

        # Must-together (hard): all items in group go to same bag
        for group in self.data.must_together_groups:
            # Filter to items actually being packed
            group_items = [i for i in group.items if i in items]
            if len(group_items) < 2:
                continue

            # For each pair of items in group, they must be in same bag
            first_item = group_items[0]
            for other_item in group_items[1:]:
                for bag_id in bags:
                    # x[first, bag] == x[other, bag] for all bags
                    self.model.Add(
                        self.x[first_item, bag_id] == self.x[other_item, bag_id]
                    )

        # Must-separate (hard): items in same set cannot be in same bag
        for group in self.data.must_separate_groups:
            for item_set in group.sets:
                # Filter to items actually being packed
                set_items = [i for i in item_set if i in items]
                if len(set_items) < 2:
                    continue

                # For each bag, at most one item from this set can be assigned
                for bag_id in bags:
                    self.model.Add(
                        sum(self.x[item_id, bag_id] for item_id in set_items) <= 1
                    )

    def _set_objective(self, items: Dict[str, Item], bags: Dict[str, Bag]):
        """Set the optimization objective."""

        objective_terms = []

        # === POSITIVE: Expected retained utility ===
        # sum_i sum_b u_i * (1 - p_b) * x_{ib}
        for item_id, item in items.items():
            for bag_id, bag in bags.items():
                coef = int(item.utility * (1 - bag.loss_probability) * self.SCALE)
                objective_terms.append(coef * self.x[item_id, bag_id])

        # === NEGATIVE: Expected monetary loss ===
        # alpha * sum_b p_b * sum_i v_i * x_{ib}
        for item_id, item in items.items():
            for bag_id, bag in bags.items():
                coef = int(self.alpha * bag.loss_probability * item.replacement_cost)
                objective_terms.append(-coef * self.x[item_id, bag_id])

        # === NEGATIVE: Environmental penalties (soft) ===
        if 'checked' in bags:
            # Moderate temperature sensitive
            for item_id, penalty in self.data.moderate_temp_sensitive.items():
                if item_id in items:
                    objective_terms.append(-int(penalty) * self.x[item_id, 'checked'])

            # Pressure sensitive
            for item_id, penalty in self.data.pressure_sensitive.items():
                if item_id in items:
                    objective_terms.append(-int(penalty) * self.x[item_id, 'checked'])

        # === NEGATIVE: Accessibility penalties ===
        for item_id, bag_penalties in self.data.accessibility_penalties.items():
            if item_id in items:
                for bag_id, penalty in bag_penalties.items():
                    if bag_id in bags:
                        objective_terms.append(-int(penalty) * self.x[item_id, bag_id])

        # === NEGATIVE: Fragility penalties ===
        for item_id, bag_penalties in self.data.fragility_penalties.items():
            if item_id in items:
                for bag_id, penalty in bag_penalties.items():
                    if bag_id in bags:
                        objective_terms.append(-int(penalty) * self.x[item_id, bag_id])

        # === NEGATIVE: Value tier (risk) penalties ===
        for item_id, bag_penalties in self.data.value_tier_penalties.items():
            if item_id in items:
                for bag_id, penalty in bag_penalties.items():
                    if bag_id in bags:
                        objective_terms.append(-int(penalty) * self.x[item_id, bag_id])

        # === NEGATIVE: Spillage penalties (liquids near electronics) ===
        # For each bag, penalize having both liquid and sensitive item
        for bag_id in bags:
            for liquid, sensitive, penalty in self.data.spillage_pairs:
                if liquid in items and sensitive in items:
                    # Create auxiliary variable: both_in_bag
                    both = self.model.NewBoolVar(f'spill_{liquid}_{sensitive}_{bag_id}')
                    self.model.AddMultiplicationEquality(
                        both,
                        [self.x[liquid, bag_id], self.x[sensitive, bag_id]]
                    )
                    objective_terms.append(-int(penalty) * both)

        # Set objective
        self.model.Maximize(sum(objective_terms))

    def _extract_solution(self, items: Dict[str, Item], bags: Dict[str, Bag],
                          status: int, solve_time: float) -> PackingSolution:
        """Extract solution from solver."""

        status_map = {
            cp_model.OPTIMAL: "OPTIMAL",
            cp_model.FEASIBLE: "FEASIBLE",
            cp_model.INFEASIBLE: "INFEASIBLE",
            cp_model.MODEL_INVALID: "MODEL_INVALID",
            cp_model.UNKNOWN: "UNKNOWN"
        }
        status_str = status_map.get(status, "UNKNOWN")

        if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            return PackingSolution(
                assignments=[],
                objective_value=0,
                status=status_str,
                solve_time_ms=solve_time
            )

        # Extract assignments
        assignments = []
        bag_weights = defaultdict(float)
        bag_volumes = defaultdict(float)

        for item_id, item in items.items():
            assigned_bag = None
            for bag_id, bag in bags.items():
                if self.solver.Value(self.x[item_id, bag_id]) == 1:
                    assigned_bag = bag_id
                    bag_weights[bag_id] += item.weight_grams
                    bag_volumes[bag_id] += item.volume_ml
                    break

            assignments.append(PackingAssignment(
                item_id=item_id,
                bag_id=assigned_bag,
                item=item,
                bag=bags.get(assigned_bag) if assigned_bag else None
            ))

        # Compute metrics
        items_packed = sum(1 for a in assignments if a.bag_id is not None)
        items_excluded = sum(1 for a in assignments if a.bag_id is None)

        # Compute objective components
        total_utility = sum(a.item.utility for a in assignments if a.bag_id)
        expected_retained = sum(
            a.item.utility * (1 - a.bag.loss_probability)
            for a in assignments if a.bag_id and a.bag
        )
        expected_loss = sum(
            self.alpha * a.bag.loss_probability * a.item.replacement_cost
            for a in assignments if a.bag_id and a.bag
        )

        # Bag utilization
        bag_weight_util = {}
        bag_volume_util = {}
        for bag_id, bag in bags.items():
            bag_weight_util[bag_id] = bag_weights[bag_id] / bag.max_weight_grams
            bag_volume_util[bag_id] = bag_volumes[bag_id] / bag.effective_volume_ml

        return PackingSolution(
            assignments=assignments,
            objective_value=self.solver.ObjectiveValue() / self.SCALE,
            status=status_str,
            solve_time_ms=solve_time,
            total_utility=total_utility,
            expected_retained_utility=expected_retained,
            expected_monetary_loss=expected_loss,
            items_packed=items_packed,
            items_excluded=items_excluded,
            bag_weights=dict(bag_weights),
            bag_volumes=dict(bag_volumes),
            bag_weight_utilization=bag_weight_util,
            bag_volume_utilization=bag_volume_util
        )

    def _empty_solution(self, reason: str, solve_time: float) -> PackingSolution:
        """Return empty solution for edge cases."""
        return PackingSolution(
            assignments=[],
            objective_value=0,
            status=f"EMPTY: {reason}",
            solve_time_ms=solve_time * 1000
        )


if __name__ == '__main__':
    from optimization_data import load_optimization_data
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else '/Users/529tech/Desktop/Research/PackingChecklistExperiments/Data'

    # Load data
    print("Loading optimization data...")
    data = load_optimization_data(data_dir)

    # Create optimizer
    optimizer = PackingOptimizer(data, alpha=1.0)

    # Test with a simple scenario
    items = [
        "DOC-PASSPORT", "DOC-ID", "DOC-BOARDING",
        "ELE-PHONE-HANDSET", "ELE-PHONE-CHARGER", "ELE-POWER",
        "ELE-LAPTOP-DEVICE", "ELE-LAPTOP-CHARGER",
        "CLO-UNDERWEAR", "CLO-SOCKS", "CLO-TSHIRT", "CLO-PANTS",
        "PER-TOOTHBRUSH", "PER-TOOTHPASTE",
        "TRV-WALLET", "TRV-CARD"
    ]

    bags = ["personal", "carry_on"]

    print(f"\nSolving packing problem...")
    print(f"  Items: {len(items)}")
    print(f"  Bags: {bags}")

    solution = optimizer.solve(items, bags, time_limit_ms=5000)

    print(f"\n{solution.summary()}")

    print("\n=== Bag Contents ===")
    for bag_id in bags:
        contents = solution.get_bag_contents(bag_id)
        print(f"\n{bag_id.upper()} ({len(contents)} items):")
        for a in contents:
            print(f"  - {a.item_id}: {a.item.weight_grams}g, ${a.item.replacement_cost}")

    unpacked = solution.get_unpacked_items()
    if unpacked:
        print(f"\nNOT PACKED ({len(unpacked)} items):")
        for a in unpacked:
            print(f"  - {a.item_id}")
