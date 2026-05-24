"""
Optimization Evaluation Module

Evaluates Stage 3 optimization performance with multiple metrics and baselines.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import time

from optimization_data import load_optimization_data, OptimizationData, Item, Bag
from optimization_model import PackingOptimizer, PackingSolution


@dataclass
class EvaluationResult:
    """Results from evaluating a single scenario."""
    scenario_id: str
    method: str

    # Feasibility
    is_feasible: bool
    hard_constraint_violations: int

    # Objective metrics
    total_utility: float
    expected_retained_utility: float
    expected_monetary_loss: float
    objective_value: float

    # Coverage metrics
    items_requested: int
    items_packed: int
    items_excluded: int
    packing_rate: float

    # Risk metrics
    high_value_protected: float  # % of high-value items in low-risk bags
    critical_items_accessible: float  # % of critical items in personal/carry-on

    # Efficiency metrics
    avg_weight_utilization: float
    avg_volume_utilization: float

    # Performance
    solve_time_ms: float


class GreedyBaseline:
    """Greedy baseline that packs by utility without full constraint reasoning."""

    def __init__(self, data: OptimizationData):
        self.data = data

    def solve(self, items_to_pack: List[str], available_bags: List[str],
              item_utilities: Optional[Dict[str, float]] = None) -> PackingSolution:
        """Greedy packing by utility, respecting only capacity."""
        start_time = time.time()

        items = {i: self.data.items[i] for i in items_to_pack if i in self.data.items}
        bags = {b: self.data.bags[b] for b in available_bags if b in self.data.bags}

        # Set utilities
        if item_utilities:
            for item_id, utility in item_utilities.items():
                if item_id in items:
                    items[item_id].utility = utility
        else:
            for item_id, item in items.items():
                items[item_id].utility = max(1, item.replacement_cost / 10)

        # Sort by utility (descending)
        sorted_items = sorted(items.keys(), key=lambda i: items[i].utility, reverse=True)

        # Track bag capacities
        bag_weights = {b: 0.0 for b in bags}
        bag_volumes = {b: 0.0 for b in bags}

        assignments = []
        from optimization_model import PackingAssignment

        for item_id in sorted_items:
            item = items[item_id]
            assigned = False

            # Try each bag in order of preference (personal first for high-value)
            bag_order = list(bags.keys())
            if item.value_tier in ['critical', 'high']:
                # Prefer lower-risk bags
                bag_order = sorted(bag_order, key=lambda b: bags[b].loss_probability)

            for bag_id in bag_order:
                bag = bags[bag_id]

                # Check capacity
                if (bag_weights[bag_id] + item.weight_grams <= bag.max_weight_grams and
                    bag_volumes[bag_id] + item.volume_ml <= bag.effective_volume_ml and
                    item.fits_in_bag(bag)):

                    bag_weights[bag_id] += item.weight_grams
                    bag_volumes[bag_id] += item.volume_ml
                    assignments.append(PackingAssignment(item_id, bag_id, item, bag))
                    assigned = True
                    break

            if not assigned:
                assignments.append(PackingAssignment(item_id, None, item, None))

        solve_time = (time.time() - start_time) * 1000

        # Compute metrics
        items_packed = sum(1 for a in assignments if a.bag_id)
        total_utility = sum(a.item.utility for a in assignments if a.bag_id)
        expected_retained = sum(
            a.item.utility * (1 - a.bag.loss_probability)
            for a in assignments if a.bag_id and a.bag
        )

        return PackingSolution(
            assignments=assignments,
            objective_value=expected_retained,
            status="GREEDY",
            solve_time_ms=solve_time,
            total_utility=total_utility,
            expected_retained_utility=expected_retained,
            items_packed=items_packed,
            items_excluded=len(items) - items_packed,
            bag_weights=dict(bag_weights),
            bag_volumes=dict(bag_volumes),
            bag_weight_utilization={b: bag_weights[b] / bags[b].max_weight_grams for b in bags},
            bag_volume_utilization={b: bag_volumes[b] / bags[b].effective_volume_ml for b in bags}
        )


class RandomBaseline:
    """Random valid assignment baseline."""

    def __init__(self, data: OptimizationData, seed: int = 42):
        self.data = data
        self.rng = np.random.RandomState(seed)

    def solve(self, items_to_pack: List[str], available_bags: List[str],
              item_utilities: Optional[Dict[str, float]] = None) -> PackingSolution:
        """Random assignment respecting only capacity."""
        start_time = time.time()

        items = {i: self.data.items[i] for i in items_to_pack if i in self.data.items}
        bags = {b: self.data.bags[b] for b in available_bags if b in self.data.bags}

        # Set utilities
        if item_utilities:
            for item_id, utility in item_utilities.items():
                if item_id in items:
                    items[item_id].utility = utility
        else:
            for item_id, item in items.items():
                items[item_id].utility = max(1, item.replacement_cost / 10)

        # Shuffle items
        shuffled_items = list(items.keys())
        self.rng.shuffle(shuffled_items)

        bag_weights = {b: 0.0 for b in bags}
        bag_volumes = {b: 0.0 for b in bags}

        assignments = []
        from optimization_model import PackingAssignment

        for item_id in shuffled_items:
            item = items[item_id]
            assigned = False

            # Random bag order
            bag_order = list(bags.keys())
            self.rng.shuffle(bag_order)

            for bag_id in bag_order:
                bag = bags[bag_id]

                if (bag_weights[bag_id] + item.weight_grams <= bag.max_weight_grams and
                    bag_volumes[bag_id] + item.volume_ml <= bag.effective_volume_ml and
                    item.fits_in_bag(bag)):

                    bag_weights[bag_id] += item.weight_grams
                    bag_volumes[bag_id] += item.volume_ml
                    assignments.append(PackingAssignment(item_id, bag_id, item, bag))
                    assigned = True
                    break

            if not assigned:
                assignments.append(PackingAssignment(item_id, None, item, None))

        solve_time = (time.time() - start_time) * 1000

        items_packed = sum(1 for a in assignments if a.bag_id)
        total_utility = sum(a.item.utility for a in assignments if a.bag_id)
        expected_retained = sum(
            a.item.utility * (1 - a.bag.loss_probability)
            for a in assignments if a.bag_id and a.bag
        )

        return PackingSolution(
            assignments=assignments,
            objective_value=expected_retained,
            status="RANDOM",
            solve_time_ms=solve_time,
            total_utility=total_utility,
            expected_retained_utility=expected_retained,
            items_packed=items_packed,
            items_excluded=len(items) - items_packed,
            bag_weights=dict(bag_weights),
            bag_volumes=dict(bag_volumes),
            bag_weight_utilization={b: bag_weights[b] / bags[b].max_weight_grams for b in bags},
            bag_volume_utilization={b: bag_volumes[b] / bags[b].effective_volume_ml for b in bags}
        )


class OptimizationEvaluator:
    """Evaluates optimization methods on test scenarios."""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.opt_data = load_optimization_data(data_dir)

        # Methods to evaluate
        self.methods = {
            'cpsat': PackingOptimizer(self.opt_data, alpha=1.0),
            'cpsat_conservative': PackingOptimizer(self.opt_data, alpha=2.0),
            'cpsat_risk_tolerant': PackingOptimizer(self.opt_data, alpha=0.5),
            'greedy': GreedyBaseline(self.opt_data),
            'random': RandomBaseline(self.opt_data)
        }

    def evaluate_solution(self, solution: PackingSolution, scenario: Dict,
                          method: str) -> EvaluationResult:
        """Evaluate a single solution."""

        items_requested = len(scenario['items_to_pack'])

        # Check hard constraint violations
        violations = self._count_violations(solution)

        # Risk metrics
        high_value_protected = self._compute_high_value_protected(solution)
        critical_accessible = self._compute_critical_accessible(solution)

        # Utilization
        utils = list(solution.bag_weight_utilization.values())
        avg_weight_util = np.mean(utils) if utils else 0
        vol_utils = list(solution.bag_volume_utilization.values())
        avg_vol_util = np.mean(vol_utils) if vol_utils else 0

        return EvaluationResult(
            scenario_id=scenario['scenario_id'],
            method=method,
            is_feasible=violations == 0,
            hard_constraint_violations=violations,
            total_utility=solution.total_utility,
            expected_retained_utility=solution.expected_retained_utility,
            expected_monetary_loss=solution.expected_monetary_loss,
            objective_value=solution.objective_value,
            items_requested=items_requested,
            items_packed=solution.items_packed,
            items_excluded=solution.items_excluded,
            packing_rate=solution.items_packed / items_requested if items_requested > 0 else 0,
            high_value_protected=high_value_protected,
            critical_items_accessible=critical_accessible,
            avg_weight_utilization=avg_weight_util,
            avg_volume_utilization=avg_vol_util,
            solve_time_ms=solution.solve_time_ms
        )

    def _count_violations(self, solution: PackingSolution) -> int:
        """Count hard constraint violations."""
        violations = 0

        for assignment in solution.assignments:
            if assignment.bag_id is None:
                continue

            item_id = assignment.item_id
            bag_id = assignment.bag_id

            # Lithium battery in checked
            if item_id in self.opt_data.carry_on_mandatory and bag_id == 'checked':
                violations += 1

            # Strict temp sensitive in checked
            if item_id in self.opt_data.strict_temp_sensitive and bag_id == 'checked':
                violations += 1

            # Prohibited assignments
            if (item_id, bag_id) in self.opt_data.prohibited_assignments:
                violations += 1

        return violations

    def _compute_high_value_protected(self, solution: PackingSolution) -> float:
        """Compute % of high-value items in low-risk bags."""
        high_value_items = []
        protected = 0

        for assignment in solution.assignments:
            if assignment.bag_id is None:
                continue

            if assignment.item.value_tier in ['critical', 'high']:
                high_value_items.append(assignment)
                if assignment.bag_id in ['personal', 'carry_on']:
                    protected += 1

        if not high_value_items:
            return 1.0

        return protected / len(high_value_items)

    def _compute_critical_accessible(self, solution: PackingSolution) -> float:
        """Compute % of critical accessibility items in accessible bags."""
        critical = []
        accessible = 0

        for assignment in solution.assignments:
            if assignment.bag_id is None:
                continue

            if assignment.item.accessibility_preference == 'immediate':
                critical.append(assignment)
                if assignment.bag_id == 'personal':
                    accessible += 1

        if not critical:
            return 1.0

        return accessible / len(critical)

    def run_evaluation(self, time_limit_ms: int = 10000) -> pd.DataFrame:
        """Run full evaluation on all scenarios and methods."""

        # Load scenarios
        scenarios_path = self.data_dir / 'optimization' / 'test_scenarios.json'
        with open(scenarios_path, 'r') as f:
            scenarios_data = json.load(f)

        results = []

        for scenario in scenarios_data['scenarios']:
            print(f"\nEvaluating: {scenario['scenario_id']}")

            for method_name, method in self.methods.items():
                print(f"  Method: {method_name}...", end=" ")

                # Run method
                if isinstance(method, PackingOptimizer):
                    solution = method.solve(
                        items_to_pack=scenario['items_to_pack'],
                        available_bags=scenario['available_bags'],
                        time_limit_ms=time_limit_ms
                    )
                else:
                    # Baselines don't support time_limit_ms
                    solution = method.solve(
                        items_to_pack=scenario['items_to_pack'],
                        available_bags=scenario['available_bags']
                    )

                # Evaluate
                result = self.evaluate_solution(solution, scenario, method_name)
                results.append(result)

                print(f"done ({result.solve_time_ms:.1f}ms, packed={result.items_packed})")

        # Convert to DataFrame
        df = pd.DataFrame([vars(r) for r in results])

        return df

    def compute_summary_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute summary statistics by method."""

        summary = df.groupby('method').agg({
            'is_feasible': 'mean',
            'hard_constraint_violations': 'sum',
            'objective_value': 'mean',
            'expected_retained_utility': 'mean',
            'expected_monetary_loss': 'mean',
            'packing_rate': 'mean',
            'high_value_protected': 'mean',
            'critical_items_accessible': 'mean',
            'avg_weight_utilization': 'mean',
            'avg_volume_utilization': 'mean',
            'solve_time_ms': 'mean'
        }).round(3)

        summary.columns = [
            'Feasibility Rate',
            'Total Violations',
            'Avg Objective',
            'Avg Retained Utility',
            'Avg Monetary Loss',
            'Avg Packing Rate',
            'High Value Protected',
            'Critical Accessible',
            'Avg Weight Util',
            'Avg Volume Util',
            'Avg Solve Time (ms)'
        ]

        return summary


def run_full_evaluation(data_dir: str, output_dir: Optional[str] = None):
    """Run complete evaluation and save results."""

    print("="*60)
    print("Stage 3 Optimization Evaluation")
    print("="*60)

    evaluator = OptimizationEvaluator(data_dir)

    # Run evaluation
    results_df = evaluator.run_evaluation(time_limit_ms=10000)

    # Compute summary
    summary_df = evaluator.compute_summary_stats(results_df)

    print("\n" + "="*60)
    print("Summary by Method")
    print("="*60)
    print(summary_df.to_string())

    # Save results
    if output_dir is None:
        output_dir = Path(data_dir).parent / 'Results' / 'optimization'

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(output_dir / 'evaluation_results.csv', index=False)
    summary_df.to_csv(output_dir / 'evaluation_summary.csv')

    # Save as JSON for visualization
    results_df.to_json(output_dir / 'evaluation_results.json', orient='records', indent=2)

    print(f"\nResults saved to {output_dir}")

    return results_df, summary_df


if __name__ == '__main__':
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else '/Users/529tech/Desktop/Research/PackingChecklistExperiments/Data'

    results_df, summary_df = run_full_evaluation(data_dir)
