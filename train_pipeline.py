#!/usr/bin/env python3
"""
Main Training Pipeline for Preference Learning

This script orchestrates the complete training pipeline:
1. Load and preprocess data
2. Extract features
3. Train inclusion model (Stage 1)
4. Train ranking model (Stage 2)
5. Create combined model
6. Evaluate and generate reports
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_loader import load_data, DataLoader
from feature_engineering import create_feature_datasets, FeatureExtractor
from inclusion_model import InclusionModel, train_inclusion_model
from ranking_model import RankingModel, train_ranking_model, HAS_LIGHTGBM
from combined_model import CombinedPreferenceModel


def print_header(text: str):
    """Print formatted header."""
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}\n")


def run_pipeline(data_dir: str, output_dir: str, results_dir: str = None,
                 inclusion_model_type: str = 'logistic',
                 ranking_model_type: str = 'lambdamart',
                 skip_evaluation: bool = False):
    """
    Run the complete preference learning pipeline.

    Args:
        data_dir: Directory containing data files (Data folder)
        output_dir: Directory to save models (Code folder)
        results_dir: Directory to save results (Results folder)
        inclusion_model_type: Type of inclusion model ('logistic', 'rf', 'gbm')
        ranking_model_type: Type of ranking model ('lambdamart', 'pairwise')
        skip_evaluation: Skip final evaluation
    """
    start_time = datetime.now()
    print_header("PREFERENCE LEARNING PIPELINE")
    print(f"Start time: {start_time}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")

    # Create output directories
    output_path = Path(output_dir)
    models_dir = output_path
    outputs_dir = Path(results_dir) if results_dir else output_path.parent / 'results'

    models_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    print(f"Models directory: {models_dir}")
    print(f"Results directory: {outputs_dir}")

    # ========================================
    # Phase 1: Load Data
    # ========================================
    print_header("PHASE 1: DATA LOADING")

    loader = load_data(data_dir)
    stats = loader.get_statistics()

    print("\n=== Data Statistics ===")
    print(f"Total trips with selections: {stats['num_trips']}")
    print(f"Unique users: {stats['num_users']}")
    print(f"Items in catalog: {stats['num_items_catalog']}")
    print(f"Rules: {stats['num_rules']}")
    print(f"\nSeed list size: {stats['seed_size']['mean']:.1f} ± {stats['seed_size']['std']:.1f}")
    print(f"Final list size: {stats['final_size']['mean']:.1f} ± {stats['final_size']['std']:.1f}")
    print(f"Overall acceptance rate: {stats['overall_acceptance_rate']:.1%}")

    # Save stats
    with open(outputs_dir / 'data_statistics.json', 'w') as f:
        json.dump(stats, f, indent=2, default=str)

    # ========================================
    # Phase 2: Feature Engineering
    # ========================================
    print_header("PHASE 2: FEATURE ENGINEERING")

    datasets = create_feature_datasets(loader, str(outputs_dir))

    X_inc, y_inc, examples_inc = datasets['inclusion']
    X_rank, relevance, groups, examples_rank = datasets['ranking']
    X_pairs, y_pairs, pairs_meta = datasets['pairwise']
    feature_names = datasets['feature_names']

    # Get user IDs for group-based CV
    user_ids_inc = np.array([ex.user_id for ex in examples_inc])
    user_ids_rank = np.array([ex.user_id for ex in examples_rank])

    print(f"\nFeature dimensions:")
    print(f"  Inclusion features: {len(feature_names['inclusion'])}")
    print(f"  Ranking features: {len(feature_names['ranking'])}")

    # ========================================
    # Phase 3: Train Inclusion Model
    # ========================================
    print_header("PHASE 3: INCLUSION MODEL (STAGE 1)")

    inclusion_model = train_inclusion_model(
        X_inc, y_inc,
        feature_names=feature_names['inclusion'],
        user_ids=user_ids_inc,
        output_dir=str(models_dir),
        model_type=inclusion_model_type
    )

    # ========================================
    # Phase 4: Train Ranking Model
    # ========================================
    print_header("PHASE 4: RANKING MODEL (STAGE 2)")

    # Fallback to pairwise if LightGBM not available
    if ranking_model_type == 'lambdamart' and not HAS_LIGHTGBM:
        print("LightGBM not installed, using pairwise logistic regression")
        ranking_model_type = 'pairwise'

    ranking_model = train_ranking_model(
        X_rank, relevance, groups,
        feature_names=feature_names['ranking'],
        X_pairwise=X_pairs,
        y_pairwise=y_pairs,
        output_dir=str(models_dir),
        model_type=ranking_model_type
    )

    # ========================================
    # Phase 5: Combined Model Evaluation
    # ========================================
    if not skip_evaluation:
        print_header("PHASE 5: COMBINED MODEL EVALUATION")

        combined = CombinedPreferenceModel(
            inclusion_model=inclusion_model,
            ranking_model=ranking_model,
            data_loader=loader
        )

        # Full evaluation
        metrics, results_df = combined.evaluate()

        # User analysis
        user_df = combined.analyze_user_patterns()

        # Save results
        results_df.to_csv(outputs_dir / 'evaluation_results.csv', index=False)
        user_df.to_csv(outputs_dir / 'user_analysis.csv', index=False)

        with open(outputs_dir / 'combined_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        combined.save(str(models_dir))

        # Example predictions
        print_header("EXAMPLE PREDICTIONS")

        # Pick a random trip
        sample_trip = loader.trips[0]
        scored = combined.score_items(sample_trip)

        print(f"Trip: {sample_trip.destination} ({sample_trip.duration_days} days, {sample_trip.purpose})")
        print(f"Activities: {sample_trip.activities}")
        print(f"\nTop 15 recommended items:")
        print("-" * 80)
        print(f"{'Rank':<5} {'Item':<30} {'Category':<8} {'P(inc)':<8} {'Rank':<8} {'Utility':<8}")
        print("-" * 80)

        for i, item in enumerate(scored[:15], 1):
            print(f"{i:<5} {item.name[:28]:<30} {item.category:<8} "
                  f"{item.p_include:<8.3f} {item.s_rank:<8.3f} {item.utility:<8.3f}")

        # Compare with user's actual selection
        final_ids = {item['item_id'] for item in sample_trip.final_items}
        print(f"\nUser selected {len(final_ids)} items")
        print(f"Overlap in top 15: {len({item.item_id for item in scored[:15]} & final_ids)}")

    # ========================================
    # Summary
    # ========================================
    print_header("PIPELINE COMPLETE")

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print(f"End time: {end_time}")
    print(f"Duration: {duration:.1f} seconds")
    print(f"\nOutputs saved to: {output_path}")
    print(f"  Models: {models_dir}")
    print(f"  Results: {outputs_dir}")

    # Summary report
    summary = {
        'start_time': str(start_time),
        'end_time': str(end_time),
        'duration_seconds': duration,
        'data_stats': stats,
        'inclusion_model': {
            'type': inclusion_model_type,
            'metrics': inclusion_model.metrics
        },
        'ranking_model': {
            'type': ranking_model_type,
            'metrics': ranking_model.metrics
        }
    }

    if not skip_evaluation:
        summary['combined_metrics'] = metrics

    with open(outputs_dir / 'pipeline_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print("\nPipeline completed successfully!")

    return {
        'loader': loader,
        'inclusion_model': inclusion_model,
        'ranking_model': ranking_model,
        'combined': combined if not skip_evaluation else None
    }


def main():
    """Main entry point."""
    base_dir = Path(__file__).parent  # release root

    parser = argparse.ArgumentParser(
        description='Train preference learning models for packing checklist optimization'
    )
    parser.add_argument(
        '--data-dir',
        default=str(base_dir / 'data'),
        help='Directory containing data files'
    )
    parser.add_argument(
        '--output-dir',
        default=str(base_dir / 'models'),
        help='Directory for saved models'
    )
    parser.add_argument(
        '--results-dir',
        default=str(base_dir / 'results'),
        help='Directory for evaluation results'
    )
    parser.add_argument(
        '--inclusion-model',
        choices=['logistic', 'rf', 'gbm'],
        default='logistic',
        help='Type of inclusion model'
    )
    parser.add_argument(
        '--ranking-model',
        choices=['lambdamart', 'pairwise'],
        default='lambdamart',
        help='Type of ranking model'
    )
    parser.add_argument(
        '--skip-eval',
        action='store_true',
        help='Skip final evaluation'
    )

    args = parser.parse_args()

    run_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        results_dir=args.results_dir,
        inclusion_model_type=args.inclusion_model,
        ranking_model_type=args.ranking_model,
        skip_evaluation=args.skip_eval
    )


if __name__ == '__main__':
    main()
