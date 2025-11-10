#!/usr/bin/env python3
"""
Script to aggregate ECG SSL results for LaTeX table generation.
Extracts results from multiple models, seeds, and label fractions to calculate
mean ± std dev and std error for AUROC and PR-AUC metrics.
"""

import os
import json
import warnings

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
from scipy import stats
from utils.helper_paths import RESULTS_PATH

warnings.filterwarnings("ignore")


def load_json_results(file_path: str) -> Dict:
    """Load test results from JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Handle different JSON formats
        if 'test_metrics' in data:
            return data['test_metrics']
        else:
            return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return {}

def collect_results(results_dir: str) -> pd.DataFrame:
    """Collect all results from the results directory structure."""
    results = []
    results_path = Path(results_dir)

    target_seeds = [3, 5, 7, 9, 42]
    
    # Label fractions to collect
    label_fractions = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
    
    # Supervised models
    supervised_models = ['cnn', 'tcn', 'transformer']
    
    # SSL models - mapping directory names to clean names
    ssl_models = {
        'TS2Vec': 'TS2Vec',
        'TS2Vec_S3': 'TS2Vec_S3', 
        'TSTCC': 'TSTCC',
        'TSTCC_S3': 'TSTCC_S3',
        'SimCLR': 'SimCLR',
        'SimCLR_S3': 'SimCLR_S3',
        'InfoTS': 'InfoTS',
        'InfoTS_S3': 'InfoTS_S3'
    }
    
    print("Collecting results...")

    # Collect Supervised results
    supervised_path = results_path / "ECG" / "Supervised"
    if supervised_path.exists():
        for model in supervised_models:
            model_path = supervised_path / model
            if not model_path.exists():
                continue
                
            for seed in target_seeds:
                seed_path = model_path / str(seed)
                if not seed_path.exists():
                    continue
                    
                for label_frac in label_fractions:
                    # Try different path structures
                    possible_paths = [
                        seed_path / str(label_frac) / "10" / "5" / "test_results.json"
                    ]
                    
                    for result_file in possible_paths:
                        if result_file.exists():
                            metrics = load_json_results(str(result_file))
                            if metrics and 'auroc' in metrics and 'pr_auc' in metrics:
                                results.append({
                                    'method_type': 'Supervised',
                                    'model': model.upper(),
                                    'seed': seed,
                                    'label_fraction': label_frac,
                                    'auroc': metrics['auroc'],
                                    'pr_auc': metrics['pr_auc'],
                                    'accuracy': metrics.get('accuracy', np.nan),
                                    'f1': metrics.get('f1', np.nan)
                                })
                                break  # Found result, don't check other paths
    
    # Collect SSL results  
    ecg_path = results_path / "ECG"
    for ssl_model_dir, ssl_model_name in ssl_models.items():
        model_path = ecg_path / ssl_model_dir / "logistic_regression"
        if not model_path.exists():
            continue
            
        for seed in target_seeds:
            seed_path = model_path / str(seed)
            if not seed_path.exists():
                continue
                
            for label_frac in label_fractions:
                # Use consistent path structure: 10/5/1.0/test_results.json
                possible_paths = [
                    seed_path / str(label_frac) / "10" / "5" / "1.0" / "test_results.json",
                ]
                
                for result_file in possible_paths:
                    if result_file.exists():
                        metrics = load_json_results(str(result_file))
                        if metrics and 'auroc' in metrics and 'pr_auc' in metrics:
                            results.append({
                                'method_type': 'SSL',
                                'model': ssl_model_name,
                                'seed': seed,
                                'label_fraction': label_frac,
                                'auroc': metrics['auroc'],
                                'pr_auc': metrics['pr_auc'],
                                'accuracy': metrics.get('accuracy', np.nan),
                                'f1': metrics.get('f1', np.nan)
                            })
                            break  # Found result, don't check other paths

    features_path = results_path / "ECG_features" / "logistic_regression"
    if features_path.exists():
        print("Collecting ECG features results...")
        for seed in target_seeds:
            seed_path = features_path / str(seed)
            if not seed_path.exists():
                continue
                
            for label_frac in label_fractions:
                # Look for results with different window sizes and shifts
                window_configurations = [
                    (10, 5),   # 10s window, 5s shift
                    (30, 10),  # 30s window, 10s shift
                    (30, 5),   # 30s window, 5s shift
                    (30, 15),  # 30s window, 15s shift
                ]
                
                for window_size, window_shift in window_configurations:
                    result_file = seed_path / str(label_frac) / str(window_size) / str(window_shift) / "test_results.json"
                    if result_file.exists():
                        metrics = load_json_results(str(result_file))
                        if metrics and 'auroc' in metrics and 'pr_auc' in metrics:
                            results.append({
                                'method_type': 'Features',
                                'model': f'ECG_Features_{window_size}s_{window_shift}s',
                                'seed': seed,
                                'label_fraction': label_frac,
                                'window_size': window_size,
                                'window_shift': window_shift,
                                'auroc': metrics['auroc'],
                                'pr_auc': metrics['pr_auc'],
                                'accuracy': metrics.get('accuracy', np.nan),
                                'f1': metrics.get('f1', np.nan)
                            })

    df = pd.DataFrame(results)
    print(f"Collected {len(df)} result entries")
    return df

def calculate_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate mean, std dev, and std error for each model and label fraction."""
    
    # Group by method_type, model, and label_fraction
    grouped = df.groupby(['method_type', 'model', 'label_fraction'])
    
    stats_results = []
    
    for (method_type, model, label_frac), group in grouped:
        # Sort group by seed for consistent ordering
        group_sorted = group.sort_values('seed')
        
        auroc_vals = group_sorted['auroc'].values
        pr_auc_vals = group_sorted['pr_auc'].values
        seeds = sorted(group_sorted['seed'].unique())
        
        # Calculate statistics
        auroc_mean = np.mean(auroc_vals)
        pr_auc_mean = np.mean(pr_auc_vals)
        
        if len(group) >= 2:  # Need at least 2 seeds for meaningful statistics
            auroc_std = np.std(auroc_vals, ddof=1)  # Sample standard deviation
            auroc_stderr = auroc_std / np.sqrt(len(auroc_vals))  # Standard error
            pr_auc_std = np.std(pr_auc_vals, ddof=1)
            pr_auc_stderr = pr_auc_std / np.sqrt(len(pr_auc_vals))
        else:
            auroc_std = 0.0
            auroc_stderr = 0.0
            pr_auc_std = 0.0
            pr_auc_stderr = 0.0
        
        # Create base result dictionary
        result = {
            'method_type': method_type,
            'model': model,
            'label_fraction': label_frac,
            'n_seeds': len(group),
            'seeds': seeds,
            'auroc_mean': auroc_mean,
            'auroc_std': auroc_std,
            'auroc_stderr': auroc_stderr,
            'pr_auc_mean': pr_auc_mean,
            'pr_auc_std': pr_auc_std,
            'pr_auc_stderr': pr_auc_stderr
        }
        
        # Add individual seed performance columns
        # Create a mapping from seed to values for this group
        seed_to_auroc = dict(zip(group_sorted['seed'], group_sorted['auroc']))
        seed_to_pr_auc = dict(zip(group_sorted['seed'], group_sorted['pr_auc']))
        
        # Add columns for each seed (up to 5 seeds: 3, 5, 7, 9, 42)
        all_seeds = [3, 5, 7, 9, 42]
        for i, seed in enumerate(all_seeds, 1):
            if seed in seed_to_auroc:
                result[f'auroc_seed_{seed}'] = seed_to_auroc[seed]
                result[f'pr_auc_seed_{seed}'] = seed_to_pr_auc[seed]
            else:
                result[f'auroc_seed_{seed}'] = np.nan
                result[f'pr_auc_seed_{seed}'] = np.nan
        
        stats_results.append(result)
    
    return pd.DataFrame(stats_results)


def wilcoxon_signed_rank_test(df: pd.DataFrame, stats_df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform Wilcoxon signed-rank test to compare methods across seeds for each label fraction.

    Tests performed:
    - TSTCC vs SimCLR
    - TSTCC vs TS2Vec
    - SimCLR vs TS2Vec

    For label fractions: 10%, 25%, 50%, 100%
    For metrics: AUROC and PR-AUC

    Uses small sample case (n=5 seeds) with critical values from Wilcoxon table.
    """

    # Critical values for Wilcoxon signed-rank test (n=5, two-tailed)
    # From table: n=5, α=0.10 (two-tailed) = 1, α=0.05 (two-tailed) = not in table
    # We'll use α=0.10 (two-tailed) which corresponds to α=0.05 (one-tailed)
    # We take the critical value to be 0
    # See: https: // users.stat.ufl.edu / ~winner / tables / wilcox_signrank.pdf
    # Or here: https://www.saskoer.ca/app/uploads/sites/313/2020/11/Wilcoxon-Signed-Rank-Test-Critical-Values-Table.pdf
    critical_value_n5_alpha010 = 0

    target_fractions = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
    target_seeds = [3, 5, 7, 9, 42]

    # Method pairs to compare
    comparisons = [
        # ('TSTCC', 'SimCLR'),
        # ('TSTCC', 'TS2Vec'),
        # ('TSTCC', 'SimCLR_S3'),
        # ('TSTCC', 'TS2Vec'),
        # ('SimCLR', 'TS2Vec'),
        # ('TS2Vec', 'SimCLR'),
        # ('TS2Vec', 'SimCLR_S3'),
        # ('TSTCC', 'TSTCC_S3'),
        # ('TS2Vec', 'TS2Vec_S3'),
        # ('SimCLR', 'SimCLR_S3'),
        # ('InfoTS', 'InfoTS_S3'),
        ('TSTCC', 'ECG_Features_10s_5s'),
        ('TSTCC_S3', 'ECG_Features_10s_5s'),
        # ('TSTCC', 'ECG_Features_30s_10s'),
        # ('TSTCC', 'ECG_Features_30s_15s'),
        # ('TSTCC_S3', 'ECG_Features_30s_5s'),
        # ('TSTCC_S3', 'ECG_Features_30s_10s'),
        # ('TSTCC_S3', 'ECG_Features_30s_15s'),
    ]

    metrics = ['auroc', 'pr_auc']

    results = []

    print("\n" + "="*80)
    print("WILCOXON SIGNED-RANK TEST RESULTS")
    print("="*80)
    print("\nCritical value for n=5, two-tailed α=0.10: W_crit = 1")
    print("Reject H₀ (equal means) if W ≤ 0")
    print("="*80)

    for label_frac in target_fractions:
        print(f"\n{'='*80}")
        print(f"LABEL FRACTION: {label_frac*100:.0f}%")
        print(f"{'='*80}")

        for method1, method2 in comparisons:
            print(f"\n{'-'*80}")
            print(f"Comparison: {method1} vs {method2}")
            print(f"{'-'*80}")

            for metric in metrics:
                # Extract data for both methods across all seeds
                method1_data = []
                method2_data = []

                for seed in target_seeds:
                    # Filter for specific method, label fraction, and seed
                    # Don't filter by method_type to allow comparisons between SSL and Supervised
                    m1_row = df[
                        (df['model'] == method1) &
                        (df['label_fraction'] == label_frac) &
                        (df['seed'] == seed)
                    ]

                    m2_row = df[
                        (df['model'] == method2) &
                        (df['label_fraction'] == label_frac) &
                        (df['seed'] == seed)
                    ]

                    if len(m1_row) > 0 and len(m2_row) > 0:
                        method1_data.append(m1_row.iloc[0][metric])
                        method2_data.append(m2_row.iloc[0][metric])

                # Check if we have complete data
                if len(method1_data) != 5 or len(method2_data) != 5:
                    print(f"\n{metric.upper()}: Incomplete data (n={len(method1_data)}), skipping")
                    continue

                # Convert to numpy arrays
                method1_vals = np.array(method1_data)
                method2_vals = np.array(method2_data)

                # Calculate differences: D = method1 - method2
                differences = method1_vals - method2_vals

                # Calculate absolute differences
                abs_diff = np.abs(differences)

                # Rank the absolute differences (excluding zeros)
                non_zero_indices = abs_diff > 0
                if np.sum(non_zero_indices) == 0:
                    print(f"\n{metric.upper()}: All differences are zero, methods are identical")
                    continue

                # Create ranking for non-zero differences
                non_zero_abs_diff = abs_diff[non_zero_indices]
                ranks = np.zeros_like(abs_diff)

                # Rank from smallest to largest absolute difference
                sorted_indices = np.argsort(non_zero_abs_diff)
                for rank, idx in enumerate(sorted_indices, 1):
                    original_idx = np.where(non_zero_indices)[0][idx]
                    ranks[original_idx] = rank

                # Apply signs to ranks
                signed_ranks = np.sign(differences) * ranks

                # Calculate W+ and W-
                w_plus = np.sum(signed_ranks[signed_ranks > 0])
                w_minus = np.abs(np.sum(signed_ranks[signed_ranks < 0]))

                # Test statistic W = min(W+, |W-|)
                w_statistic = min(w_plus, w_minus)

                # Decision
                reject_h0 = w_statistic <= critical_value_n5_alpha010

                # Determine which method is better
                if w_plus > w_minus:
                    better_method = method1
                elif w_minus > w_plus:
                    better_method = method2
                else:
                    better_method = "Tie"

                # Print detailed results
                print(f"\n{metric.upper()}:")
                print(f"  Seeds: {target_seeds}")
                print(f"  {method1}: {[f'{v:.3f}' for v in method1_vals]}")
                print(f"  {method2}: {[f'{v:.3f}' for v in method2_vals]}")
                print(f"  Differences (D): {[f'{d:+.3f}' for d in differences]}")
                print(f"  |D|: {[f'{d:.3f}' for d in abs_diff]}")
                print(f"  Ranks: {ranks.astype(int)}")
                print(f"  Signed Ranks: {signed_ranks.astype(int)}")
                print(f"  W⁺ = {w_plus:.0f}, W⁻ = {w_minus:.0f}")
                print(f"  W = min(W⁺, W⁻) = {w_statistic:.0f}")
                print(f"  Critical value (α=0.10, two-tailed): {critical_value_n5_alpha010}")

                if reject_h0:
                    print(f"  ✓ REJECT H₀: W = {w_statistic:.0f} ≤ {critical_value_n5_alpha010}")
                    print(f"  → {better_method} is SIGNIFICANTLY BETTER (p < 0.10)")
                else:
                    print(f"  ✗ FAIL TO REJECT H₀: W = {w_statistic:.0f} > {critical_value_n5_alpha010}")
                    print(f"  → No significant difference (p ≥ 0.10)")

                # Also perform scipy's wilcoxon test for verification
                try:
                    # Remove zero differences for scipy implementation
                    non_zero_diff = differences[differences != 0]
                    if len(non_zero_diff) > 0:
                        scipy_result = stats.wilcoxon(non_zero_diff, alternative='two-sided')
                        print(f"  [Scipy verification: W={scipy_result.statistic:.0f}, p={scipy_result.pvalue:.4f}]")
                except Exception as e:
                    print(f"  [Scipy verification failed: {e}]")

                # Store results
                results.append({
                    'label_fraction': label_frac,
                    'comparison': f'{method1} vs {method2}',
                    'metric': metric.upper(),
                    'method1': method1,
                    'method2': method2,
                    'method1_mean': np.mean(method1_vals),
                    'method2_mean': np.mean(method2_vals),
                    'mean_difference': np.mean(differences),
                    'w_plus': w_plus,
                    'w_minus': w_minus,
                    'w_statistic': w_statistic,
                    'critical_value': critical_value_n5_alpha010,
                    'reject_h0': reject_h0,
                    'significant': reject_h0,
                    'better_method': better_method if reject_h0 else 'No significant difference',
                    'n_seeds': len(method1_data)
                })

    # Create summary DataFrame
    results_df = pd.DataFrame(results)

    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)

    if len(results_df) > 0:
        for label_frac in target_fractions:
            frac_results = results_df[results_df['label_fraction'] == label_frac]
            if len(frac_results) == 0:
                continue

            print(f"\nLabel Fraction: {label_frac*100:.0f}%")
            print(f"{'-'*80}")
            print(f"{'Comparison':<25} {'Metric':<10} {'Significant':<12} {'Better Method':<20}")
            print(f"{'-'*80}")

            for _, row in frac_results.iterrows():
                sig_str = "YES" if row['significant'] else "NO"
                print(f"{row['comparison']:<25} {row['metric']:<10} {sig_str:<12} {row['better_method']:<20}")

    return results_df

def export_to_csv(stats_df: pd.DataFrame, output_path: str) -> None:
    """Export results to CSV for further analysis."""
    # Convert percentages and format columns
    export_df = stats_df.copy()
    
    # Convert to percentages - main statistics
    for col in ['auroc_mean', 'auroc_std', 'auroc_stderr', 'pr_auc_mean', 'pr_auc_std', 'pr_auc_stderr']:
        export_df[col] = export_df[col]
    
    # Convert to percentages - individual seed columns
    all_seeds = [3, 5, 7, 9, 42]
    for seed in all_seeds:
        auroc_col = f'auroc_seed_{seed}'
        pr_auc_col = f'pr_auc_seed_{seed}'
        if auroc_col in export_df.columns:
            export_df[auroc_col] = export_df[auroc_col]
        if pr_auc_col in export_df.columns:
            export_df[pr_auc_col] = export_df[pr_auc_col]
    
    # Add formatted strings for LaTeX
    export_df['auroc_latex'] = export_df.apply(
        lambda x: f"{x['auroc_mean']:.1f}±{x['auroc_std']:.2f}", axis=1
    )
    export_df['pr_auc_latex'] = export_df.apply(
        lambda x: f"{x['pr_auc_mean']:.1f}±{x['pr_auc_std']:.2f}", axis=1
    )
    
    export_df.to_csv(output_path, index=False, float_format='%.3f')
    print(f"\nResults exported to: {output_path}")

def main():
    """Main function to run the aggregation."""
    output_csv = "ecg_ssl_aggregated_results.csv"
    
    if not os.path.exists(RESULTS_PATH):
        print(f"Results directory '{RESULTS_PATH}' not found!")
        return
    
    print("Starting ECG SSL results aggregation...")
    
    # Collect all results
    df = collect_results(RESULTS_PATH)
    
    if len(df) == 0:
        print("No results found!")
        return
    
    print(f"\nFound results for models: {sorted(df['model'].unique())}")
    print(f"Label fractions: {sorted(df['label_fraction'].unique())}")
    print(f"Seeds: {sorted(df['seed'].unique())}")
    
    # Calculate statistics
    print("\nCalculating statistics...")
    stats_df = calculate_statistics(df)

    # Perform Wilcoxon signed-rank tests
    print("\n" + "="*80)
    print("PERFORMING STATISTICAL SIGNIFICANCE TESTS")
    print("="*80)
    wilcoxon_results_df = wilcoxon_signed_rank_test(df, stats_df)

    # Export main results to CSV
    export_to_csv(stats_df, output_csv)

    # Export Wilcoxon test results to separate CSV
    if len(wilcoxon_results_df) > 0:
        wilcoxon_output_csv = "ecg_ssl_wilcoxon_test_results.csv"
        wilcoxon_results_df.to_csv(
            os.path.join(RESULTS_PATH, wilcoxon_output_csv), index=False, float_format='%.4f'
        )
        print(f"\nWilcoxon test results exported to: {wilcoxon_output_csv}")

    print(f"\n{'='*80}")
    print("SUMMARY")
    print("="*80)
    print(f"Total result entries processed: {len(df)}")
    print(f"Unique model-fraction combinations: {len(stats_df)}")
    print(f"Results saved to: {output_csv}")
    if len(wilcoxon_results_df) > 0:
        print(f"Wilcoxon test results saved to: ecg_ssl_wilcoxon_test_results.csv")
        print(f"Total statistical tests performed: {len(wilcoxon_results_df)}")

if __name__ == "__main__":
    main()