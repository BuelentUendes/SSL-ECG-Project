#!/usr/bin/env python3
"""
Script to aggregate ECG SSL results for LaTeX table generation.
Extracts results from multiple models, seeds, and label fractions to calculate
mean ± std dev and std error for AUROC and PR-AUC metrics.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
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
    
    def find_result_files(base_path: Path, expected_depth: int = 0) -> List[Path]:
        """Recursively find test_results.json files at any depth."""
        result_files = []
        if base_path.is_file() and base_path.name == "test_results.json":
            result_files.append(base_path)
        elif base_path.is_dir():
            for child in base_path.iterdir():
                result_files.extend(find_result_files(child, expected_depth + 1))
        return result_files
    
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

def format_for_latex(stats_df: pd.DataFrame) -> None:
    """Format and print results for LaTeX table integration."""
    
    print("\n" + "="*80)
    print("RESULTS FORMATTED FOR LATEX TABLE")
    print("="*80)
    
    # Define the label fractions we want in the table
    target_fractions = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
    
    for frac in target_fractions:
        frac_data = stats_df[stats_df['label_fraction'] == frac]
        if len(frac_data) == 0:
            continue
            
        print(f"\n{'='*60}")
        print(f"LABEL FRACTION: {frac*100:.1f}%")
        print(f"{'='*60}")
        
        # Get all methods for this fraction
        supervised_data = frac_data[frac_data['method_type'] == 'Supervised'].sort_values('model')
        ssl_data = frac_data[frac_data['method_type'] == 'SSL']
        features_data = frac_data[frac_data['method_type'] == 'Features']
        
        # Print table header
        print(f"{'Method':<25} {'AUROC':<15} {'PR-AUC':<15} {'Seeds':<8}")
        print("-" * 65)
        
        # Print supervised methods
        if len(supervised_data) > 0:
            print("SUPERVISED:")
            for _, row in supervised_data.iterrows():
                auroc_str = f"{row['auroc_mean']:.1f}±{row['auroc_std']:.2f}"
                pr_auc_str = f"{row['pr_auc_mean']:.1f}±{row['pr_auc_std']:.2f}"
                print(f"  {row['model']:<23} {auroc_str:<15} {pr_auc_str:<15} {row['n_seeds']:<8}")
        
        # Print features methods
        if len(features_data) > 0:
            print("\nFEATURES:")
            for _, row in features_data.iterrows():
                auroc_str = f"{row['auroc_mean']:.1f}±{row['auroc_std']:.2f}"
                pr_auc_str = f"{row['pr_auc_mean']:.1f}±{row['pr_auc_std']:.2f}"
                print(f"  {row['model']:<23} {auroc_str:<15} {pr_auc_str:<15} {row['n_seeds']:<8}")
        
        # Print SSL methods organized by base method
        if len(ssl_data) > 0:
            print("\nSSL METHODS:")
            
            # Group by base method (before S3)
            ssl_methods = {}
            for _, row in ssl_data.iterrows():
                base_method = row['model'].replace('_S3', '')
                if base_method not in ssl_methods:
                    ssl_methods[base_method] = {'base': None, 's3': None}
                
                if row['model'].endswith('_S3'):
                    ssl_methods[base_method]['s3'] = row
                else:
                    ssl_methods[base_method]['base'] = row
            
            # Sort methods for consistent output
            for base_method in sorted(ssl_methods.keys()):
                variants = ssl_methods[base_method]
                base_row = variants['base']
                s3_row = variants['s3']
                
                if base_row is not None:
                    auroc_str = f"{base_row['auroc_mean']:.1f}±{base_row['auroc_std']:.2f}"
                    pr_auc_str = f"{base_row['pr_auc_mean']:.1f}±{base_row['pr_auc_std']:.2f}"
                    print(f"  {base_method:<23} {auroc_str:<15} {pr_auc_str:<15} {base_row['n_seeds']:<8}")
                
                if s3_row is not None:
                    auroc_str = f"{s3_row['auroc_mean']:.1f}±{s3_row['auroc_std']:.2f}"
                    pr_auc_str = f"{s3_row['pr_auc_mean']:.1f}±{s3_row['pr_auc_std']:.2f}"
                    print(f"  {base_method}_S3{'':<23} {auroc_str:<15} {pr_auc_str:<15} {s3_row['n_seeds']:<8}")
        
        print()  # Empty line after each fraction

def export_to_csv(stats_df: pd.DataFrame, output_path: str) -> None:
    """Export results to CSV for further analysis."""
    # Convert percentages and format columns
    export_df = stats_df.copy()
    
    # Convert to percentages - main statistics
    for col in ['auroc_mean', 'auroc_std', 'auroc_stderr', 'pr_auc_mean', 'pr_auc_std', 'pr_auc_stderr']:
        export_df[col] = export_df[col] * 100
    
    # Convert to percentages - individual seed columns
    all_seeds = [3, 5, 7, 9, 42]
    for seed in all_seeds:
        auroc_col = f'auroc_seed_{seed}'
        pr_auc_col = f'pr_auc_seed_{seed}'
        if auroc_col in export_df.columns:
            export_df[auroc_col] = export_df[auroc_col] * 100
        if pr_auc_col in export_df.columns:
            export_df[pr_auc_col] = export_df[pr_auc_col] * 100
    
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
    results_dir = "results"
    output_csv = "ecg_ssl_aggregated_results.csv"
    
    if not os.path.exists(results_dir):
        print(f"Results directory '{results_dir}' not found!")
        return
    
    print("Starting ECG SSL results aggregation...")
    
    # Collect all results
    df = collect_results(results_dir)
    
    if len(df) == 0:
        print("No results found!")
        return
    
    print(f"\nFound results for models: {sorted(df['model'].unique())}")
    print(f"Label fractions: {sorted(df['label_fraction'].unique())}")
    print(f"Seeds: {sorted(df['seed'].unique())}")
    
    # Calculate statistics
    print("\nCalculating statistics...")
    stats_df = calculate_statistics(df)
    
    # Format for LaTeX
    format_for_latex(stats_df)
    
    # Export to CSV
    export_to_csv(stats_df, output_csv)
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print("="*80)
    print(f"Total result entries processed: {len(df)}")
    print(f"Unique model-fraction combinations: {len(stats_df)}")
    print(f"Results saved to: {output_csv}")
    print("\nYou can now copy the formatted results above into your LaTeX table!")

if __name__ == "__main__":
    main()