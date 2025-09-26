import argparse
import os
import json
import numpy as np
import pandas as pd

from utils.helper_paths import RESULTS_PATH

def load_results(
        base_path, transfer_path_approach, model_type, seeds, label_fraction=1.0, window_size=10, step_size=5, is_zero_shot=False
):
    """Load test results for a specific model type."""
    results = []
    
    for seed in seeds:
        if is_zero_shot:
            # For zero shot, files are named zero_shot_results.json
            if model_type.startswith('feature_engineered'):
                # feature_engineered_ws30_ss10 -> ws=30, ss=10
                # feature_engineered_ws10_ss5 -> ws=10, ss=5
                parts = model_type.split('_')
                ws = int(parts[2][2:])  # Extract window size from 'ws30'
                ss = int(parts[3][2:])  # Extract step size from 'ss10'
                file_path = os.path.join(base_path, transfer_path_approach, 'feature_engineered',
                                       'logistic_regression', str(seed), str(label_fraction), 
                                       str(ws), str(ss), 'zero_shot_results.json')
            elif model_type == 'cnn':
                file_path = os.path.join(base_path, transfer_path_approach, 'cnn', str(seed), str(label_fraction),
                                       'zero_shot_results.json')
            else:
                # TSTCC, TSTCC_S3, etc.
                file_path = os.path.join(base_path, transfer_path_approach, model_type,
                                       'logistic_regression', str(seed), str(label_fraction), 
                                       'zero_shot_results.json')
        else:
            # Original transfer learning paths
            if model_type == 'cnn':
                file_path = os.path.join(base_path, transfer_path_approach, 'cnn', str(seed), str(label_fraction),
                                       str(window_size), str(step_size), 'test_results.json')
            elif model_type in ['TSTCC', 'TSTCC_S3']:
                file_path = os.path.join(base_path, transfer_path_approach, model_type,
                                       'logistic_regression', str(seed), str(label_fraction), 
                                       str(window_size), str(step_size), 'test_results.json')
            else:
                # Generic path for other models
                file_path = os.path.join(base_path, transfer_path_approach, model_type,
                                       'logistic_regression', str(seed), str(label_fraction), 
                                       str(window_size), str(step_size), 'test_results.json')
        
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)
                if is_zero_shot:
                    # Zero shot results have different structure
                    result_dict = {
                        'seed': seed,
                        'auroc': data['zero_shot_roc_auc'],
                        'pr_auc': data['zero_shot_pr_auc'],
                        'accuracy': data.get('zero_shot_accuracy', None),
                    }
                    # Some models don't have all metrics
                    if 'zero_shot_balanced_accuracy' in data:
                        result_dict['balanced_accuracy'] = data['zero_shot_balanced_accuracy']
                    if 'zero_shot_f1_score' in data:
                        result_dict['f1_score'] = data['zero_shot_f1_score']
                    results.append(result_dict)
                else:
                    # Transfer learning results
                    test_metrics = data['test_metrics']
                    results.append({
                        'seed': seed,
                        'auroc': test_metrics['auroc'],
                        'pr_auc': test_metrics['pr_auc']
                    })
        else:
            print(f"Missing file: {file_path}")
    
    return results

def compute_stats(values):
    """Compute mean, std, stderr, and n for a list of values."""
    return {
        'mean': np.mean(values),
        'std': np.std(values),
        'se': np.std(values)/np.sqrt(len(values)),
        'n': len(values)
    }

def main(base_path, transfer_path_approach, models_to_compare, is_zero_shot=False):
    all_results = []
    summary_stats = []
    aggregated_results = {}  # For JSON output similar to averaged_zero_shot_results
    seeds = [3, 5, 7, 9, 42]

    for model_type in models_to_compare:
        results = load_results(base_path, transfer_path_approach, model_type, seeds, is_zero_shot=is_zero_shot)
        
        if results:
            auroc_values = [r['auroc'] for r in results]
            pr_auc_values = [r['pr_auc'] for r in results]
            
            if is_zero_shot:
                # Extract available metrics, handling missing values
                accuracy_values = [r['accuracy'] for r in results if r.get('accuracy') is not None]
                balanced_accuracy_values = [r['balanced_accuracy'] for r in results if r.get('balanced_accuracy') is not None]
                f1_values = [r['f1_score'] for r in results if r.get('f1_score') is not None]
                
                # Create aggregated results structure for zero shot
                aggregated_results[model_type] = {
                    'zero_shot_roc_auc': compute_stats(auroc_values), 
                    'zero_shot_pr_auc': compute_stats(pr_auc_values)
                }
                
                if accuracy_values:
                    aggregated_results[model_type]['zero_shot_accuracy'] = compute_stats(accuracy_values)
                if balanced_accuracy_values:
                    aggregated_results[model_type]['zero_shot_balanced_accuracy'] = compute_stats(balanced_accuracy_values)
                if f1_values:
                    aggregated_results[model_type]['zero_shot_f1_score'] = compute_stats(f1_values)
                
                # Print results
                print(f"{model_type} Results:")
                print(f"AUROC: {np.mean(auroc_values):.3f} ± {np.std(auroc_values):.3f} (stderr: {np.std(auroc_values)/np.sqrt(len(auroc_values)):.4f})")
                print(f"PR-AUC: {np.mean(pr_auc_values):.3f} ± {np.std(pr_auc_values):.3f} (stderr: {np.std(pr_auc_values)/np.sqrt(len(pr_auc_values)):.4f})")
                if accuracy_values:
                    print(f"Accuracy: {np.mean(accuracy_values):.3f} ± {np.std(accuracy_values):.3f} (stderr: {np.std(accuracy_values)/np.sqrt(len(accuracy_values)):.4f})")
                if balanced_accuracy_values:
                    print(f"Balanced Accuracy: {np.mean(balanced_accuracy_values):.3f} ± {np.std(balanced_accuracy_values):.3f} (stderr: {np.std(balanced_accuracy_values)/np.sqrt(len(balanced_accuracy_values)):.4f})")
                if f1_values:
                    print(f"F1 Score: {np.mean(f1_values):.3f} ± {np.std(f1_values):.3f} (stderr: {np.std(f1_values)/np.sqrt(len(f1_values)):.4f})")
                print(f"Seeds: {[r['seed'] for r in results]} (n={len(results)})")
                print()
                
                # Add to individual results
                for r in results:
                    all_results.append({'model': model_type, **r})
                
                # Add to summary stats (extended for zero shot) - only for available metrics
                summary_stats.extend([
                    {
                        'model': model_type,
                        'metric': 'AUROC',
                        'mean': np.mean(auroc_values),
                        'std': np.std(auroc_values),
                        'stderr': np.std(auroc_values)/np.sqrt(len(auroc_values)),
                        'n_seeds': len(auroc_values)
                    },
                    {
                        'model': model_type,
                        'metric': 'PR_AUC',
                        'mean': np.mean(pr_auc_values),
                        'std': np.std(pr_auc_values),
                        'stderr': np.std(pr_auc_values)/np.sqrt(len(pr_auc_values)),
                        'n_seeds': len(pr_auc_values)
                    }
                ])
                
                if accuracy_values:
                    summary_stats.append({
                        'model': model_type,
                        'metric': 'Accuracy',
                        'mean': np.mean(accuracy_values),
                        'std': np.std(accuracy_values),
                        'stderr': np.std(accuracy_values)/np.sqrt(len(accuracy_values)),
                        'n_seeds': len(accuracy_values)
                    })
                
                if balanced_accuracy_values:
                    summary_stats.append({
                        'model': model_type,
                        'metric': 'Balanced_Accuracy',
                        'mean': np.mean(balanced_accuracy_values),
                        'std': np.std(balanced_accuracy_values),
                        'stderr': np.std(balanced_accuracy_values)/np.sqrt(len(balanced_accuracy_values)),
                        'n_seeds': len(balanced_accuracy_values)
                    })
                
                if f1_values:
                    summary_stats.append({
                        'model': model_type,
                        'metric': 'F1_Score',
                        'mean': np.mean(f1_values),
                        'std': np.std(f1_values),
                        'stderr': np.std(f1_values)/np.sqrt(len(f1_values)),
                        'n_seeds': len(f1_values)
                    })
            else:
                # Original transfer learning logic
                # Print results
                print(f"{model_type} Results:")
                print(f"AUROC: {np.mean(auroc_values):.3f} ± {np.std(auroc_values):.3f} (stderr: {np.std(auroc_values)/np.sqrt(len(auroc_values)):.4f})")
                print(f"PR-AUC: {np.mean(pr_auc_values):.3f} ± {np.std(pr_auc_values):.3f} (stderr: {np.std(pr_auc_values)/np.sqrt(len(pr_auc_values)):.4f})")
                print(f"Seeds: {[r['seed'] for r in results]} (n={len(results)})")
                print()
                
                # Add to individual results
                for r in results:
                    all_results.append({'model': model_type, **r})
                
                # Add to summary stats
                summary_stats.extend([
                    {
                        'model': model_type,
                        'metric': 'AUROC',
                        'mean': np.mean(auroc_values),
                        'std': np.std(auroc_values),
                        'stderr': np.std(auroc_values)/np.sqrt(len(auroc_values)),
                        'n_seeds': len(auroc_values)
                    },
                    {
                        'model': model_type,
                        'metric': 'PR_AUC',
                        'mean': np.mean(pr_auc_values),
                        'std': np.std(pr_auc_values),
                        'stderr': np.std(pr_auc_values)/np.sqrt(len(pr_auc_values)),
                        'n_seeds': len(pr_auc_values)
                    }
                ])
    
    # Save individual results
    df_individual = pd.DataFrame(all_results)
    suffix = '_zero_shot' if is_zero_shot else ''
    individual_output = os.path.join(base_path, transfer_path_approach, f'individual_results_comparison{suffix}.csv')
    df_individual.to_csv(individual_output, index=False)
    print(f"Individual results saved to: {individual_output}")
    
    # Save summary statistics
    df_summary = pd.DataFrame(summary_stats)
    summary_output = os.path.join(base_path, transfer_path_approach, f'summary_statistics_comparison{suffix}.csv')
    df_summary.to_csv(summary_output, index=False)
    print(f"Summary statistics saved to: {summary_output}")
    
    # Save aggregated JSON results for zero shot
    if is_zero_shot and aggregated_results:
        dataset_name = "WESAD" if "WESAD" in base_path else "StressID"
        json_output = os.path.join(base_path, transfer_path_approach, f'aggregated_zero_shot_results_{dataset_name}.json')
        with open(json_output, 'w') as f:
            json.dump(aggregated_results, f, indent=2)
        print(f"Aggregated JSON results saved to: {json_output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get summary results for transfer learning results.")
    parser.add_argument("--dataset", default="wesad", type=str, choices=("wesad", "stressid"))
    parser.add_argument("--fine_tune_strategy", type=str, default="head_only",
                        choices=("head_only", "full", "zero_shot"))

    args = parser.parse_args()

    dataset_name = "WESAD" if args.dataset == "wesad" else "StressID"
    is_zero_shot = args.fine_tune_strategy == "zero_shot"
    
    if is_zero_shot:
        transfer_path_approach = "zero_shot_performance"
        # For zero shot, include feature engineered models with different window/step sizes
        models_to_compare = ['cnn', 'TSTCC', 'TSTCC_S3', 'feature_engineered_ws30_ss10', 'feature_engineered_ws10_ss5']
        print("===Zero Shot Results Comparison ===")
    else:
        transfer_path_approach = "fine_tuned_encoder_new_head" if args.fine_tune_strategy == "full" \
            else "pretrained_encoder_new_head"
        models_to_compare = ['cnn', 'TSTCC', 'TSTCC_S3']
        print("===Transfer Learning Results Comparison ===")
        print(f"Label fraction: 1.0, Window size: 10/5")

    base_path = os.path.join(RESULTS_PATH, "Transfer_learning", dataset_name)
    print()

    main(base_path, transfer_path_approach, models_to_compare, is_zero_shot=is_zero_shot)