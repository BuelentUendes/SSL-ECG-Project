import os
import math
import argparse
import json
import pathlib

from utils.helper_paths import RESULTS_PATH
import numpy as np


def load_json_results(path, seed=42, label_fraction=1.0, window_size=30, step_size=10):
    if str(path).split("/")[-1] == "feature_engineered":
        json_file_path = os.path.join(path, "logistic_regression", str(seed), str(label_fraction), str(window_size), str(step_size), "zero_shot_results.json")

    elif str(path).split("/")[-1] == "cnn":
        json_file_path = os.path.join(path, str(seed), str(label_fraction), "zero_shot_results.json")
    else:
        json_file_path = os.path.join(path, "logistic_regression", str(seed), str(label_fraction), "zero_shot_results.json")

    if not os.path.exists(json_file_path):
        raise FileNotFoundError(f"Results file not found: {json_file_path}")
    
    with open(json_file_path) as f:
        return json.load(f)


def average_metrics_across_seeds(results):
    """
    Averages performance metrics across all seeds for each method.

    Args:
        results (dict): Nested dictionary with structure:
                       {method: {seed: {metric: value}}}

    Returns:
        dict: Dictionary with structure {method: {metric: {mean, std, se}}}
    """
    averaged_results = {}

    for method, seeds_data in results.items():
        # Initialize method dictionary
        averaged_results[method] = {}
        
        # Get all unique metric names across all seeds for this method
        seeds = list(seeds_data.keys())

        all_metrics = set()
        for seed_data in seeds_data.values():
            all_metrics.update(seed_data.keys())

        for metric in all_metrics:
            # Safely extract values, skipping seeds where metric doesn't exist
            method_values = []
            for seed in seeds:
                if metric in seeds_data[seed]:
                    method_values.append(seeds_data[seed][metric])
            
            if len(method_values) > 0:
                mean_val = np.mean(method_values)
                std_val = np.std(method_values, ddof=1) if len(method_values) > 1 else 0.0
                se_val = std_val / np.sqrt(len(method_values)) if len(method_values) > 1 else 0.0
                
                averaged_results[method][metric] = {
                    "mean": mean_val,
                    "std": std_val, 
                    "se": se_val,
                    "n": len(method_values)
                }

    return averaged_results



def main(args):
    dataset_name = "WESAD" if args.dataset == "wesad" else "StressID"
    results_path = os.path.join(RESULTS_PATH, "Transfer_learning", dataset_name, "zero_shot_performance")
    seeds = [3,5,7,9,42]
    zero_shot_results = {}
    # Get all model_names
    folder_paths = list(pathlib.Path(results_path).iterdir())
    model_names = [str(path).split("/")[-1] for path in folder_paths]

    # Define (window_size, step_size) pairs for feature-engineered
    window_step_pairs = [(30, 10), (10, 5)]
    
    for model_path, model in zip(folder_paths, model_names):
        if model == "feature_engineered":
            # For feature-engineered, load results for different window/step combinations
            for window_size, step_size in window_step_pairs:
                model_key = f"{model}_ws{window_size}_ss{step_size}"
                model_results = {}
                seeds_found = 0
                
                for seed in seeds:
                    try:
                        model_results[str(seed)] = load_json_results(model_path, str(seed), label_fraction=1.0, 
                                                                   window_size=window_size, step_size=step_size)
                        seeds_found += 1
                    except FileNotFoundError:
                        print(f"Warning: Missing results for {model_key}, seed={seed}")
                        continue
                
                if seeds_found > 0:
                    zero_shot_results[model_key] = model_results
                    print(f"✓ Loaded {model_key}: {seeds_found}/{len(seeds)} seeds found")
                else:
                    print(f"✗ Skipping {model_key}: No results found for any seed")
        else:
            # For other models, use default parameters
            model_results = {}
            seeds_found = 0
            
            for seed in seeds:
                try:
                    model_results[str(seed)] = load_json_results(model_path, str(seed))
                    seeds_found += 1
                except FileNotFoundError:
                    print(f"Warning: Missing results for {model}, seed={seed}")
                    continue
            
            if seeds_found > 0:
                zero_shot_results[model] = model_results
                print(f"✓ Loaded {model}: {seeds_found}/{len(seeds)} seeds found")
            else:
                print(f"✗ Skipping {model}: No results found for any seed")

    averaged_zero_shot_results = average_metrics_across_seeds(zero_shot_results)

    save_path = os.path.join(RESULTS_PATH, "Transfer_learning", dataset_name, "zero_shot_performance")
    save_name = os.path.join(save_path, f"averaged_zero_shot_results_{dataset_name}.json")
    with open(save_name, "w") as f:
        json.dump(averaged_zero_shot_results, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze the zero_shot_results")
    parser.add_argument("--dataset", default="stressid", type=str, help="Which dataset to analyze")
    args = parser.parse_args()
    main(args)

