import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from utils.helper_paths import RESULTS_PATH, FIGURES_PATH
from utils.torch_utilities import create_directory
from pathlib import Path


def load_results_from_structure(base_path):
    """
    Load results from the hierarchical folder structure.
    Expected structure: results/ECG/[method]/[model_type]/[seed]/[label_fraction]/[window_size]/[window_shift]/test_results.json
                       results/ECG_features/[model_type]/[seed]/[label_fraction]/test_results.json
    """
    results = []
    base_path = Path(base_path)

    # Define the expected structure paths
    paths_to_check = [
        "ECG/Supervised/cnn",
        # "ECG/Supervised/tcn",
        # "ECG/Supervised/transformer",
        # "ECG/Supervised/deep_ecg_net",
        "ECG/TSTCC/logistic_regression",
        # "ECG/TSTCC/mlp",
        "ECG/TSTCC_S3/logistic_regression",
        # "ECG/TSTCC/xgboost",
        "ECG_features/logistic_regression",
    ]

    for path_str in paths_to_check:
        method_path = base_path / path_str

        if not method_path.exists():
            print(f"Warning: Path {method_path} does not exist")
            continue

        # Parse method info from path
        path_parts = Path(path_str).parts
        if "ECG_features" in path_parts:
            method_type = "ECG_features"
            model_type = path_parts[-1] 
            learning_method = "Feature-engineered"
        else:
            method_type = "ECG"
            learning_method = path_parts[1]  # Supervised or TSTCC
            if learning_method == "TSTCC_S3":
                learning_method = "TSTCC+S3"
            model_type = path_parts[2]

        for seed_folder in method_path.iterdir():
            if seed_folder.is_dir():
                # Handle both numeric seeds (3, 5, 7, 9, 42) and special seeds like "42_s3"
                if seed_folder.name.isdigit():
                    seed = int(seed_folder.name)
                else:
                    continue

                # Look for label fraction folders
                for label_folder in seed_folder.iterdir():
                    if label_folder.is_dir():
                        try:
                            label_fraction = float(label_folder.name)

                            window_combinations = [
                                (10, 5),   # 10s windows, 5s shift
                                # (30, 5),   # 30s windows, 5s shift
                                # (30, 10), # 30s windows, 10s shift
                                # (30, 15)   # 30s windows, 15s shift
                            ]

                            for window_size, window_shift in window_combinations:
                                if learning_method == "TSTCC" and window_size == 30:
                                    continue
                                # Use consistent path structure: always use nested 1.0 folder for TSTCC methods
                                if model_type in ["logistic_regression", "mlp"] and learning_method in ["TSTCC", "TSTCC+S3"]:
                                    json_file = label_folder / str(window_size) / str(window_shift) / "1.0" / "test_results.json"
                                elif model_type == "xgboost":
                                    json_file = label_folder / str(window_size) / str(window_shift) / "0.75" / "test_results.json"
                                else:
                                    json_file = label_folder / str(window_size) / str(window_shift) / "test_results.json"
                                
                                if json_file.exists():
                                    try:
                                        with open(json_file, 'r') as f:
                                            data = json.load(f)
                                        results.append({
                                            'method_type': method_type,
                                            'learning_method': learning_method,
                                            'model_type': model_type,
                                            'seed': seed,
                                            'label_fraction': label_fraction,
                                            'window_size': window_size,
                                            'window_shift': window_shift,
                                            'auroc': data["test_metrics"].get('auroc', np.nan),
                                            'accuracy': data["test_metrics"].get('accuracy', np.nan),
                                            'pr_auc': data["test_metrics"].get('pr_auc', np.nan),
                                            'f1': data["test_metrics"].get('f1', np.nan),
                                        })
                                    except (json.JSONDecodeError, KeyError) as e:
                                        print(f"Error reading {json_file}: {e}")
                                        continue

                        except ValueError:
                            # Skip folders that aren't numeric label fractions
                            continue

    return pd.DataFrame(results)


def load_transfer_learning_features_results(base_path, dataset_name):
    """
    Load feature-based baseline results for transfer learning datasets.
    Expected structure: results/[dataset]_features/[window_size]/[window_shift]/[model_type]/[seed]/[label_fraction]/test_results.json
    
    Args:
        base_path: Base results path
        dataset_name: 'WESAD' or 'StressID'
    """
    results = []
    base_path = Path(base_path)
    
    # Path to feature results for this dataset
    features_base = base_path / f"{dataset_name}_features"
    
    if not features_base.exists():
        print(f"Warning: Features path {features_base} does not exist")
        return pd.DataFrame(results)
    
    # Look for window size folders (e.g., 30)
    for window_size_folder in features_base.iterdir():
        if not window_size_folder.is_dir() or not window_size_folder.name.isdigit():
            continue
            
        window_size = int(window_size_folder.name)
        
        # Look for window shift folders (e.g., 10)
        for window_shift_folder in window_size_folder.iterdir():
            if not window_shift_folder.is_dir() or not window_shift_folder.name.isdigit():
                continue

            # Here we can exclude window shifts
            window_shift = int(window_shift_folder.name)

            if window_size == 30 and window_shift in [5, 15]:
                continue

            # Look for model type folders (e.g., logistic_regression)
            for model_folder in window_shift_folder.iterdir():

                if not model_folder.is_dir():
                    continue
                    
                model_type = model_folder.name

                if model_type == "xgboost":
                    continue
                # Look for seed folders
                for seed_folder in model_folder.iterdir():
                    if not (seed_folder.is_dir() and seed_folder.name.isdigit()):
                        continue
                        
                    seed = int(seed_folder.name)
                    
                    # Look for label fraction folders
                    for label_folder in seed_folder.iterdir():
                        if not label_folder.is_dir():
                            continue
                            
                        try:
                            label_fraction = float(label_folder.name)
                            
                            # Look for test_results.json
                            json_file = label_folder / "test_results.json"
                            if json_file.exists():
                                try:
                                    with open(json_file, 'r') as f:
                                        data = json.load(f)
                                    results.append({
                                        'dataset': dataset_name,
                                        'method_type': 'Features_Baseline',
                                        'learning_method': 'Feature-engineered',
                                        'model_type': model_type,
                                        'seed': seed,
                                        'label_fraction': label_fraction,
                                        'window_size': window_size,
                                        'window_shift': window_shift,
                                        'auroc': data["test_metrics"].get('auroc', np.nan),
                                        'accuracy': data["test_metrics"].get('accuracy', np.nan),
                                        'pr_auc': data["test_metrics"].get('pr_auc', np.nan),
                                        'f1': data["test_metrics"].get('f1', np.nan),
                                    })
                                except (json.JSONDecodeError, KeyError) as e:
                                    print(f"Error reading {json_file}: {e}")
                                    continue
                                    
                        except ValueError:
                            # Skip folders that aren't numeric label fractions
                            continue
    
    return pd.DataFrame(results)


def load_transfer_learning_results(base_path, dataset_name):
    """
    Load transfer learning results for a specific dataset (WESAD or StressID).
    Expected structure: results/Transfer_learning/[dataset]/[pretrained_encoder|trained_from_scratch]/[method]/[model]/[seed]/[label_fraction]/[window_size]/[window_shift]/test_results.json
    
    Args:
        base_path: Base results path
        dataset_name: 'WESAD' or 'StressID'
    """
    results = []
    base_path = Path(base_path)
    
    # Path to transfer learning results for this dataset
    transfer_base = base_path / "Transfer_learning" / dataset_name
    
    if not transfer_base.exists():
        print(f"Warning: Transfer learning path {transfer_base} does not exist")
        return pd.DataFrame(results)
    
    # Define the transfer types "trained_from_scratch" excluded
    transfer_types = ["pretrained_encoder", "pretrained_encoder_fine_tuned_encoder", "cnn"]

    for transfer_type in transfer_types:
        transfer_path = transfer_base / transfer_type
        
        if not transfer_path.exists():
            print(f"Warning: Transfer type path {transfer_path} does not exist")
            continue
        
        if transfer_type == "cnn":
            # Handle CNN results directly (different structure)
            # Look for seed folders directly under cnn/
            for seed_folder in transfer_path.iterdir():
                if not (seed_folder.is_dir() and seed_folder.name.isdigit()):
                    continue
                    
                seed = int(seed_folder.name)
                
                # Look for label fraction folders
                for label_folder in seed_folder.iterdir():
                    if not label_folder.is_dir():
                        continue
                        
                    try:
                        label_fraction = float(label_folder.name)
                        
                        # Look for window_size/window_shift folders
                        window_combinations = [
                            (10, 5),   # 10s windows, 5s shift
                            (30, 5),   # 30s windows, 5s shift
                            # (30, 10),  # 30s windows, 10s shift
                        ]
                        
                        for window_size, window_shift in window_combinations:

                            json_file = label_folder / str(window_size) / str(
                                window_shift) / "test_results.json"

                            if json_file.exists():
                                try:
                                    with open(json_file, 'r') as f:
                                        data = json.load(f)
                                    results.append({
                                        'dataset': dataset_name,
                                        'transfer_type': transfer_type,
                                        'method_type': 'Transfer_Learning',
                                        'learning_method': 'Supervised',
                                        'model_type': 'cnn',
                                        'seed': seed,
                                        'label_fraction': label_fraction,
                                        'window_size': window_size,
                                        'window_shift': window_shift,
                                        'auroc': data["test_metrics"].get('auroc', np.nan),
                                        'accuracy': data["test_metrics"].get('accuracy', np.nan),
                                        'pr_auc': data["test_metrics"].get('pr_auc', np.nan),
                                        'f1': data["test_metrics"].get('f1', np.nan),
                                    })
                                except (json.JSONDecodeError, KeyError) as e:
                                    print(f"Error reading {json_file}: {e}")
                                    continue
                                    
                    except ValueError:
                        # Skip folders that aren't numeric label fractions
                        continue
        else:
            # Handle TSTCC results (existing structure)
            # Look for TSTCC method folders
            tstcc_path = transfer_path / "TSTCC"
            if not tstcc_path.exists():
                continue
                
            # Look for model type folders (logistic_regression, mlp, etc.)
            for model_folder in tstcc_path.iterdir():
                if not model_folder.is_dir():
                    continue
                    
                model_type = model_folder.name
                
                # Look for seed folders
                for seed_folder in model_folder.iterdir():
                    if not (seed_folder.is_dir() and seed_folder.name.isdigit()):
                        continue
                        
                    seed = int(seed_folder.name)
                    
                    # Look for label fraction folders
                    for label_folder in seed_folder.iterdir():
                        if not label_folder.is_dir():
                            continue
                            
                        try:
                            label_fraction = float(label_folder.name)
                            
                            # Look for window_size/window_shift folders
                            window_combinations = [
                                (10, 5),   # 10s windows, 5s shift
                                # (30, 5),   # 30s windows, 5s shift
                                # (30, 10),  # 30s windows, 10s shift
                            ]
                            
                            for window_size, window_shift in window_combinations:
                                json_file = label_folder / str(window_size) / str(window_shift) / "test_results.json"
                                if json_file.exists():
                                    try:
                                        with open(json_file, 'r') as f:
                                            data = json.load(f)
                                        results.append({
                                            'dataset': dataset_name,
                                            'transfer_type': transfer_type,
                                            'method_type': 'Transfer_Learning',
                                            'learning_method': 'TSTCC',
                                            'model_type': model_type,
                                            'seed': seed,
                                            'label_fraction': label_fraction,
                                            'window_size': window_size,
                                            'window_shift': window_shift,
                                            'auroc': data["test_metrics"].get('auroc', np.nan),
                                            'accuracy': data["test_metrics"].get('accuracy', np.nan),
                                            'pr_auc': data["test_metrics"].get('pr_auc', np.nan),
                                            'f1': data["test_metrics"].get('f1', np.nan),
                                        })
                                    except (json.JSONDecodeError, KeyError) as e:
                                        print(f"Error reading {json_file}: {e}")
                                        continue
                                        
                        except ValueError:
                            # Skip folders that aren't numeric label fractions
                            continue
    
    return pd.DataFrame(results)


def load_combined_transfer_learning_results(base_path, dataset_name):
    """
    Load both transfer learning and feature baseline results for a dataset.
    
    Args:
        base_path: Base results path
        dataset_name: 'WESAD' or 'StressID'
    
    Returns:
        Combined DataFrame with both transfer learning and feature baseline results
    """
    # Load transfer learning results
    transfer_df = load_transfer_learning_results(base_path, dataset_name)
    
    # Load feature baseline results
    features_df = load_transfer_learning_features_results(base_path, dataset_name)
    
    # Combine the dataframes
    combined_df = pd.concat([transfer_df, features_df], ignore_index=True)
    
    return combined_df


def create_method_labels(df):
    """Create clean method labels for plotting with window size information"""

    def clean_model_name(model_name):
        """Clean up model names - capitalize first letter only"""
        model_map = {
            'cnn': 'CNN',
            'tcn': 'TCN', 
            'transformer': 'Transformer',
            'deep_ecg_net': 'DeepECGNet',
            'logistic_regression': 'Logistic Regression',
            'mlp': 'MLP',
            'linear': 'Linear'
        }
        return model_map.get(model_name.lower(), model_name.title())

    def make_label(row):
        clean_model = clean_model_name(row['model_type'])
        window_info = f"{row['window_size']}s"
        
        # Add window shift info for 30s windows to differentiate different shifts
        # Also add for 10s windows when window_shift is present
        if (row['window_size'] == 30 or row['window_size'] == 10) and 'window_shift' in row:
            window_info = f"{row['window_size']}s/{row['window_shift']}s"
        
        if row['method_type'] == 'ECG_features':
            return f"Feature-engineered ({clean_model}, {window_info})"
        elif row['method_type'] == 'Transfer_Learning':
            # Create transfer learning labels
            if row['transfer_type'] == 'cnn':
                return f"Supervised ({clean_model}, {window_info})"
            else:
                if row['transfer_type'] == 'pretrained_encoder':
                    transfer_label = "Pre-trained"
                elif row['transfer_type'] == 'pretrained_encoder_fine_tuned_encoder':
                    transfer_label = "Pre-trained Fine-tuned"
                else:
                    transfer_label = "From Scratch"
                return f"{transfer_label} ({clean_model}, {window_info})"
        elif row['method_type'] == 'Features_Baseline':
            return f"Feature Baseline ({clean_model}, {window_info})"
        elif  row["learning_method"] == "Supervised":
            return f"{clean_model} ({window_info})"
        else:
            return f"{row['learning_method']} ({clean_model}, {window_info})"

    df['method_label'] = df.apply(make_label, axis=1)
    return df


def plot_transfer_learning_results(df, dataset_name, metric="auroc", save_path=None,
                                   use_participant_count=False, total_participants=None, use_standard_error=False):
    """Create a plot comparing transfer learning approaches for a specific dataset
    
    Args:
        df: DataFrame with transfer learning results
        dataset_name: Name of the dataset (for title and participant count)
        metric: Metric to plot ('auroc' or 'pr_auc')
        save_path: Path to save the plot
        use_participant_count: If True, show number of participants instead of percentages
        total_participants: Total number of training participants for this dataset
    """
    
    # Set default participant counts and PR-AUC baselines if not provided
    if total_participants is None:
        if dataset_name == "WESAD":
            total_participants = 15  # Adjust based on actual WESAD participant count
        elif dataset_name == "StressID":
            total_participants = 35  # Adjust based on actual StressID participant count
        else:
            total_participants = 101  # Default fallback
    
    # Set dataset-specific PR-AUC baseline (random chance for each dataset)
    if dataset_name == "WESAD":
        pr_auc_baseline = 0.3625
    elif dataset_name == "StressID":
        pr_auc_baseline = 0.3510
    else:
        pr_auc_baseline = 0.5736  # Default fallback
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7)

    # Group by method and calculate mean and std
    grouped = df.groupby(['method_label', 'label_fraction'])[metric].agg(['mean', 'std', 'count']).reset_index()
    
    # Calculate standard error if requested
    if use_standard_error:
        grouped['error'] = grouped['std'] / np.sqrt(grouped['count'])
    else:
        grouped['error'] = grouped['std']
    
    # Calculate number of labeled participants
    def calculate_labeled_participants(label_fraction, total_participants):
        if total_participants <= 20:
            return max(3, int(total_participants * label_fraction))
        return max(5, int(total_participants * label_fraction))
    
    grouped['n_labeled_participants'] = grouped['label_fraction'].apply(calculate_labeled_participants,
                                                                        total_participants=total_participants)

    # Define colors and markers for transfer learning methods - aligned with plot_metric_vs_label_fraction
    method_styles = {
        # Pre-trained encoder methods - TSTCC methods use light blue like main plot
        'Pre-trained (Logistic Regression, 10s/5s)': {'color': '#ffa600', 'marker': 'v', 'linestyle': '-'},
        'Pre-trained (MLP, 10s/5s)': {'color': '#88CCEE', 'marker': 'p', 'linestyle': '-'},
        'Pre-trained (Logistic Regression, 30s/5s)': {'color': '#88CCEE', 'marker': 'o', 'linestyle': '-'},
        'Pre-trained (Logistic Regression, 30s/10s)': {'color': '#88CCEE', 'marker': 's', 'linestyle': '-'},
        'Pre-trained (Logistic Regression, 30s/15s)': {'color': '#88CCEE', 'marker': '^', 'linestyle': ':'},
        'Pre-trained (MLP, 30s/5s)': {'color': '#88CCEE', 'marker': '8', 'linestyle': '-'},
        'Pre-trained (MLP, 30s/10s)': {'color': '#88CCEE', 'marker': 'D', 'linestyle': '--'},
        'Pre-trained (MLP, 30s/15s)': {'color': '#88CCEE', 'marker': 'h', 'linestyle': ':'},

        # From scratch methods - use different colors to distinguish from pre-trained
        'From Scratch (Logistic Regression, 10s/5s)': {'color': '#CC79A7', 'marker': 'v', 'linestyle': '-'},
        'From Scratch (MLP, 10s/5s)': {'color': '#CC79A7', 'marker': 'p', 'linestyle': '-'},
        'From Scratch (Logistic Regression, 30s/5s)': {'color': '#CC79A7', 'marker': 'o', 'linestyle': '-'},
        'From Scratch (Logistic Regression, 30s/10s)': {'color': '#CC79A7', 'marker': 's', 'linestyle': '-'},
        'From Scratch (Logistic Regression, 30s/15s)': {'color': '#CC79A7', 'marker': '^', 'linestyle': ':'},
        'From Scratch (MLP, 30s/5s)': {'color': '#CC79A7', 'marker': '8', 'linestyle': '-'},
        'From Scratch (MLP, 30s/10s)': {'color': '#CC79A7', 'marker': 'D', 'linestyle': '--'},
        'From Scratch (MLP, 30s/15s)': {'color': '#CC79A7', 'marker': 'h', 'linestyle': ':'},
        
        # Feature baseline methods - use orange colors like main plot feature-engineered methods
        'Feature Baseline (Logistic Regression, 30s/10s)': {'color': '#bc5090', 'marker': 'o', 'linestyle': '-',
                                                              'linewidth': 2., 'alpha': 0.2},
        'Feature Baseline (Logistic Regression, 10s/5s)': {'color': '#ef5675', 'marker': 'x', 'linestyle': '-',
                                                             'linewidth': 2., 'alpha': 0.2},


        'Feature Baseline (Logistic Regression, 30s/5s)': {'color': '#ff6361', 'marker': 'v', 'linestyle': '-'},
        # 'Feature Baseline (Logistic Regression, 30s/10s)': {'color': '#E69F00', 'marker': 'o', 'linestyle': '-'},
        'Feature Baseline (Logistic Regression, 30s/15s)': {'color': '#665191', 'marker': 's', 'linestyle': ':'},
        # 'Feature Baseline (Logistic Regression, 10s/5s)': {'color': '#F0746E', 'marker': 'x', 'linestyle': '-'},
        'Feature Baseline (MLP, 30s/5s)': {'color': '#E69F00', 'marker': '8', 'linestyle': '-'},
        'Feature Baseline (MLP, 30s/10s)': {'color': '#E69F00', 'marker': 'D', 'linestyle': '--'},
        'Feature Baseline (MLP, 30s/15s)': {'color': '#E69F00', 'marker': 'h', 'linestyle': ':'},

        # # CNN Supervised methods - use same colors as main plot supervised methods
        # 'Supervised (CNN, 10s/5s)': {'color': '#226E9C', 'marker': '^', 'linestyle': '-'},
        'Supervised (CNN, 10s/5s)': {'color': '#226E9C', 'marker': '^', 'linestyle': '-'},
        'Supervised (CNN, 30s/5s)': {'color': '#D55E00', 'marker': 'v', 'linestyle': '-'},
        'Supervised (CNN, 30s/10s)': {'color': '#D55E00', 'marker': 'o', 'linestyle': '--'},
        'Supervised (CNN, 30s/15s)': {'color': '#D55E00', 'marker': 's', 'linestyle': ':'},

        # Pre-trained Fine-tuned encoder methods - use darker blue to distinguish from regular pre-trained
        'Pre-trained Fine-tuned (Logistic Regression, 10s/5s)': {'color': '#DDA853', 'marker': 'v', 'linestyle': '--'},
        'Pre-trained Fine-tuned (MLP, 10s/5s)': {'color': '#4477AA', 'marker': 'p', 'linestyle': '--'},
        'Pre-trained Fine-tuned (Logistic Regression, 30s/5s)': {'color': '#4477AA', 'marker': 'o', 'linestyle': '--'},
        'Pre-trained Fine-tuned (Logistic Regression, 30s/10s)': {'color': '#4477AA', 'marker': 's', 'linestyle': '--'},
        'Pre-trained Fine-tuned (Logistic Regression, 30s/15s)': {'color': '#4477AA', 'marker': '^', 'linestyle': ':'},
        'Pre-trained Fine-tuned (MLP, 30s/5s)': {'color': '#4477AA', 'marker': '8', 'linestyle': '--'},
        'Pre-trained Fine-tuned (MLP, 30s/10s)': {'color': '#4477AA', 'marker': 'D', 'linestyle': ':'},
        'Pre-trained Fine-tuned (MLP, 30s/15s)': {'color': '#4477AA', 'marker': 'h', 'linestyle': ':'},
    }

    # Plot each method
    for method in grouped['method_label'].unique():
        method_data = grouped[grouped['method_label'] == method].sort_values('label_fraction')
        style = method_styles.get(method, {'color': 'black', 'marker': 'o', 'linestyle': '-'})

        # Choose x-axis values based on use_participant_count parameter
        if use_participant_count:
            x_vals = method_data['n_labeled_participants']
        else:
            x_vals = method_data['label_fraction'] * 100
        y_vals = method_data['mean']

        # Plot main line
        linewidth = style.get('linewidth', 2.0)
        ax.plot(x_vals, y_vals,
                color=style['color'],
                marker=style['marker'],
                linestyle=style['linestyle'],
                linewidth=linewidth,
                markersize=8,
                label=method,
                markerfacecolor='white',
                markeredgewidth=2,
                markeredgecolor=style['color'])

        # Add error visualization with fill_between if we have multiple seeds
        if method_data['count'].max() > 1:
            error_vals = method_data['error'].fillna(0)
            
            # Use fill_between for better uncertainty visualization
            ax.fill_between(x_vals, 
                          y_vals - error_vals, 
                          y_vals + error_vals,
                          color=style['color'], 
                          alpha=0.2, 
                          interpolate=True)

    # Customize the plot
    if use_participant_count:
        ax.set_xlabel('# Labeled Training Participants', fontsize=14)
        # Set x-axis scale and limits for participant count
        # ax.set_xscale('log')
        min_participants = calculate_labeled_participants(label_fraction=0., total_participants=total_participants)
        ax.set_xlim(min_participants -0.2, total_participants +0.2)
        # Customize x-axis ticks for participant counts
        if total_participants <= 20:
            x_ticks = [min_participants, int(min_participants*2), total_participants]
        else:
            # x_ticks = list(np.arange(5, total_participants+1, 1))
            x_ticks = [min_participants, 10, 25, total_participants]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([str(x) for x in x_ticks])
    else:
        ax.set_xlabel('Label Fraction (% of Training Participants Labeled)', fontsize=14)
        # Set x-axis to log scale for better visualization of small fractions
        ax.set_xscale('log')
        ax.set_xlim(0.8, 120)
        # Customize x-axis ticks for percentages
        x_ticks = [1, 5, 10, 25, 50, 100]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f'{x}%' for x in x_ticks])

    y_name = 'AUROC' if metric == "auroc" else "PR-AUC"
    ax.set_ylabel(y_name, fontsize=14)
    # ax.set_title(f'{dataset_name} Transfer Learning: {y_name} vs Label Fraction', fontsize=16, pad=20)

    # Set y-axis limits and ticks
    ax.set_ylim(0.3, 1.0)
    ax.set_yticks(np.arange(0.5, 1.05, 0.1))

    # Add grid
    ax.set_axisbelow(True)

    if metric == "auroc":
        ax.axhline(y=0.5, color='0.45', linestyle='solid', alpha=0.7, linewidth=2, label="Random Baseline")

    elif metric == "pr_auc":
        ax.axhline(y=pr_auc_baseline, color='0.45', linestyle='solid', alpha=0.7, linewidth=2, label="Random Baseline")

    # Customize legend
    error_type = "Standard Error" if use_standard_error else "Standard Deviation"
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                       frameon=True, fancybox=True, shadow=False,
                       fontsize=11, ncol=2)  # Optional: arrange legend items horizontally

    # Improve overall appearance
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Plot saved to {save_path}")

    plt.show()
    plt.close()

    return fig, ax


def plot_metric_vs_label_fraction(df, metric="auroc", save_path=None, use_participant_count=False,
                                total_participants=101, use_standard_error=False):
    """Create an excellent plot of AUROC vs Label Fraction with improved error visualization
    
    Args:
        df: DataFrame with results
        metric: Metric to plot ('auroc' or 'pr_auc')
        save_path: Path to save the plot
        use_participant_count: If True, show number of participants instead of percentages
        total_participants: Total number of training participants (default: 101)
        use_standard_error: If True, use standard error instead of standard deviation
    """

    with plt.style.context(['ieee']):

        # sns.set_palette("husl")

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.grid(axis='y', linestyle='--', color='0.35', linewidth=0.7, alpha=0.7)

        # Group by method and calculate mean and std
        grouped = df.groupby(['method_label', 'label_fraction'])[metric].agg(['mean', 'std', 'count']).reset_index()

        # Calculate standard error if requested
        if use_standard_error:
            grouped['error'] = grouped['std'] / np.sqrt(grouped['count'])
        else:
            grouped['error'] = grouped['std']

        # Calculate number of labeled participants
        def calculate_labeled_participants(label_fraction):
            return max(1, int(total_participants * label_fraction))

        grouped['n_labeled_participants'] = grouped['label_fraction'].apply(calculate_labeled_participants)

        method_styles = {
            # Feature-engineered (different window configurations) #4a2377  #56B4E9 #d31f11
            'Feature-engineered (Logistic Regression, 30s/5s)': {'color': '#ff6361', 'marker': 'v', 'linestyle': '--'},
            'Feature-engineered (Logistic Regression, 30s/10s)': {'color': "#d31f11", 'marker': 'o', 'linestyle': '--',
                                                                  'linewidth': 1.5, 'alpha': 0.15},
            # D4A842 #D4A842  #E69F00
            'Feature-engineered (Logistic Regression, 10s/5s)': {'color': "#E69F00", 'marker': 'x', 'linestyle': '--',
                                                                 'linewidth': 1.5, 'alpha': 0.15},

            'Feature-engineered (MLP, 30s/5s)': {'color': '#E69F00', 'marker': '8', 'linestyle': '-'},
            'Feature-engineered (MLP, 30s/10s)': {'color': '#E69F00', 'marker': 'D', 'linestyle': '--'},
            'Feature-engineered (MLP, 30s/15s)': {'color': '#E69F00', 'marker': 'h', 'linestyle': ':'},

            # Supervised methods (10s) #0d7d87
            'CNN (10s/5s)': {'color': "#7FAB79", 'marker': '^', 'linestyle': 'dotted', 'linewidth': 1.5,
                                         'alpha': 0.15},
            'TCN (10s/5s)': {'color': "#D55E00", 'marker': '>', 'linestyle': '-', 'linewidth': 1.5,
                                         'alpha': 0.1},
            'Transformer (10s/5s)': {'color': '#845ec2', 'marker': '<', 'linestyle': '-', 'linewidth': 1.5,
                                                 'alpha': 0.1},
            'Supervised (DeepECGNet, 10s/5s)': {'color': '#117733', 'marker': 'o', 'linestyle': '-'},

            # TSTCC (10s and 30s) - ALL LIGHT BLUE '#88CCEE' #56B4E9 #53AED9" #1C9FEB
            # 0072B2
            'TSTCC (Logistic Regression, 10s/5s)': {'color': "#0E9FEB", 'marker': 'v', 'linestyle': '-', 'alpha': 0.15},
            'TSTCC (Logistic Regression, 30s/5s)': {'color': '#88CCEE', 'marker': 'o', 'linestyle': '-'},
            'TSTCC (Logistic Regression, 30s/10s)': {'color': '#88CCEE', 'marker': 's', 'linestyle': '--'},
            'TSTCC (Logistic Regression, 30s/15s)': {'color': '#88CCEE', 'marker': '^', 'linestyle': ':'},
            'TSTCC (MLP, 10s/5s)': {'color': '#88CCEE', 'marker': 'p', 'linestyle': '-'},
            'TSTCC (MLP, 30s/5s)': {'color': '#88CCEE', 'marker': '8', 'linestyle': '-'},
            'TSTCC (MLP, 30s/10s)': {'color': '#88CCEE', 'marker': 'D', 'linestyle': '--'},
            'TSTCC (MLP, 30s/15s)': {'color': '#88CCEE', 'marker': 'h', 'linestyle': ':'},
            'TSTCC (Linear, 10s/5s)': {'color': '#88CCEE', 'marker': '*', 'linestyle': '-'},
            'TSTCC (Linear, 30s/5s)': {'color': '#88CCEE', 'marker': '1', 'linestyle': '-'},
            'TSTCC (Linear, 30s/10s)': {'color': '#88CCEE', 'marker': '+', 'linestyle': '--'},
            'TSTCC (Linear, 30s/15s)': {'color': '#88CCEE', 'marker': 'x', 'linestyle': ':'},

            # #009E73 #f47a00    4a2377 #0B7395   #004886
            'TSTCC+S3 (Logistic Regression, 10s/5s)': {'color': "#005377", 'marker': 's', 'linestyle': '-', 'alpha': 0.15},

            # Supervised methods (30s) - in case they exist
            'Supervised (CNN, 30s/5s)': {'color': '#D55E00', 'marker': 'v', 'linestyle': '-', 'linewidth': 1.5, 'alpha': 0.15},
            'Supervised (CNN, 30s/10s)': {'color': '#D55E00', 'marker': 'o', 'linestyle': '--'},
            'Supervised (CNN, 30s/15s)': {'color': '#D55E00', 'marker': 's', 'linestyle': ':'},
            'Supervised (TCN, 30s/5s)': {'color': '#44AA99', 'marker': 'p', 'linestyle': '-'},
            'Supervised (TCN, 30s/10s)': {'color': '#44AA99', 'marker': 's', 'linestyle': '--'},
            'Supervised (TCN, 30s/15s)': {'color': '#44AA99', 'marker': 'D', 'linestyle': ':'},
            'Supervised (Transformer, 30s/5s)': {'color': '#58508d', 'marker': '8', 'linestyle': '-'},
            'Supervised (Transformer, 30s/10s)': {'color': '#58508d', 'marker': 'D', 'linestyle': '--'},
            'Supervised (Transformer, 30s/15s)': {'color': '#58508d', 'marker': 'h', 'linestyle': ':'},
            'Supervised (DeepECGNet, 30s/5s)': {'color': '#117733', 'marker': '1', 'linestyle': '-'},
            'Supervised (DeepECGNet, 30s/10s)': {'color': '#117733', 'marker': '>', 'linestyle': '--'},
            'Supervised (DeepECGNet, 30s/15s)': {'color': '#117733', 'marker': '<', 'linestyle': ':'},
        }

        #58508d
        # Old #DDCC77
        # 003f5c
        # Plot each method
        for method in grouped['method_label'].unique():
            method_data = grouped[grouped['method_label'] == method].sort_values('label_fraction')
            if "S3" in method:
                method.replace("_S3","+S3")

            style = method_styles.get(method, {'color': 'black', 'marker': 'o', 'linestyle': '-'})

            # Choose x-axis values based on use_participant_count parameter
            if use_participant_count:
                x_vals = method_data['n_labeled_participants']
            else:
                x_vals = method_data['label_fraction'] * 100
            y_vals = method_data['mean']

            # Plot main line
            linewidth = style.get('linewidth', 1.5)
            ax.plot(x_vals, y_vals,
                    color=style['color'],
                    marker=style['marker'],
                    linestyle=style['linestyle'],
                    linewidth=linewidth,
                    markersize=8,
                    label=method,
                    markerfacecolor='white',
                    markeredgewidth=2,
                    markeredgecolor=style['color'])

            # Add error visualization with fill_between if we have multiple seeds
            if method_data['count'].max() > 1:
                error_vals = method_data['error'].fillna(0)

                # Use fill_between for better uncertainty visualization
                ax.fill_between(x_vals,
                              y_vals - error_vals,
                              y_vals + error_vals,
                              color=style['color'],
                              alpha=style.get('alpha', 0.1),
                              interpolate=True)

        # Customize the plot
        if use_participant_count:
            ax.set_xlabel('# Labeled Training Participants', fontsize=20)
            # Set x-axis scale and limits for participant count
            ax.set_xscale('log')
            ax.set_xlim(0.8, 110)
            # Customize x-axis ticks for participant counts
            x_ticks = [1, 2, 5, 10, 25, 50, 101]
            ax.set_xticks(x_ticks)
            ax.set_xticklabels([str(x) for x in x_ticks])
        else:
            ax.set_xlabel('Label Fraction (% of Training Participants Labeled)', fontsize=20)
            # Set x-axis to log scale for better visualization of small fractions
            ax.set_xscale('log')
            ax.set_xlim(0.8, 120)
            # Customize x-axis ticks for percentages
            x_ticks = [1, 2.5, 5, 10, 25, 50, 100]
            ax.set_xticks(x_ticks)
            ax.set_xticklabels([f'{x}%' for x in x_ticks])

        y_name = 'AUROC' if metric == "auroc" else "AUPRC"
        ax.set_ylabel(y_name, fontsize=20)
        # ax.set_title('ECG Classification Performance vs Label Fraction', fontsize=16, fontweight='bold', pad=20)

        # Set y-axis limits and ticks
        ax.set_ylim(0.45, 1.0) if y_name == "AUROC" else ax.set_ylim(0.5, 1.0)
        ax.set_yticks(np.arange(0.5, 1.05, 0.1))
        # ax.axvline(50, linestyle='--', linewidth=1.0, color='0.45', zorder=1)
        # Add grid
        # ax.grid(False)
        ax.set_axisbelow(True)

        # Add horizontal line at AUROC = 0.5 (random chance)
        # if metric == "auroc":
        #     ax.axhline(y=0.5, color='0.35', linestyle='--', alpha=0.8, linewidth=1.5, label="Random Baseline")
        #
        # elif metric == "pr_auc":
        #     ax.axhline(y=0.5736, color='0.35', linestyle='--', alpha=0.8, linewidth=1.5, label="Random Baseline")

        # Customize legend
        # error_type = "Standard Error" if use_standard_error else "Standard Deviation"
        ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=False,
                           fontsize=20)

        # Improve overall appearance
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=500, bbox_inches='tight', facecolor='white')
            print(f"Plot saved to {save_path}")

        plt.show()
        plt.close()

    return fig, ax


def print_summary_statistics(df):
    """Print summary statistics of the results"""
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    print(f"\nTotal experiments loaded: {len(df)}")
    print(f"Methods: {df['method_label'].nunique()}")
    print(f"Label fractions: {sorted(df['label_fraction'].unique())}")
    print(f"Seeds: {sorted(df['seed'].unique())}")

    print("\n" + "-" * 40)
    print("AUROC PERFORMANCE BY METHOD")
    print("-" * 40)

    # Calculate average performance across all label fractions for each method
    method_avg = df.groupby('method_label')['auroc'].agg(['mean', 'std', 'min', 'max']).round(4)
    print(method_avg)

    print("\n" + "-" * 40)
    print("BEST PERFORMANCE AT EACH LABEL FRACTION")
    print("-" * 40)

    for frac in sorted(df['label_fraction'].unique()):
        frac_data = df[df['label_fraction'] == frac]
        best_idx = frac_data['auroc'].idxmax()
        best_method = frac_data.loc[best_idx, 'method_label']
        best_auroc = frac_data.loc[best_idx, 'auroc']
        print(f"{frac * 100:5.1f}%: {best_method:<25} (AUROC: {best_auroc:.4f})")


def load_feature_vs_ssl_comparison_results(base_path):
    """Load results for comparing feature-engineered vs SSL methods.
    
    This function loads results from:
    - ECG_features/logistic_regression, random_forest, xgboost
    - ECG/TSTCC/logistic_regression, xgboost (trained from scratch)
    
    Args:
        base_path: Base results path
    
    Returns:
        DataFrame with combined results for comparison
    """
    results = []
    base_path = Path(base_path)
    
    # Load ECG_features results (logistic_regression, random_forest, xgboost)
    ecg_features_path = base_path / "ECG_features"
    feature_models = ["logistic_regression"]
    
    for model_type in feature_models:
        model_path = ecg_features_path / model_type
        if not model_path.exists():
            print(f"Warning: ECG_features path {model_path} does not exist")
            continue
            
        # Look for seed folders
        for seed_folder in model_path.iterdir():
            if not (seed_folder.is_dir() and seed_folder.name.isdigit()):
                continue
                
            seed = int(seed_folder.name)
            
            # Look for label fraction folders
            for label_folder in seed_folder.iterdir():
                if not label_folder.is_dir():
                    continue
                    
                try:
                    label_fraction = float(label_folder.name)
                    
                    # Look for window size/shift combinations
                    window_combinations = [
                        (10, 5),   # 10s windows, 5s shift
                        (30, 5),   # 30s windows, 5s shift
                        (30, 10),  # 30s windows, 10s shift
                        (30, 15),  # 30s windows, 15s shift
                    ]
                    
                    for window_size, window_shift in window_combinations:
                        json_file = label_folder / str(window_size) / str(window_shift) / "test_results.json"
                        if json_file.exists():
                            try:
                                with open(json_file, 'r') as f:
                                    data = json.load(f)
                                results.append({
                                    'method_type': 'Feature_Engineered',
                                    'learning_method': 'Feature-engineered',
                                    'model_type': model_type,
                                    'seed': seed,
                                    'label_fraction': label_fraction,
                                    'window_size': window_size,
                                    'window_shift': window_shift,
                                    'auroc': data["test_metrics"].get('auroc', np.nan),
                                    'accuracy': data["test_metrics"].get('accuracy', np.nan),
                                    'pr_auc': data["test_metrics"].get('pr_auc', np.nan),
                                    'f1': data["test_metrics"].get('f1', np.nan),
                                })
                            except (json.JSONDecodeError, KeyError) as e:
                                print(f"Error reading {json_file}: {e}")
                                continue
                                
                except ValueError:
                    continue
    
    # Load ECG/TSTCC results (logistic_regression, xgboost trained from scratch)
    ssl_methods = ["TSTCC", "TSTCC_S3"]

    for ssl_method in ssl_methods:

        tstcc_path = base_path / "ECG" / ssl_method
        head_models = ["logistic_regression"]

        for model_type in head_models:
            model_path = tstcc_path / model_type
            if not model_path.exists():
                print(f"Warning: TSTCC path {model_path} does not exist")
                continue

            # Look for seed folders
            for seed_folder in model_path.iterdir():
                if not (seed_folder.is_dir() and seed_folder.name.isdigit()):
                    continue

                seed = int(seed_folder.name)

                # Look for label fraction folders
                for label_folder in seed_folder.iterdir():
                    if not label_folder.is_dir():
                        continue

                    try:
                        label_fraction = float(label_folder.name)

                        # Look for window size/shift combinations - only 10s/5s for TSTCC
                        window_combinations = [
                            (10, 5),   # 10s windows, 5s shift only
                        ]

                        for window_size, window_shift in window_combinations:
                            json_file = label_folder / str(window_size) / str(window_shift) / "1.0"/"test_results.json"

                            # Special handling for xgboost with 0.75 subfolder
                            if model_type == "xgboost":
                                json_file = label_folder / str(window_size) / str(window_shift) / "0.75" / "test_results.json"

                            if json_file.exists():
                                try:
                                    with open(json_file, 'r') as f:
                                        data = json.load(f)
                                    results.append({
                                        'method_type': ssl_method,
                                        'learning_method': ssl_method,
                                        'model_type': model_type,
                                        'seed': seed,
                                        'label_fraction': label_fraction,
                                        'window_size': window_size,
                                        'window_shift': window_shift,
                                        'auroc': data["test_metrics"].get('auroc', np.nan),
                                        'accuracy': data["test_metrics"].get('accuracy', np.nan),
                                        'pr_auc': data["test_metrics"].get('pr_auc', np.nan),
                                        'f1': data["test_metrics"].get('f1', np.nan),
                                    })
                                except (json.JSONDecodeError, KeyError) as e:
                                    print(f"Error reading {json_file}: {e}")
                                    continue

                    except ValueError:
                        continue

    return pd.DataFrame(results)


def plot_feature_vs_ssl_comparison(df, metric="auroc", save_path=None, use_participant_count=False, total_participants=101):
    """Create a comparison plot between feature-engineered and SSL methods.
    
    Args:
        df: DataFrame with comparison results
        metric: Metric to plot ('auroc' or 'pr_auc')
        save_path: Path to save the plot
        use_participant_count: If True, show number of participants instead of percentages
        total_participants: Total number of training participants (default: 101)
    """

    with plt.style.context(['ieee']):

        # sns.set_palette("husl")

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.grid(axis='y', linestyle='--', color='0.35', linewidth=0.7, alpha=0.7)

        # Group by method and calculate mean, std, and stderr
        grouped = df.groupby(['method_label', 'label_fraction'])[metric].agg(['mean', 'std', 'count']).reset_index()
        grouped['stderr'] = grouped['std'] / np.sqrt(grouped['count'])

        # Calculate number of labeled participants
        def calculate_labeled_participants(label_fraction):
            return max(1, int(total_participants * label_fraction))

        grouped['n_labeled_participants'] = grouped['label_fraction'].apply(calculate_labeled_participants)

        # Define colors and markers for comparison methods
        method_styles = {
            # Feature-engineered (different window configurations) #4a2377  #56B4E9 #d31f11
            'Feature-engineered (Logistic Regression, 30s/5s)': {'color': '#ff6361', 'marker': 'v', 'linestyle': '--'},
            'Feature-engineered (Logistic Regression, 30s/10s)': {'color': "#d31f11", 'marker': 'o', 'linestyle': '--',
                                                                'alpha': 0.15},
            'Feature-engineered (Logistic Regression, 30s/15s)': {'color': '#EB7317', 'marker': 'd', 'linestyle': '--',
                                                                  'alpha': 0.15},


            # D4A842 #D4A842  #E69F00
            'Feature-engineered (Logistic Regression, 10s/5s)': {'color': "#E69F00", 'marker': 'x', 'linestyle': '--',
                                                                  'alpha': 0.15},

            'Feature-engineered (MLP, 30s/5s)': {'color': '#E69F00', 'marker': '8', 'linestyle': '-'},
            'Feature-engineered (MLP, 30s/10s)': {'color': '#E69F00', 'marker': 'D', 'linestyle': '--'},
            'Feature-engineered (MLP, 30s/15s)': {'color': '#E69F00', 'marker': 'h', 'linestyle': ':'},

            # Supervised methods (10s) #0d7d87
            'CNN (10s/5s)': {'color': "#7FAB79", 'marker': '^', 'linestyle': 'dotted', 'linewidth': 1.5,
                             'alpha': 0.15},
            'TCN (10s/5s)': {'color': "#D55E00", 'marker': '>', 'linestyle': '-', 'linewidth': 1.5,
                             'alpha': 0.1},
            'Transformer (10s/5s)': {'color': '#845ec2', 'marker': '<', 'linestyle': '-', 'linewidth': 1.5,
                                     'alpha': 0.1},
            'Supervised (DeepECGNet, 10s/5s)': {'color': '#117733', 'marker': 'o', 'linestyle': '-'},

            # TSTCC (10s and 30s) - ALL LIGHT BLUE '#88CCEE' #56B4E9 #53AED9" #1C9FEB
            # 0072B2
            'TSTCC (Logistic Regression, 10s/5s)': {'color': "#0E9FEB", 'marker': 'v', 'linestyle': '-', 'alpha': 0.15},
            'TSTCC (Logistic Regression, 30s/5s)': {'color': '#88CCEE', 'marker': 'o', 'linestyle': '-'},
            'TSTCC (Logistic Regression, 30s/10s)': {'color': '#88CCEE', 'marker': 's', 'linestyle': '--'},
            'TSTCC (Logistic Regression, 30s/15s)': {'color': '#88CCEE', 'marker': '^', 'linestyle': ':'},
            'TSTCC (MLP, 10s/5s)': {'color': '#88CCEE', 'marker': 'p', 'linestyle': '-'},
            'TSTCC (MLP, 30s/5s)': {'color': '#88CCEE', 'marker': '8', 'linestyle': '-'},
            'TSTCC (MLP, 30s/10s)': {'color': '#88CCEE', 'marker': 'D', 'linestyle': '--'},
            'TSTCC (MLP, 30s/15s)': {'color': '#88CCEE', 'marker': 'h', 'linestyle': ':'},
            'TSTCC (Linear, 10s/5s)': {'color': '#88CCEE', 'marker': '*', 'linestyle': '-'},
            'TSTCC (Linear, 30s/5s)': {'color': '#88CCEE', 'marker': '1', 'linestyle': '-'},
            'TSTCC (Linear, 30s/10s)': {'color': '#88CCEE', 'marker': '+', 'linestyle': '--'},
            'TSTCC (Linear, 30s/15s)': {'color': '#88CCEE', 'marker': 'x', 'linestyle': ':'},

            # #009E73 #f47a00    4a2377 #0B7395   #004886
            'TSTCC+S3 (Logistic Regression, 10s/5s)': {'color': "#005377", 'marker': 's', 'linestyle': '-',
                                                       'alpha': 0.15},

            # Supervised methods (30s) - in case they exist
            'Supervised (CNN, 30s/5s)': {'color': '#D55E00', 'marker': 'v', 'linestyle': '-', 'linewidth': 1.5,
                                         'alpha': 0.15},
            'Supervised (CNN, 30s/10s)': {'color': '#D55E00', 'marker': 'o', 'linestyle': '--'},
            'Supervised (CNN, 30s/15s)': {'color': '#D55E00', 'marker': 's', 'linestyle': ':'},
            'Supervised (TCN, 30s/5s)': {'color': '#44AA99', 'marker': 'p', 'linestyle': '-'},
            'Supervised (TCN, 30s/10s)': {'color': '#44AA99', 'marker': 's', 'linestyle': '--'},
            'Supervised (TCN, 30s/15s)': {'color': '#44AA99', 'marker': 'D', 'linestyle': ':'},
            'Supervised (Transformer, 30s/5s)': {'color': '#58508d', 'marker': '8', 'linestyle': '-'},
            'Supervised (Transformer, 30s/10s)': {'color': '#58508d', 'marker': 'D', 'linestyle': '--'},
            'Supervised (Transformer, 30s/15s)': {'color': '#58508d', 'marker': 'h', 'linestyle': ':'},
            'Supervised (DeepECGNet, 30s/5s)': {'color': '#117733', 'marker': '1', 'linestyle': '-'},
            'Supervised (DeepECGNet, 30s/10s)': {'color': '#117733', 'marker': '>', 'linestyle': '--'},
            'Supervised (DeepECGNet, 30s/15s)': {'color': '#117733', 'marker': '<', 'linestyle': ':'},
        }

        # Plot each method
        for method in grouped['method_label'].unique():
            method_data = grouped[grouped['method_label'] == method].sort_values('label_fraction')
            if "S3" in method:
                method = method.replace("_S3", "+S3")
            style = method_styles.get(method, {'color': 'black', 'marker': 'o', 'linestyle': '-'})

            # Choose x-axis values based on use_participant_count parameter
            if use_participant_count:
                x_vals = method_data['n_labeled_participants']
            else:
                x_vals = method_data['label_fraction'] * 100
            y_vals = method_data['mean']

            # Plot main line
            linewidth = style.get('linewidth', 2.0)
            ax.plot(x_vals, y_vals,
                    color=style['color'],
                    marker=style['marker'],
                    linestyle=style['linestyle'],
                    linewidth=linewidth,
                    markersize=8,
                    label=method,
                    markerfacecolor='white',
                    markeredgewidth=2,
                    markeredgecolor=style['color'])

            # Add error visualization with fill_between if we have multiple seeds
            if method_data['count'].max() > 1:
                yerr = method_data['stderr'].fillna(0)  # Use standard error instead of std
                
                # Use fill_between for better uncertainty visualization
                ax.fill_between(x_vals, 
                              y_vals - yerr, 
                              y_vals + yerr,
                              color=style['color'], 
                              alpha=style.get('alpha', 0.2), 
                              interpolate=True)

        # Customize the plot
        if use_participant_count:
            ax.set_xlabel('# Labeled Training Participants', fontsize=20)
            # Set x-axis scale and limits for participant count
            ax.set_xscale('log')
            ax.set_xlim(0.8, 110)
            # Customize x-axis ticks for participant counts
            x_ticks = [1, 2, 5, 10, 25, 50, 101]
            ax.set_xticks(x_ticks)
            ax.set_xticklabels([str(x) for x in x_ticks])
        else:
            ax.set_xlabel('Label Fraction (% of Training Participants Labeled)', fontsize=20)
            # Set x-axis to log scale for better visualization of small fractions
            ax.set_xscale('log')
            ax.set_xlim(0.8, 120)
            # Customize x-axis ticks for percentages
            x_ticks = [1, 2.5, 5, 10, 25, 50, 100]
            ax.set_xticks(x_ticks)
            ax.set_xticklabels([f'{x}%' for x in x_ticks])

        y_name = 'AUROC' if metric == "auroc" else "AUPRC"
        ax.set_ylabel(y_name, fontsize=20)

        # Set y-axis limits and ticks
        ax.set_ylim(0.45, 1.0) if y_name == "AUROC" else ax.set_ylim(0.55, 1.0)
        ax.set_yticks(np.arange(0.5, 1.05, 0.1)) if y_name == "AUROC" else ax.set_yticks(np.arange(0.55, 1.05, 0.1))

        # Add grid
        # ax.grid(False)
        ax.set_axisbelow(True)

        # Add horizontal line at random baseline
        # if metric == "auroc":
        #     ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.7, linewidth=2, label="Random Baseline")
        # elif metric == "pr_auc":
        #     ax.axhline(y=0.5736, color='black', linestyle='--', alpha=0.7, linewidth=2, label="Random Baseline")

        # Customize legend
        ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=False,
                           fontsize=20,)

        # Improve overall appearance
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=500, bbox_inches='tight', facecolor='white')
            print(f"Plot saved to {save_path}")

        plt.show()
        plt.close()

    return fig, ax


def create_feature_vs_ssl_method_labels(df):
    """Create clean method labels for feature vs SSL comparison plotting"""

    def clean_model_name(model_name):
        """Clean up model names - capitalize appropriately"""
        model_map = {
            'logistic_regression': 'Logistic Regression',
            'random_forest': 'Random Forest',
            'xgboost': 'XGBoost'
        }
        return model_map.get(model_name.lower(), model_name.title())

    def make_label(row):
        clean_model = clean_model_name(row['model_type'])
        window_info = f"{row['window_size']}s/{row['window_shift']}s"
        
        if row['method_type'] == 'Feature_Engineered':
            return f"Feature-engineered ({clean_model}, {window_info})"
        elif row['method_type'] == 'SSL_TSTCC':
            return f"SSL TSTCC ({clean_model}, {window_info})"
        else:
            return f"{row['learning_method']} ({clean_model}, {window_info})"

    df['method_label'] = df.apply(make_label, axis=1)
    return df


def load_ssl_comparison_results(base_path, ssl_methods=None, include_features=False):
    """Load results for comparing different SSL methods.
    
    Args:
        base_path: Base results path
        ssl_methods: List of SSL methods to compare (e.g., ['TSTCC', 'TSTCC_S3', 'SimCLR', 'SimCLR_S3'])
        include_features: If True, also include feature-engineered baseline results
        
    Returns:
        DataFrame with SSL comparison results
    """
    if ssl_methods is None:
        ssl_methods = ['TSTCC', 'TSTCC_S3', 'SimCLR', 'SimCLR_S3']
    
    results = []
    base_path = Path(base_path)
    
    # Load SSL method results
    for ssl_method in ssl_methods:
        method_path = base_path / "ECG" / ssl_method
        
        if not method_path.exists():
            print(f"Warning: SSL method path {method_path} does not exist")
            continue
            
        # Look for model type folders (logistic_regression, linear, but not xgboost)
        for model_folder in method_path.iterdir():
            if not model_folder.is_dir():
                continue
                
            model_type = model_folder.name
            
            # Skip MLP and XGBoost for TSTCC methods as requested
            if ssl_method in ['TSTCC', 'TSTCC_S3'] and model_type in ['mlp', 'xgboost']:
                continue
            
            # Skip XGBoost for all SSL methods
            if model_type == 'xgboost':
                continue
            
            # Look for seed folders
            for seed_folder in model_folder.iterdir():
                if not seed_folder.is_dir():
                    continue
                    
                # Handle both numeric seeds and special seeds like "42_s3"
                if seed_folder.name.isdigit():
                    seed = int(seed_folder.name)
                else:
                    continue
                
                # Look for label fraction folders
                for label_folder in seed_folder.iterdir():
                    if not label_folder.is_dir():
                        continue
                        
                    try:
                        label_fraction = float(label_folder.name)
                        
                        # Look for window size/shift combinations
                        window_combinations = [
                            (10, 5),   # 10s windows, 5s shift
                        ]
                        
                        for window_size, window_shift in window_combinations:
                            # Use consistent path structure: always use nested 1.0 folder for SSL methods
                            if model_type in ["logistic_regression", "linear"]:
                                json_file = label_folder / str(window_size) / str(window_shift) / "1.0" / "test_results.json"
                            elif model_type == "xgboost":
                                json_file = label_folder / str(window_size) / str(window_shift) / "0.75" / "test_results.json"
                            else:
                                json_file = label_folder / str(window_size) / str(window_shift) / "test_results.json"
                            
                            if json_file.exists():
                                try:
                                    with open(json_file, 'r') as f:
                                        data = json.load(f)
                                    results.append({
                                        'method_type': 'SSL',
                                        'learning_method': ssl_method,
                                        'model_type': model_type,
                                        'seed': seed,
                                        'label_fraction': label_fraction,
                                        'window_size': window_size,
                                        'window_shift': window_shift,
                                        'auroc': data["test_metrics"].get('auroc', np.nan),
                                        'accuracy': data["test_metrics"].get('accuracy', np.nan),
                                        'pr_auc': data["test_metrics"].get('pr_auc', np.nan),
                                        'f1': data["test_metrics"].get('f1', np.nan),
                                    })
                                except (json.JSONDecodeError, KeyError) as e:
                                    print(f"Error reading {json_file}: {e}")
                                    continue
                                    
                    except ValueError:
                        # Skip folders that aren't numeric label fractions
                        continue
    
    # Load feature-engineered baseline results if requested
    if include_features:
        ecg_features_path = base_path / "ECG_features" / "logistic_regression"
        
        if ecg_features_path.exists():
            # Look for seed folders
            for seed_folder in ecg_features_path.iterdir():
                if not (seed_folder.is_dir() and seed_folder.name.isdigit()):
                    continue
                    
                seed = int(seed_folder.name)
                
                # Look for label fraction folders
                for label_folder in seed_folder.iterdir():
                    if not label_folder.is_dir():
                        continue
                        
                    try:
                        label_fraction = float(label_folder.name)
                        
                        # Look for window size/shift combinations
                        window_combinations = [
                            (10, 5),   # 10s windows, 5s shift
                            (30, 10),  # 30s windows, 10s shift
                            (30, 15),
                            (30, 5),
                        ]
                        
                        for window_size, window_shift in window_combinations:
                            json_file = label_folder / str(window_size) / str(window_shift) / "test_results.json"
                            if json_file.exists():
                                try:
                                    with open(json_file, 'r') as f:
                                        data = json.load(f)
                                    results.append({
                                        'method_type': 'Feature_Engineered',
                                        'learning_method': 'Feature-engineered',
                                        'model_type': 'logistic_regression',
                                        'seed': seed,
                                        'label_fraction': label_fraction,
                                        'window_size': window_size,
                                        'window_shift': window_shift,
                                        'auroc': data["test_metrics"].get('auroc', np.nan),
                                        'accuracy': data["test_metrics"].get('accuracy', np.nan),
                                        'pr_auc': data["test_metrics"].get('pr_auc', np.nan),
                                        'f1': data["test_metrics"].get('f1', np.nan),
                                    })
                                except (json.JSONDecodeError, KeyError) as e:
                                    print(f"Error reading {json_file}: {e}")
                                    continue
                                    
                    except ValueError:
                        # Skip folders that aren't numeric label fractions
                        continue
        else:
            print(f"Warning: Feature-engineered path {ecg_features_path} does not exist")
    
    return pd.DataFrame(results)


def create_ssl_method_labels(df):
    """Create clean method labels for SSL comparison plotting"""

    def clean_model_name(model_name):
        """Clean up model names - capitalize appropriately"""
        model_map = {
            'logistic_regression': 'Logistic Regression',
            'mlp': 'MLP',
            'linear': 'Linear'
        }
        return model_map.get(model_name.lower(), model_name.title())

    def make_label(row):
        clean_model = clean_model_name(row['model_type'])
        window_info = f"{row['window_size']}s/{row['window_shift']}s"
        
        # Create labels with SSL method name or feature-engineered
        if row['method_type'] == 'Feature_Engineered':
            return f"Feature-engineered ({clean_model}, {window_info})"
        else:
            return f"{row['learning_method']} ({clean_model}, {window_info})"

    df['method_label'] = df.apply(make_label, axis=1)
    return df


def plot_fine_tuned_vs_pretrained_comparison(
        df, dataset_name, metric="auroc", save_path=None, use_participant_count=False,
        total_participants=None, use_standard_error=False
):
    """Create a comparison plot specifically between fine-tuned vs pretrained encoders.
    
    Args:
        df: DataFrame with transfer learning results
        dataset_name: Name of the dataset (for title and participant count)
        metric: Metric to plot ('auroc' or 'pr_auc')
        save_path: Path to save the plot
        use_participant_count: If True, show number of participants instead of percentages
        total_participants: Total number of training participants for this dataset
        use_standard_error: If True, use standard error instead of standard deviation
    """
    
    # Set default participant counts if not provided
    if total_participants is None:
        if dataset_name == "WESAD":
            total_participants = 12
        elif dataset_name == "StressID":
            total_participants = 52
        else:
            total_participants = 101
    
    # Set dataset-specific PR-AUC baseline (random chance for each dataset)
    if dataset_name == "WESAD":
        pr_auc_baseline = 0.3625
    elif dataset_name == "StressID":
        pr_auc_baseline = 0.3510
    else:
        pr_auc_baseline = 0.5736
    
    # Filter to only include pretrained_encoder and pretrained_encoder_fine_tuned_encoder
    comparison_df = df[df['transfer_type'].isin(['pretrained_encoder', 'pretrained_encoder_fine_tuned_encoder'])].copy()
    
    if comparison_df.empty:
        print(f"No fine-tuned vs pretrained comparison data found for {dataset_name}")
        return None, None
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")

    fig, ax = plt.subplots(figsize=(12, 8))

    # Group by method and calculate mean and std
    grouped = comparison_df.groupby(['method_label', 'label_fraction'])[metric].agg(['mean', 'std', 'count']).reset_index()
    
    # Calculate standard error if requested
    if use_standard_error:
        grouped['error'] = grouped['std'] / np.sqrt(grouped['count'])
    else:
        grouped['error'] = grouped['std']
    
    # Calculate number of labeled participants
    def calculate_labeled_participants(label_fraction, total_participants):
        if total_participants <= 20:
            return max(3, int(total_participants * label_fraction))
        return max(5, int(total_participants * label_fraction))
    
    grouped['n_labeled_participants'] = grouped['label_fraction'].apply(calculate_labeled_participants, total_participants=total_participants)

    # Define colors and markers for comparison - distinguish fine-tuned from pretrained
    method_styles = {
        # Pre-trained encoder methods - light blue
        'Pre-trained (Logistic Regression, 10s/5s)': {'color': '#88CCEE', 'marker': 'v', 'linestyle': '-', 'linewidth': 3},
        'Pre-trained (MLP, 10s/5s)': {'color': '#88CCEE', 'marker': 'p', 'linestyle': '-', 'linewidth': 3},
        'Pre-trained (Logistic Regression, 30s/5s)': {'color': '#88CCEE', 'marker': 'o', 'linestyle': '-', 'linewidth': 3},
        'Pre-trained (Logistic Regression, 30s/10s)': {'color': '#88CCEE', 'marker': 's', 'linestyle': '-', 'linewidth': 3},
        'Pre-trained (MLP, 30s/5s)': {'color': '#88CCEE', 'marker': '8', 'linestyle': '-', 'linewidth': 3},
        'Pre-trained (MLP, 30s/10s)': {'color': '#88CCEE', 'marker': 'D', 'linestyle': '-', 'linewidth': 3},

        # Pre-trained Fine-tuned encoder methods - darker blue with dashed lines
        'Pre-trained Fine-tuned (Logistic Regression, 10s/5s)': {'color': '#4477AA', 'marker': 'v', 'linestyle': '--', 'linewidth': 3},
        'Pre-trained Fine-tuned (MLP, 10s/5s)': {'color': '#4477AA', 'marker': 'p', 'linestyle': '--', 'linewidth': 3},
        'Pre-trained Fine-tuned (Logistic Regression, 30s/5s)': {'color': '#4477AA', 'marker': 'o', 'linestyle': '--', 'linewidth': 3},
        'Pre-trained Fine-tuned (Logistic Regression, 30s/10s)': {'color': '#4477AA', 'marker': 's', 'linestyle': '--', 'linewidth': 3},
        'Pre-trained Fine-tuned (MLP, 30s/5s)': {'color': '#4477AA', 'marker': '8', 'linestyle': '--', 'linewidth': 3},
        'Pre-trained Fine-tuned (MLP, 30s/10s)': {'color': '#4477AA', 'marker': 'D', 'linestyle': '--', 'linewidth': 3},
    }

    # Plot each method
    for method in grouped['method_label'].unique():
        method_data = grouped[grouped['method_label'] == method].sort_values('label_fraction')
        style = method_styles.get(method, {'color': 'black', 'marker': 'o', 'linestyle': '-', 'linewidth': 2.5})

        # Choose x-axis values based on use_participant_count parameter
        if use_participant_count:
            x_vals = method_data['n_labeled_participants']
        else:
            x_vals = method_data['label_fraction'] * 100
        y_vals = method_data['mean']

        # Plot main line
        linewidth = style.get('linewidth', 2.5)
        ax.plot(x_vals, y_vals,
                color=style['color'],
                marker=style['marker'],
                linestyle=style['linestyle'],
                linewidth=linewidth,
                markersize=10,
                label=method,
                markerfacecolor='white',
                markeredgewidth=2,
                markeredgecolor=style['color'])

        # Add error visualization with fill_between if we have multiple seeds
        if method_data['count'].max() > 1:
            error_vals = method_data['error'].fillna(0)
            
            # Use fill_between for better uncertainty visualization
            ax.fill_between(x_vals, 
                          y_vals - error_vals, 
                          y_vals + error_vals,
                          color=style['color'], 
                          alpha=0.3, 
                          interpolate=True)

    # Customize the plot
    if use_participant_count:
        ax.set_xlabel('# Labeled Training Participants', fontsize=14, fontweight='bold')
        # Set x-axis scale and limits for participant count
        min_participants = calculate_labeled_participants(label_fraction=0., total_participants=total_participants)
        ax.set_xlim(min_participants -0.2, total_participants +0.2)
        # Customize x-axis ticks for participant counts
        if total_participants <= 20:
            x_ticks = [min_participants, int(min_participants*2), total_participants]
        else:
            x_ticks = [min_participants, 10, 25, total_participants]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([str(x) for x in x_ticks])
    else:
        ax.set_xlabel('Label Fraction (% of Training Participants Labeled)', fontsize=14, fontweight='bold')
        # Set x-axis to log scale for better visualization of small fractions
        ax.set_xscale('log')
        ax.set_xlim(0.8, 120)
        # Customize x-axis ticks for percentages
        x_ticks = [1, 5, 10, 25, 50, 100]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f'{x}%' for x in x_ticks])

    y_name = 'AUROC' if metric == "auroc" else "AUPRC"
    ax.set_ylabel(y_name, fontsize=14, fontweight='bold')
    ax.set_title(f'{dataset_name}: Fine-tuned vs Pre-trained Encoders - {y_name}', fontsize=16, fontweight='bold', pad=20)

    # Set y-axis limits and ticks
    ax.set_ylim(0.3, 1.0)
    ax.set_yticks(np.arange(0.5, 1.05, 0.1))

    # Add grid
    ax.grid(False)
    ax.set_axisbelow(True)

    if metric == "auroc":
        ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.7, linewidth=2, label="Random Baseline")
    elif metric == "pr_auc":
        ax.axhline(y=pr_auc_baseline, color='black', linestyle='--', alpha=0.7, linewidth=2, label="Random Baseline")

    # Customize legend
    error_type = "Standard Error" if use_standard_error else "Standard Deviation"
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                       frameon=True, fancybox=True, shadow=False,
                       fontsize=11, title=f'Fine-tuning Effect (±{error_type})', title_fontsize=12,
                       ncol=2)

    # Improve overall appearance
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Fine-tuned vs pre-trained comparison plot saved to {save_path}")

    plt.show()
    plt.close()

    return fig, ax


def plot_ssl_comparison(df, metric="auroc", save_path=None, use_participant_count=False, 
                       total_participants=101, use_standard_error=False, include_features=False):
    """Create a comparison plot between different SSL methods.
    
    Args:
        df: DataFrame with SSL comparison results
        metric: Metric to plot ('auroc' or 'pr_auc')
        save_path: Path to save the plot
        use_participant_count: If True, show number of participants instead of percentages
        total_participants: Total number of training participants (default: 101)
        use_standard_error: If True, use standard error instead of standard deviation
        include_features: If True, include feature-engineered methods in the plot (default: True)
    """
    

    with plt.style.context(['ieee']):
        fig, ax = plt.subplots(figsize=(10, 8))  # Wider to accommodate right labels
        ax.grid(axis='y', linestyle='--', color='0.35', linewidth=0.7, alpha=0.7)

        # Group by method and calculate mean and std
        grouped = df.groupby(['method_label', 'label_fraction'])[metric].agg(['mean', 'std', 'count']).reset_index()

        # Filter out feature-engineered methods if include_features is False
        if not include_features:
            # Filter out methods that contain "Feature-engineered" in their name
            grouped = grouped[~grouped['method_label'].str.contains('Feature-engineered', case=False, na=False)]

        # Calculate standard error if requested
        if use_standard_error:
            grouped['error'] = grouped['std'] / np.sqrt(grouped['count'])
        else:
            grouped['error'] = grouped['std']

        # Calculate number of labeled participants
        def calculate_labeled_participants(label_fraction):
            return max(1, int(total_participants * label_fraction))

        grouped['n_labeled_participants'] = grouped['label_fraction'].apply(calculate_labeled_participants)

        # Define colors and markers for SSL methods
        method_styles = {
            # Feature-engineered methods - Same colors as main plot (10/5s was before F0746E)
            'Feature-engineered (Logistic Regression, 10s/5s)': {'color': '0.35', 'marker': 'x', 'linestyle': 'dotted', 'linewidth': 1.5},
            'Feature-engineered (Logistic Regression, 30s/10s)': {'color': '#E69F00', 'marker': 'o', 'linestyle': '--', 'linewidth': 2},


            # TSTCC (regular) - Light Blue
            'TSTCC (Logistic Regression, 10s/5s)': {'color': "#0E9FEB", 'marker': 'v', 'linestyle': '-', 'linewidth': 2},
            'TSTCC (MLP, 10s/5s)': {'color': '#88CCEE', 'marker': 's', 'linestyle': '-', 'linewidth': 2},
            'TSTCC (Linear, 10s/5s)': {'color': '#88CCEE', 'marker': '^', 'linestyle': '-', 'linewidth': 2},

            # TSTCC_S3 (soft version) - Darker Blue
            'TSTCC+S3 (Logistic Regression, 10s/5s)': {'color': "#005377", 'marker': 'v', 'linestyle': '--', 'linewidth': 1.5},
            'TSTCC+S3 (MLP, 10s/5s)': {'color': '#4477AA', 'marker': 's', 'linestyle': '--', 'linewidth': 2},
            'TSTCC+S3 (Linear, 10s/5s)': {'color': '#4477AA', 'marker': '^', 'linestyle': '--', 'linewidth': 2},

            # SimCLR (regular) - Light Green
            'SimCLR (Logistic Regression, 10s/5s)': {'color': '#DBABE0', 'marker': 's', 'linestyle': '-', 'linewidth': 1.5},
            'SimCLR (MLP, 10s/5s)': {'color': '#44AA99', 'marker': 'D', 'linestyle': '-', 'linewidth': 2},
            'SimCLR (Linear, 10s/5s)': {'color': '#44AA99', 'marker': 'p', 'linestyle': '-', 'linewidth': 2},

            # SimCLR_S3 (soft version) - Darker Green #D65048
            'SimCLR+S3 (Logistic Regression, 10s/5s)': {'color': '#78248E', 'marker': 's', 'linestyle': '--', 'linewidth': 1.5},
            'SimCLR+S3 (MLP, 10s/5s)': {'color': '#117733', 'marker': 'D', 'linestyle': '--', 'linewidth': 2},
            'SimCLR+S3 (Linear, 10s/5s)': {'color': '#117733', 'marker': 'p', 'linestyle': '--', 'linewidth': 2},

            # TS2Vec (regular) - Light Purple/Magenta
            'TS2Vec (Logistic Regression, 10s/5s)': {'color': '#D2B48C', 'marker': 'D', 'linestyle': '-', 'linewidth': 1.5},
            'TS2Vec (MLP, 10s/5s)': {'color': '#CC79A7', 'marker': '8', 'linestyle': '-', 'linewidth': 2},
            'TS2Vec (Linear, 10s/5s)': {'color': '#CC79A7', 'marker': '*', 'linestyle': '-', 'linewidth': 2},

            # TS2Vec_S3 (soft version) - Darker Purple/Magenta
            'TS2Vec+S3 (Logistic Regression, 10s/5s)': {'color': '#944D0F', 'marker': 'D', 'linestyle': '--', 'linewidth': 1.5},
            'TS2Vec+S3 (MLP, 10s/5s)': {'color': '#882255', 'marker': '8', 'linestyle': '--', 'linewidth': 2},
            'TS2Vec+S3 (Linear, 10s/5s)': {'color': '#882255', 'marker': '*', 'linestyle': '--', 'linewidth': 2},

            # InfoTS (regular) - Light Orange/Gold
            'InfoTS (Logistic Regression, 10s/5s)': {'color': '#EBB952', 'marker': '>', 'linestyle': '-', 'linewidth': 1.5},
            'InfoTS (MLP, 10s/5s)': {'color': '#DDCC77', 'marker': '<', 'linestyle': '-', 'linewidth': 2},
            'InfoTS (Linear, 10s/5s)': {'color': '#DDCC77', 'marker': '1', 'linestyle': '-', 'linewidth': 2},

            # InfoTS_S3 (soft version) - Darker Orange/Gold
            'InfoTS+S3 (Logistic Regression, 10s/5s)': {'color': '#F08D00', 'marker': '>', 'linestyle': '--', 'linewidth': 1.5},
            'InfoTS+S3 (MLP, 10s/5s)': {'color': '#AA6C39', 'marker': '<', 'linestyle': '--', 'linewidth': 2},
            'InfoTS+S3 (Linear, 10s/5s)': {'color': '#AA6C39', 'marker': '1', 'linestyle': '--', 'linewidth': 2},

            # SimCLR with TSTCC Encoder (regular) - Light Orange/Coral
            'SimCLR_TSTCC_Encoder (Logistic Regression, 10s/5s)': {'color': '#FF9999', 'marker': 's', 'linestyle': '-', 'linewidth': 1.5},
            'SimCLR_TSTCC_Encoder (MLP, 10s/5s)': {'color': '#FF9999', 'marker': 'X', 'linestyle': '-', 'linewidth': 2},
            'SimCLR_TSTCC_Encoder (Linear, 10s/5s)': {'color': '#FF9999', 'marker': '+', 'linestyle': '-', 'linewidth': 2},

            # SimCLR_S3 with TSTCC Encoder (soft version) - Darker Orange/Red
            'SimCLR+S3_TSTCC_Encoder (Logistic Regression, 10s/5s)': {'color': '#CC3333', 'marker': 's', 'linestyle': '--', 'linewidth': 1.5},
            'SimCLR+S3_TSTCC_Encoder (MLP, 10s/5s)': {'color': '#CC3333', 'marker': 'X', 'linestyle': '--', 'linewidth': 2},
            'SimCLR+S3_TSTCC_Encoder (Linear, 10s/5s)': {'color': '#CC3333', 'marker': '+', 'linestyle': '--', 'linewidth': 2},

            # SimCLR_TSTCC_Encoder_S3 (alternative naming) - Same as SimCLR+S3_TSTCC_Encoder
            'SimCLR_TSTCC_Encoder+S3 (Logistic Regression, 10s/5s)': {'color': '#CC3333', 'marker': 's', 'linestyle': '--', 'linewidth': 1.5},
            'SimCLR_TSTCC_Encoder+S3 (MLP, 10s/5s)': {'color': '#CC3333', 'marker': 'X', 'linestyle': '--', 'linewidth': 2},
            'SimCLR_TSTCC_Encoder+S3 (Linear, 10s/5s)': {'color': '#CC3333', 'marker': '+', 'linestyle': '--', 'linewidth': 2},
        }

        # Plot each method
        for method in grouped['method_label'].unique():
            method_data = grouped[grouped['method_label'] == method].sort_values('label_fraction')

            # Create display name for legend (replace _S3 with +S3 for display)
            display_method = method.replace("_S3", "+S3") if "_S3" in method else method

            # Use display name for style lookup
            style = method_styles.get(display_method, {'color': 'black', 'marker': 'o', 'linestyle': '-', 'linewidth': 2.})

            # Choose x-axis values based on use_participant_count parameter
            if use_participant_count:
                x_vals = method_data['n_labeled_participants']
            else:
                x_vals = method_data['label_fraction'] * 100
            y_vals = method_data['mean']

            # Plot main line
            linewidth = style.get('linewidth', 2.)
            ax.plot(x_vals, y_vals,
                    color=style['color'],
                    marker=style['marker'],
                    linestyle=style['linestyle'],
                    linewidth=linewidth,
                    markersize=8,
                    label=display_method if "Feature" in display_method else display_method.split(" ")[0],
                    markerfacecolor='white',
                    markeredgewidth=2,
                    markeredgecolor=style['color'])

            # Add error visualization with fill_between if we have multiple seeds
            if method_data['count'].max() > 1:
                error_vals = method_data['error'].fillna(0)

                # Use fill_between for better uncertainty visualization
                ax.fill_between(x_vals,
                              y_vals - error_vals,
                              y_vals + error_vals,
                              color=style['color'],
                              alpha=0.20,
                              interpolate=True)

        # Customize the plot
        if use_participant_count:
            ax.set_xlabel('# Labeled Training Participants', fontsize=20)
            # Set x-axis scale and limits for participant count
            ax.set_xscale('log')
            ax.set_xlim(0.8, 110)
            # Customize x-axis ticks for participant counts
            x_ticks = [1, 2, 5, 10, 25, 50, 101]
            ax.set_xticks(x_ticks)
            ax.set_xticklabels([str(x) for x in x_ticks])
        else:
            ax.set_xlabel('Label Fraction (% of Training Participants Labeled)', fontsize=20)
            # Set x-axis to log scale for better visualization of small fractions
            ax.set_xscale('log')
            ax.set_xlim(0.8, 120)
            # Customize x-axis ticks for percentages
            x_ticks = [1, 2.5, 5, 10, 25, 50, 100]
            ax.set_xticks(x_ticks)
            ax.set_xticklabels([f'{x}%' for x in x_ticks])

        y_name = 'AUROC' if metric == "auroc" else "AUPRC"
        ax.set_ylabel(y_name, fontsize=20)
        # ax.set_title(f'SSL Methods Comparison: {y_name} vs Label Fraction', fontsize=16, fontweight='bold', pad=20)

        # Set y-axis limits and ticks
        ax.set_ylim(0.5, 1.0) if y_name == "PR-AUC" else ax.set_ylim(0.45, 1.0)
        ax.set_yticks(np.arange(0.5, 1.05, 0.1))

        # Add grid
        ax.set_axisbelow(True)

        ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=False,
                           fontsize=20, ncols=2)

        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        # Improve overall appearance
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=500, bbox_inches='tight', facecolor='white')
            print(f"Plot saved to {save_path}")

        plt.show()
        plt.close()

    return fig, ax


def load_zero_shot_results(base_path, dataset_name):
    """
    Load zero-shot performance results from the hierarchical folder structure.
    Expected structure: results/Transfer_learning/[dataset]/zero_shot_performance/[method]/[model_type]/[seed]/[label_fraction]/zero_shot_results.json
                       results/Transfer_learning/[dataset]/zero_shot_performance/feature_engineered/[model_type]/[seed]/[label_fraction]/[window_size]/[window_shift]/zero_shot_results.json
    
    Args:
        base_path: Base results path
        dataset_name: 'WESAD' or 'StressID'
    """
    results = []
    base_path = Path(base_path)
    
    # Path to zero-shot results for this dataset
    zero_shot_base = base_path / "Transfer_learning" / dataset_name / "zero_shot_performance"
    
    if not zero_shot_base.exists():
        print(f"Warning: Zero-shot path {zero_shot_base} does not exist")
        return pd.DataFrame(results)
    
    # Define the expected method paths
    method_paths = [
        ("TSTCC", "logistic_regression", "TSTCC"),
        ("feature_engineered", "logistic_regression", "Feature-engineered"),
    ]
    
    for method_folder, model_type, learning_method in method_paths:
        method_path = zero_shot_base / method_folder / model_type
        
        if not method_path.exists():
            print(f"Warning: Method path {method_path} does not exist")
            continue
        
        # Look for seed folders
        for seed_folder in method_path.iterdir():
            if seed_folder.is_dir() and seed_folder.name.isdigit():
                seed = int(seed_folder.name)
                
                # Look for label fraction folders
                for label_folder in seed_folder.iterdir():
                    if label_folder.is_dir():
                        try:
                            label_fraction = float(label_folder.name)
                            
                            # Handle different file structures
                            if learning_method == "Feature-engineered":
                                # Feature-engineered has window_size/window_shift structure
                                window_combinations = [
                                    (30, 10),  # 30s windows, 10s shift
                                ]
                                
                                for window_size, window_shift in window_combinations:
                                    json_file = label_folder / str(window_size) / str(window_shift) / "zero_shot_results.json"
                                    
                                    if json_file.exists():
                                        try:
                                            with open(json_file, 'r') as f:
                                                data = json.load(f)
                                            results.append({
                                                'learning_method': learning_method,
                                                'model_type': model_type,
                                                'seed': seed,
                                                'label_fraction': label_fraction,
                                                'window_size': window_size,
                                                'window_shift': window_shift,
                                                'auroc': data.get('zero_shot_roc_auc', np.nan),
                                                'accuracy': data.get('zero_shot_accuracy', np.nan),
                                                'pr_auc': data.get('zero_shot_pr_auc', np.nan),
                                                'f1': data.get('zero_shot_f1_score', np.nan),
                                                'balanced_accuracy': data.get('zero_shot_balanced_accuracy', np.nan),
                                            })
                                        except (json.JSONDecodeError, KeyError) as e:
                                            print(f"Error reading {json_file}: {e}")
                                            continue
                            else:
                                # SSL methods have direct structure
                                json_file = label_folder / "zero_shot_results.json"
                                
                                if json_file.exists():
                                    try:
                                        with open(json_file, 'r') as f:
                                            data = json.load(f)
                                        results.append({
                                            'learning_method': learning_method,
                                            'model_type': model_type,
                                            'seed': seed,
                                            'label_fraction': label_fraction,
                                            'window_size': 10,  # Default for SSL methods
                                            'window_shift': 5,
                                            'auroc': data.get('zero_shot_roc_auc', np.nan),
                                            'accuracy': data.get('zero_shot_accuracy', np.nan),
                                            'pr_auc': data.get('zero_shot_pr_auc', np.nan),
                                            'f1': data.get('zero_shot_f1_score', np.nan),
                                            'balanced_accuracy': data.get('zero_shot_balanced_accuracy', np.nan),
                                        })
                                    except (json.JSONDecodeError, KeyError) as e:
                                        print(f"Error reading {json_file}: {e}")
                                        continue
                                        
                        except ValueError:
                            # Skip folders that aren't numeric label fractions
                            continue
    
    return pd.DataFrame(results)


def create_zero_shot_method_labels(df):
    """Create readable method labels for zero-shot results"""
    def create_label(row):
        method = row['learning_method']
        model = row['model_type'].replace('_', ' ').title()
        window = f"{row['window_size']}s/{row['window_shift']}s"
        
        if method == "TSTCC":
            return f"TSTCC ({model}, {window})"
        elif method == "Feature-engineered":
            return f"Feature-engineered ({model}, {window})"
        else:
            return f"{method} ({model}, {window})"
    
    df['method_label'] = df.apply(create_label, axis=1)
    return df


def plot_zero_shot_results(
        df, dataset_name, metric="auroc", save_path=None, use_participant_count=False, total_participants=None,
        use_standard_error=False):
    """Create a plot showing zero-shot transfer performance for a specific dataset
    
    Args:
        df: DataFrame with zero-shot results
        dataset_name: Name of the dataset (for title)
        metric: Metric to plot ('auroc' or 'pr_auc')
        save_path: Path to save the plot
        use_participant_count: If True, show number of participants instead of percentages
        total_participants: Total number of training participants for this dataset
        use_standard_error: If True, use standard error instead of standard deviation
    """
    plt.style.use('default')
    sns.set_palette("husl")

    # Set default participant counts if not provided
    if total_participants is None:
        if dataset_name == "WESAD":
            total_participants = 15  # Adjust based on actual WESAD participant count
        elif dataset_name == "StressID":
            total_participants = 35  # Adjust based on actual StressID participant count
        else:
            total_participants = 101  # Default fallback
    
    # Set dataset-specific PR-AUC baseline (random chance for each dataset)
    if dataset_name == "WESAD":
        pr_auc_baseline = 0.3625
    elif dataset_name == "StressID":
        pr_auc_baseline = 0.3510
    else:
        pr_auc_baseline = 0.5736  # Default fallback
    
    # Set up the plotting style
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7)

    # Group by method and calculate mean and std
    grouped = df.groupby(['method_label', 'label_fraction'])[metric].agg(['mean', 'std', 'count']).reset_index()
    
    # Calculate standard error if requested
    if use_standard_error:
        grouped['error'] = grouped['std'] / np.sqrt(grouped['count'])
    else:
        grouped['error'] = grouped['std']
    
    # Calculate number of labeled participants for training data (ECG dataset)
    def calculate_labeled_participants(label_fraction, total_participants):
        # Using ECG dataset size (127 participants)
        if total_participants <= 20:
            return max(3, int(total_participants * label_fraction))
        return max(1, int(total_participants * label_fraction))
    
    grouped['n_labeled_participants'] = grouped['label_fraction'].apply(calculate_labeled_participants, total_participants=127)

    # Define colors and markers for zero-shot methods

    method_styles = {
        'TSTCC (Logistic Regression, 10s/5s)': {'color': '#ffa600', 'marker': 'v', 'linestyle': '-'},
        'Feature-engineered (Logistic Regression, 30s/10s)': {'color': '#bc5090', 'marker': 'o', 'linestyle': '-'},
    }

    # Plot each method
    for method_label in grouped['method_label'].unique():
        method_data = grouped[grouped['method_label'] == method_label].sort_values('label_fraction')
        
        if len(method_data) == 0:
            continue
            
        # Get style for this method
        style = method_styles.get(method_label, {'color': 'gray', 'marker': 'o', 'linestyle': '-'})
        
        if use_participant_count:
            x_values = method_data['n_labeled_participants']
        else:
            x_values = method_data['label_fraction']
        
        # Plot line with error bars (handle NaN values)
        error_values = method_data['error'].fillna(0)  # Replace NaN with 0 for error bars
        ax.errorbar(x_values, method_data['mean'], yerr=error_values,
                   label=method_label, marker=style['marker'], color=style['color'], 
                   linestyle=style['linestyle'], capsize=5, capthick=2, 
                   linewidth=2, markersize=8, alpha=0.8)

    # Add baseline line for PR-AUC
    if metric == "pr_auc":
        ax.axhline(y=pr_auc_baseline, color='0.45', linestyle='solid', alpha=0.7,
                  label=f'Random Baseline')
    else:
        ax.axhline(y=0.5, color='0.45', linestyle='solid', alpha=0.7,
                   label=f'Random Baseline')
    # Formatting
    if use_participant_count:
        ax.set_xlabel('# Labeled Training Participants', fontsize=14)
    else:
        ax.set_xlabel('Label Fraction', fontsize=14)
    
    if metric == "auroc":
        ax.set_ylabel('AUROC', fontsize=14)
        title = f'{dataset_name} Zero-Shot Transfer - AUROC'
    else:
        ax.set_ylabel('PR-AUC', fontsize=14)
        title = f'{dataset_name} Zero-Shot Transfer - PR-AUC'
    
    error_type = "Standard Error" if use_standard_error else "Standard Deviation"
    # ax.set_title(f'{title}\n(Error bars: {error_type})', fontsize=16, pad=20)

    # Formatting
    # ax.grid(True, alpha=0.2)
    ax.legend(loc='upper left', fontsize=20, frameon=True, fancybox=True, shadow=True)
    ax.set_axisbelow(True)

    # Set y-axis limits based on metric
    if metric == "auroc":
        ax.set_ylim(0.4, 1.0)
    else:
        ax.set_ylim(0.3, 0.7)
    
    # Set x-axis scale and ticks
    if use_participant_count:
        ax.set_xscale('log')
        ax.set_xticks([1, 2, 5, 10, 20, 50, 127])
        ax.set_xticklabels(['1', '2', '5', '10', '20', '50', '127'])
    else:
        ax.set_xscale('log')
        ax.set_xticks([0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0])
        ax.set_xticklabels(['1%', '2.5%', '5%', '10%', '25%', '50%', '100%'])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Plot saved to {save_path}")

    plt.show()
    plt.close()

    return fig, ax


def main():
    """Main function to load data and create plots"""

    base_path = RESULTS_PATH
    use_participant_count = False
    create_directory(FIGURES_PATH)

    print("Loading results from folder structure...")
    df = load_results_from_structure(base_path)

    if df.empty:
        print("No results found! Please check your folder structure and paths.")
        print("Expected structure:")
        print("ECG/[Supervised|TSTCC]/[cnn|linear|mlp]/[seed]/[label_fraction]/test_results.json")
        print("ECG_features/[linear|mlp]/[seed]/[label_fraction]/test_results.json")
        return

    print(f"Successfully loaded {len(df)} results!")

    # Create method labels
    df = create_method_labels(df)

    # Print summary statistics
    print_summary_statistics(df)

    # Create the plot
    print("\nCreating plot...")

    # Also create plots with standard error
    plot_metric_vs_label_fraction(
        df,
        save_path=os.path.join(FIGURES_PATH,'ecg_auroc_vs_label_fraction_stderr.png'),
        use_participant_count=use_participant_count,
        use_standard_error=True)

    plot_metric_vs_label_fraction(
        df,
        metric="pr_auc",
        save_path=os.path.join(FIGURES_PATH,'ecg_pr_auc_vs_label_fraction_stderr.png'),
        use_participant_count=use_participant_count,
        use_standard_error=True
    )

    # Also save the data to CSV for further analysis
    df.to_csv('ecg_results_summary.csv', index=False)

    # Create SSL comparison plots
    print("\nLoading SSL comparison results...")
    ssl_methods_to_compare = ['TSTCC_S3', 'TSTCC', 'TS2Vec', 'TS2Vec_S3', "InfoTS", "InfoTS_S3", 'SimCLR_S3', 'SimCLR']

    ssl_comparison_df = load_ssl_comparison_results(base_path, ssl_methods=ssl_methods_to_compare,
                                                    include_features=True)
    
    if not ssl_comparison_df.empty:
        print(f"Successfully loaded {len(ssl_comparison_df)} SSL comparison results!")
        
        # Create method labels for SSL comparison data
        ssl_comparison_df = create_ssl_method_labels(ssl_comparison_df)
        
        # Create SSL comparison plots
        print("Creating SSL methods comparison plots...")
        plot_ssl_comparison(
            ssl_comparison_df,
            metric="auroc",
            save_path=os.path.join(FIGURES_PATH, 'ssl_methods_auroc_comparison.png'),
            use_participant_count=use_participant_count,
            use_standard_error=True,
            include_features=False,
        )

        plot_ssl_comparison(
            ssl_comparison_df,
            metric="pr_auc",
            save_path=os.path.join(FIGURES_PATH, 'ssl_methods_pr_auc_comparison.png'),
            use_participant_count=use_participant_count,
            use_standard_error=True,
            include_features=False,
        )

        # Save SSL comparison data to CSV
        ssl_comparison_df.to_csv(os.path.join(RESULTS_PATH, 'ssl_methods_comparison_results.csv'), index=False)
        print("SSL comparison results saved to 'ssl_methods_comparison_results.csv'")
    else:
        print("No SSL comparison results found!")

    # # Create feature vs SSL comparison plot
    comparison_df = load_feature_vs_ssl_comparison_results(base_path)

    if not comparison_df.empty:
        print(f"Successfully loaded {len(comparison_df)} comparison results!")

        # Create method labels for comparison data
        comparison_df = create_feature_vs_ssl_method_labels(comparison_df)

        # Create comparison plots
        print("Creating feature vs SSL comparison plots...")
        plot_feature_vs_ssl_comparison(
            comparison_df,
            metric="auroc",
            save_path=os.path.join(FIGURES_PATH, 'feature_vs_ssl_auroc_comparison.png'),
            use_participant_count=use_participant_count
        )
        plot_feature_vs_ssl_comparison(
            comparison_df,
            metric="pr_auc",
            save_path=os.path.join(FIGURES_PATH, 'feature_vs_ssl_pr_auc_comparison.png'),
            use_participant_count=use_participant_count
        )

        # Save comparison data to CSV
        comparison_df.to_csv(os.path.join(RESULTS_PATH, 'feature_vs_ssl_comparison_results.csv'), index=False)
        print("Comparison results saved to 'feature_vs_ssl_comparison_results.csv'")
    else:
        print("No feature vs SSL comparison results found!")


if __name__ == "__main__":
    results_df = main()