import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from utils.helper_paths import RESULTS_PATH
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
        "ECG/Supervised/tcn",
        "ECG/Supervised/transformer",
        # "ECG/Supervised/deep_ecg_net",
        "ECG/TSTCC/logistic_regression",
        # "ECG/TSTCC/mlp",
        "ECG/TSTCC/linear",
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
            model_type = path_parts[2]

        # Look for seed folders (including "42_s3" for TSTCC)
        for seed_folder in method_path.iterdir():
            if seed_folder.is_dir():
                # Handle both numeric seeds (3, 5, 7, 9, 42) and special seeds like "42_s3"
                if seed_folder.name.isdigit():
                    seed = int(seed_folder.name)
                elif seed_folder.name == "42_s3":
                    seed = 43  # Use 43 to distinguish from regular seed 42
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
                                (30, 10), # 30s windows, 10s shift
                                # (30, 15)   # 30s windows, 15s shift
                            ]

                            for window_size, window_shift in window_combinations:
                                if learning_method == "TSTCC" and window_size == 30:
                                    continue
                                json_file = label_folder / str(window_size) / str(window_shift) / "test_results.json"
                                
                                # Special handling for different file structures
                                if not json_file.exists():
                                    # Try with additional subfolder for some models
                                    if model_type == "xgboost":
                                        json_file = label_folder / str(window_size) / str(window_shift) / "0.75" / "test_results.json"
                                    elif model_type in ["logistic_regression", "mlp"] and learning_method == "TSTCC":
                                        json_file = label_folder / str(window_size) / str(window_shift) / "1.0" / "test_results.json"
                                
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
                
            window_shift = int(window_shift_folder.name)
            
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
    
    # Define the transfer types
    transfer_types = ["pretrained_encoder", "trained_from_scratch", "cnn"]

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
                                (30, 5),   # 30s windows, 5s shift
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
                transfer_label = "Pre-trained" if row['transfer_type'] == 'pretrained_encoder' else "From Scratch"
                return f"{transfer_label} ({clean_model}, {window_info})"
        elif row['method_type'] == 'Features_Baseline':
            return f"Feature Baseline ({clean_model}, {window_info})"
        else:
            return f"{row['learning_method']} ({clean_model}, {window_info})"

    df['method_label'] = df.apply(make_label, axis=1)
    return df


def plot_transfer_learning_results(df, dataset_name, metric="auroc", save_path=None, use_participant_count=False, total_participants=None, use_standard_error=False):
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
    
    grouped['n_labeled_participants'] = grouped['label_fraction'].apply(calculate_labeled_participants, total_participants=total_participants)

    # Define colors and markers for transfer learning methods
    method_styles = {
        # Pre-trained encoder methods (solid lines) - TSTCC in light blue
        'Pre-trained (Logistic Regression, 10s/5s)': {'color': '#88CCEE', 'marker': '^', 'linestyle': '-'},
        'Pre-trained (MLP, 10s/5s)': {'color': '#88CCEE', 'marker': 'o', 'linestyle': '-'},
        'Pre-trained (Logistic Regression, 30s/5s)': {'color': '#88CCEE', 'marker': 'v', 'linestyle': '-'},
        'Pre-trained (Logistic Regression, 30s/10s)': {'color': '#88CCEE', 'marker': 's', 'linestyle': '--'},
        'Pre-trained (Logistic Regression, 30s/15s)': {'color': '#88CCEE', 'marker': '^', 'linestyle': ':'},
        'Pre-trained (MLP, 30s/5s)': {'color': '#88CCEE', 'marker': '8', 'linestyle': '-'},
        'Pre-trained (MLP, 30s/10s)': {'color': '#88CCEE', 'marker': 'D', 'linestyle': '--'},
        'Pre-trained (MLP, 30s/15s)': {'color': '#88CCEE', 'marker': 'h', 'linestyle': ':'},

        # From scratch methods (dashed lines) - TSTCC in light blue
        'From Scratch (Logistic Regression, 10s/5s)': {'color': '#CC79A7', 'marker': 'v', 'linestyle': '-'},
        'From Scratch (MLP, 10s/5s)': {'color': '#009E73', 'marker': 'p', 'linestyle': '-'},
        'From Scratch (Logistic Regression, 30s/5s)': {'color': '#CC79A7', 'marker': 'o', 'linestyle': '-'},
        'From Scratch (Logistic Regression, 30s/10s)': {'color': '#CC79A7', 'marker': 'x', 'linestyle': '--'},
        'From Scratch (Logistic Regression, 30s/15s)': {'color': '#CC79A7', 'marker': '*', 'linestyle': ':'},
        'From Scratch (MLP, 30s/5s)': {'color': '#009E73', 'marker': '8', 'linestyle': '-'},
        'From Scratch (MLP, 30s/10s)': {'color': '#009E73', 'marker': '+', 'linestyle': '--'},
        'From Scratch (MLP, 30s/15s)': {'color': '#009E73', 'marker': '1', 'linestyle': ':'},
        
        # Feature baseline methods (thick dotted lines) - ALL ORANGE
        'Feature Baseline (Logistic Regression, 30s/5s)': {'color': '#E69F00', 'marker': 'v', 'linestyle': ':', 'linewidth': 3},
        'Feature Baseline (Logistic Regression, 30s/10s)': {'color': '#E69F00', 'marker': 'h', 'linestyle': ':', 'linewidth': 3},
        'Feature Baseline (Logistic Regression, 30s/15s)': {'color': '#E69F00', 'marker': 's', 'linestyle': ':', 'linewidth': 3},
        'Feature Baseline (MLP, 30s/5s)': {'color': '#E69F00', 'marker': '1', 'linestyle': ':', 'linewidth': 3},
        'Feature Baseline (MLP, 30s/10s)': {'color': '#E69F00', 'marker': '8', 'linestyle': ':', 'linewidth': 3},
        'Feature Baseline (MLP, 30s/15s)': {'color': '#E69F00', 'marker': 'D', 'linestyle': ':', 'linewidth': 3},

        # CNN Supervised methods
        'Supervised (CNN, 10s/5s)': {'color': '#FF6B35', 'marker': 's', 'linestyle': '-', 'linewidth': 3},
        'Supervised (CNN, 30s/5s)': {'color': '#FF6B35', 'marker': 'v', 'linestyle': '-', 'linewidth': 3},
        'Supervised (CNN, 30s/10s)': {'color': '#FF6B35', 'marker': 'o', 'linestyle': '--', 'linewidth': 3},
        'Supervised (CNN, 30s/15s)': {'color': '#FF6B35', 'marker': '^', 'linestyle': ':', 'linewidth': 3},
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
        linewidth = style.get('linewidth', 2.5)
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
        ax.set_xlabel('# Labeled Training Participants', fontsize=14, fontweight='bold')
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
        ax.set_xlabel('Label Fraction (% of Training Participants Labeled)', fontsize=14, fontweight='bold')
        # Set x-axis to log scale for better visualization of small fractions
        ax.set_xscale('log')
        ax.set_xlim(0.8, 120)
        # Customize x-axis ticks for percentages
        x_ticks = [1, 5, 10, 25, 50, 100]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f'{x}%' for x in x_ticks])

    y_name = 'AUROC' if metric == "auroc" else "PR-AUC"
    ax.set_ylabel(y_name, fontsize=14, fontweight='bold')
    ax.set_title(f'{dataset_name} Transfer Learning: {y_name} vs Label Fraction', fontsize=16, fontweight='bold', pad=20)

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
                       fontsize=11, title=f'Transfer Learning Methods (±{error_type})', title_fontsize=12,
                       ncol=2)  # Optional: arrange legend items horizontally

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

    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")

    fig, ax = plt.subplots(figsize=(12, 8))

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

    # Define colors and markers for different methods
    method_styles = {
        # Feature-engineered (different window configurations)
        'Feature-engineered (Logistic Regression, 30s/5s)': {'color': '#ff6361', 'marker': 'v', 'linestyle': '-'},
        'Feature-engineered (Logistic Regression, 30s/10s)': {'color': '#ff6361', 'marker': 'o', 'linestyle': '--'},
        'Feature-engineered (Logistic Regression, 30s/15s)': {'color': '#665191', 'marker': 's', 'linestyle': ':'},
        'Feature-engineered (Logistic Regression, 10s/5s)': {'color': '#E69F00', 'marker': 'x', 'linestyle': '-'},
        'Feature-engineered (MLP, 30s/5s)': {'color': '#E69F00', 'marker': '8', 'linestyle': '-'},
        'Feature-engineered (MLP, 30s/10s)': {'color': '#E69F00', 'marker': 'D', 'linestyle': '--'},
        'Feature-engineered (MLP, 30s/15s)': {'color': '#E69F00', 'marker': 'h', 'linestyle': ':'},

        #082a54 Dark Blue
        #E69F00 Orange
        #2066a8

        # Supervised methods (10s)
        'Supervised (CNN, 10s/5s)': {'color': '#D55E00', 'marker': '^', 'linestyle': '-'},
        'Supervised (TCN, 10s/5s)': {'color': '#44AA99', 'marker': '>', 'linestyle': '-'},
        'Supervised (Transformer, 10s/5s)': {'color': '#58508d', 'marker': '<', 'linestyle': '-'},
        'Supervised (DeepECGNet, 10s/5s)': {'color': '#117733', 'marker': 'o', 'linestyle': '-'},
        
        # TSTCC (10s and 30s) - ALL LIGHT BLUE
        'TSTCC (Logistic Regression, 10s/5s)': {'color': '#88CCEE', 'marker': 'v', 'linestyle': '-'},
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
        
        # Supervised methods (30s) - in case they exist
        'Supervised (CNN, 30s/5s)': {'color': '#D55E00', 'marker': 'v', 'linestyle': '-'},
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
        style = method_styles.get(method, {'color': 'black', 'marker': 'o', 'linestyle': '-'})

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
        ax.set_xlabel('# Labeled Training Participants', fontsize=14, fontweight='bold')
        # Set x-axis scale and limits for participant count
        ax.set_xscale('log')
        ax.set_xlim(0.8, 110)
        # Customize x-axis ticks for participant counts
        x_ticks = [1, 2, 5, 10, 25, 50, 101]
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

    y_name = 'AUROC' if metric == "auroc" else "PR-AUC"
    ax.set_ylabel(y_name, fontsize=14, fontweight='bold')
    # ax.set_title('ECG Classification Performance vs Label Fraction', fontsize=16, fontweight='bold', pad=20)

    # Set y-axis limits and ticks
    ax.set_ylim(0.3, 1.0)
    ax.set_yticks(np.arange(0.5, 1.05, 0.1))

    # Add grid
    ax.grid(False)
    ax.set_axisbelow(True)

    # Add horizontal line at AUROC = 0.5 (random chance)
    if metric == "auroc":
        ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.7, linewidth=2, label="Random Baseline")

    elif metric == "pr_auc":
        ax.axhline(y=0.5736, color='black', linestyle='--', alpha=0.7, linewidth=2, label="Random Baseline")

    # Customize legend
    error_type = "Standard Error" if use_standard_error else "Standard Deviation"
    ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=False,
                       fontsize=11, title=f'Methods (±{error_type})', title_fontsize=12)

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
    feature_models = ["logistic_regression", "random_forest", "xgboost"]
    
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
                        # (30, 10),  # 30s windows, 10s shift
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
    tstcc_path = base_path / "ECG" / "TSTCC"
    tstcc_models = ["logistic_regression", "xgboost"]
    
    for model_type in tstcc_models:
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
                        json_file = label_folder / str(window_size) / str(window_shift) / "test_results.json"
                        
                        # Special handling for xgboost with 0.75 subfolder
                        if model_type == "xgboost":
                            json_file = label_folder / str(window_size) / str(window_shift) / "0.75" / "test_results.json"
                        
                        if json_file.exists():
                            try:
                                with open(json_file, 'r') as f:
                                    data = json.load(f)
                                results.append({
                                    'method_type': 'SSL_TSTCC',
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
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")

    fig, ax = plt.subplots(figsize=(12, 8))

    # Group by method and calculate mean and std
    grouped = df.groupby(['method_label', 'label_fraction'])[metric].agg(['mean', 'std', 'count']).reset_index()
    
    # Calculate number of labeled participants
    def calculate_labeled_participants(label_fraction):
        return max(1, int(total_participants * label_fraction))
    
    grouped['n_labeled_participants'] = grouped['label_fraction'].apply(calculate_labeled_participants)

    # Define colors and markers for comparison methods
    method_styles = {
        # Feature-engineered methods - Same orange as main plots
        'Feature-engineered (Logistic Regression, 10s/5s)': {'color': '#8ba6b5', 'marker': 'o', 'linestyle': '-'},
        'Feature-engineered (XGBoost, 10s/5s)': {'color': '#8ba6b5', 'marker': 'x', 'linestyle': '--'},

        'Feature-engineered (Logistic Regression, 30s/15s)': {'color': '#E69F00', 'marker': '^', 'linestyle': '-'},
        'Feature-engineered (Random Forest, 30s/15s)': {'color': '#E69F00', 'marker': 'D', 'linestyle': '-'},
        'Feature-engineered (XGBoost, 30s/15s)': {'color': '#E69F00', 'marker': 'p', 'linestyle': '--'},
        
        # SSL TSTCC methods - Same light blue as main plots (only 10s/5s)
        'SSL TSTCC (Logistic Regression, 10s/5s)': {'color': '#88CCEE', 'marker': 'o', 'linestyle': '-'},
        'SSL TSTCC (XGBoost, 10s/5s)': {'color': '#88CCEE', 'marker': 'v', 'linestyle': '--'},
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
        linewidth = style.get('linewidth', 2.5)
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

        # Add error bars if we have multiple seeds
        if method_data['count'].max() > 1:
            yerr = method_data['std'].fillna(0)
            ax.errorbar(x_vals, y_vals, yerr=yerr,
                        color=style['color'], alpha=0.3, capsize=4, capthick=1.5)

    # Customize the plot
    if use_participant_count:
        ax.set_xlabel('# Labeled Training Participants', fontsize=14, fontweight='bold')
        # Set x-axis scale and limits for participant count
        ax.set_xscale('log')
        ax.set_xlim(0.8, 110)
        # Customize x-axis ticks for participant counts
        x_ticks = [1, 2, 5, 10, 25, 50, 101]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([str(x) for x in x_ticks])
    else:
        ax.set_xlabel('Label Fraction (% of Training Participants Labeled)', fontsize=14, fontweight='bold')
        # Set x-axis to log scale for better visualization of small fractions
        ax.set_xscale('log')
        ax.set_xlim(0.8, 120)
        # Customize x-axis ticks for percentages
        x_ticks = [1, 2.5, 5, 10, 25, 50, 100]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f'{x}%' for x in x_ticks])

    y_name = 'AUROC' if metric == "auroc" else "PR-AUC"
    ax.set_ylabel(y_name, fontsize=14, fontweight='bold')
    ax.set_title(f'Feature-Engineered vs SSL Comparison: {y_name} vs Label Fraction', fontsize=16, fontweight='bold', pad=20)

    # Set y-axis limits and ticks
    ax.set_ylim(0.3, 1.0)
    ax.set_yticks(np.arange(0.5, 1.05, 0.1))

    # Add grid
    ax.grid(False)
    ax.set_axisbelow(True)

    # Add horizontal line at random baseline
    if metric == "auroc":
        ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.7, linewidth=2, label="Random Baseline")
    elif metric == "pr_auc":
        ax.axhline(y=0.5736, color='black', linestyle='--', alpha=0.7, linewidth=2, label="Random Baseline")

    # Customize legend
    ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=False,
                       fontsize=10, title='Methods', title_fontsize=11)

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


def main():
    """Main function to load data and create plots"""

    # Set the base path (adjust this to your actual path)
    base_path = RESULTS_PATH  # Uses the RESULTS_PATH from utils.helper_paths
    use_participant_count = True

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
    # plot_metric_vs_label_fraction(df, save_path='ecg_auroc_vs_label_fraction.png', use_participant_count=use_participant_count)
    # plot_metric_vs_label_fraction(df, metric="pr_auc", save_path='ecg_pr_auc_vs_label_fraction.png', use_participant_count=use_participant_count)
    #
    # Also create plots with standard error
    plot_metric_vs_label_fraction(df, save_path='ecg_auroc_vs_label_fraction_stderr.png', use_participant_count=use_participant_count, use_standard_error=True)
    plot_metric_vs_label_fraction(df, metric="pr_auc", save_path='ecg_pr_auc_vs_label_fraction_stderr.png', use_participant_count=use_participant_count, use_standard_error=True)

    # Also save the data to CSV for further analysis
    df.to_csv('ecg_results_summary.csv', index=False)
    print("Results saved to 'ecg_results_summary.csv'")

    # Create feature vs SSL comparison plot
    print("\nLoading feature vs SSL comparison results...")
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
            save_path='feature_vs_ssl_auroc_comparison.png',
            use_participant_count=use_participant_count
        )
        plot_feature_vs_ssl_comparison(
            comparison_df, 
            metric="pr_auc", 
            save_path='feature_vs_ssl_pr_auc_comparison.png',
            use_participant_count=use_participant_count
        )
        
        # Save comparison data to CSV
        comparison_df.to_csv('feature_vs_ssl_comparison_results.csv', index=False)
        print("Comparison results saved to 'feature_vs_ssl_comparison_results.csv'")
    else:
        print("No feature vs SSL comparison results found!")

    # Load and plot transfer learning results for both datasets
    print("\nLoading transfer learning results...")
    datasets = ["WESAD", "StressID"]
    
    for dataset_name in datasets:
        print(f"\nProcessing {dataset_name} transfer learning results...")
        
        # Load combined transfer learning and feature baseline results for this dataset
        transfer_df = load_combined_transfer_learning_results(base_path, dataset_name)
        
        if transfer_df.empty:
            print(f"No transfer learning results found for {dataset_name}")
            continue
            
        print(f"Successfully loaded {len(transfer_df)} transfer learning results for {dataset_name}")
        
        # Create method labels for transfer learning data
        transfer_df = create_method_labels(transfer_df)
        
        # Set participant counts based on dataset
        if dataset_name == "WESAD":
            total_participants = 12  #
        elif dataset_name == "StressID":
            total_participants = 52  #
        else:
            total_participants = 101  # Default fallback
        
        # Create transfer learning plots for AUROC
        print(f"Creating {dataset_name} transfer learning AUROC plot...")
        plot_transfer_learning_results(
            transfer_df, 
            dataset_name=dataset_name,
            metric="auroc", 
            save_path=f'{dataset_name.lower()}_transfer_learning_auroc.png',
            use_participant_count=use_participant_count,
            total_participants=total_participants,
            use_standard_error=True
        )
        
        # Create transfer learning plots for PR-AUC
        print(f"Creating {dataset_name} transfer learning PR-AUC plot...")
        plot_transfer_learning_results(
            transfer_df, 
            dataset_name=dataset_name,
            metric="pr_auc", 
            save_path=f'{dataset_name.lower()}_transfer_learning_pr_auc.png',
            use_participant_count=use_participant_count,
            total_participants=total_participants,
            use_standard_error=True
        )
        
        # Save transfer learning data to CSV
        transfer_df.to_csv(f'{dataset_name.lower()}_transfer_learning_results_summary.csv', index=False)
        print(f"{dataset_name} transfer learning results saved to '{dataset_name.lower()}_transfer_learning_results_summary.csv'")

    return df


if __name__ == "__main__":
    results_df = main()