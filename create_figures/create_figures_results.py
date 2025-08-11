import json
import os
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
        "ECG/TSTCC/logistic_regression",
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

        # Look for seed folders (like "42")
        for seed_folder in method_path.iterdir():
            if seed_folder.is_dir() and seed_folder.name.isdigit():
                seed = int(seed_folder.name)

                # Look for label fraction folders
                for label_folder in seed_folder.iterdir():
                    if label_folder.is_dir():
                        try:
                            label_fraction = float(label_folder.name)
                            
                            if method_type == "ECG_features":
                                # ECG_features: direct path to test_results.json
                                json_file = label_folder / "test_results.json"
                                if json_file.exists():
                                    with open(json_file, 'r') as f:
                                        data = json.load(f)
                                    results.append({
                                        'method_type': method_type,
                                        'learning_method': learning_method,
                                        'model_type': model_type,
                                        'seed': seed,
                                        'label_fraction': label_fraction,
                                        'window_size': 30,  # Feature-engineered uses 30s windows
                                        'window_shift': 10,
                                        'auroc': data["test_metrics"].get('auroc', np.nan),
                                        'accuracy': data["test_metrics"].get('accuracy', np.nan),
                                        'pr_auc': data["test_metrics"].get('pr_auc', np.nan),
                                        'f1': data["test_metrics"].get('f1', np.nan),
                                    })
                            else:
                                # ECG methods: look for window_size/window_shift folders
                                window_combinations = [
                                    (10, 5),   # 10s windows, 5s shift
                                    (30, 10),  # 30s windows, 10s shift
                                ]
                                
                                for window_size, window_shift in window_combinations:
                                    json_file = label_folder / str(window_size) / str(window_shift) / "test_results.json"
                                    if json_file.exists():
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

                        except ValueError:
                            # Skip folders that aren't numeric label fractions
                            continue

    return pd.DataFrame(results)


def create_method_labels(df):
    """Create clean method labels for plotting with window size information"""

    def clean_model_name(model_name):
        """Clean up model names - capitalize first letter only"""
        model_map = {
            'cnn': 'CNN',
            'tcn': 'TCN', 
            'transformer': 'Transformer',
            'logistic_regression': 'Logistic Regression',
            'mlp': 'MLP',
            'linear': 'Linear'
        }
        return model_map.get(model_name.lower(), model_name.title())

    def make_label(row):
        clean_model = clean_model_name(row['model_type'])
        window_info = f"{row['window_size']}s"
        
        if row['method_type'] == 'ECG_features':
            return f"Feature-engineered ({clean_model}, {window_info})"
        else:
            return f"{row['learning_method']} ({clean_model}, {window_info})"

    df['method_label'] = df.apply(make_label, axis=1)
    return df


def plot_metric_vs_label_fraction(df, metric="auroc", save_path=None, use_participant_count=False, total_participants=101):
    """Create an excellent plot of AUROC vs Label Fraction
    
    Args:
        df: DataFrame with results
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

    # Define colors and markers for different methods
    method_styles = {
        # Feature-engineered (30s)
        'Feature-engineered (Logistic Regression, 30s)': {'color': '#E69F00', 'marker': 'o', 'linestyle': '-'},
        'Feature-engineered (MLP, 30s)': {'color': '#56B4E9', 'marker': 's', 'linestyle': '--'},
        
        # Supervised methods (10s)
        'Supervised (CNN, 10s)': {'color': '#D55E00', 'marker': '^', 'linestyle': '-'},
        'Supervised (TCN, 10s)': {'color': '#44AA99', 'marker': '>', 'linestyle': '-'},
        'Supervised (Transformer, 10s)': {'color': '#DDCC77', 'marker': 'v', 'linestyle': '-'},
        
        # TSTCC (10s and 30s)
        'TSTCC (Logistic Regression, 10s)': {'color': '#88CCEE', 'marker': 'v', 'linestyle': '-'},
        'TSTCC (Logistic Regression, 30s)': {'color': '#88CCEE', 'marker': 's', 'linestyle': '--'},
        'TSTCC (MLP, 10s)': {'color': '#CC79A7', 'marker': 'p', 'linestyle': '-'},
        'TSTCC (MLP, 30s)': {'color': '#CC79A7', 'marker': 'D', 'linestyle': '--'},
        'TSTCC (Linear, 10s)': {'color': '#28B463', 'marker': 'D', 'linestyle': '-'},
        'TSTCC (Linear, 30s)': {'color': '#28B463', 'marker': '^', 'linestyle': '--'},
        
        # Supervised methods (30s) - in case they exist
        'Supervised (CNN, 30s)': {'color': '#D55E00', 'marker': 'o', 'linestyle': '--'},
        'Supervised (TCN, 30s)': {'color': '#44AA99', 'marker': 's', 'linestyle': '--'}, 
        'Supervised (Transformer, 30s)': {'color': '#DDCC77', 'marker': 'D', 'linestyle': '--'},
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
        ax.plot(x_vals, y_vals,
                color=style['color'],
                marker=style['marker'],
                linestyle=style['linestyle'],
                linewidth=2.5,
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
        ax.set_xlabel('Number of Labeled Training Participants', fontsize=14, fontweight='bold')
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
    # ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.grid(False)
    ax.set_axisbelow(True)

    # Customize legend
    legend = ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True,
                       fontsize=11, title='Methods', title_fontsize=12)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)

    # Add horizontal line at AUROC = 0.5 (random chance)
    if metric == "auroc":
        ax.axhline(y=0.5, color='black', linestyle='-', alpha=0.7, linewidth=2,)
        ax.text(0.98, 0.32, 'Random Chance', fontsize=10, color='black', style='italic', fontweight='bold', ha='right', transform=ax.transAxes)

    elif metric == "pr_auc":
        ax.axhline(y=0.5736, color='black', linestyle='-', alpha=0.7, linewidth=2, )
        ax.text(0.98, 0.42, 'Random Chance', fontsize=10, color='black', style='italic', fontweight='bold',
                ha='right', transform=ax.transAxes)

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
    plot_metric_vs_label_fraction(df, save_path='ecg_auroc_vs_label_fraction.png', use_participant_count=use_participant_count)
    plot_metric_vs_label_fraction(df, metric="pr_auc", save_path='ecg_pr_auc_vs_label_fraction.png', use_participant_count=use_participant_count)

    # Also save the data to CSV for further analysis
    df.to_csv('ecg_results_summary.csv', index=False)
    print("Results saved to 'ecg_results_summary.csv'")

    return df


if __name__ == "__main__":
    results_df = main()