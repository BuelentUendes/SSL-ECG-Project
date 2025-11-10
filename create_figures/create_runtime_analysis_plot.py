import matplotlib.pyplot as plt
from utils.helper_paths import RESULTS_PATH, FIGURES_PATH
from utils.torch_utilities import create_directory
import pandas as pd
import numpy as np
import json
import os


def load_ssl_runtime_data(results_base_path="results/ECG"):
    """Load actual runtime and memory data from JSON files.
    
    Args:
        results_base_path: Base path to the results directory

    Returns:
        dict: Dictionary containing averaged metrics for each method
    """
    
    # Available methods and their common seeds
    methods_seeds = {
        "TSTCC": [3, 5, 7, 9, 42],
        "TSTCC_S3": [3, 5, 7, 9, 42], 
        "SimCLR": [3, 5, 7, 9, 42],
        "SimCLR_S3": [3, 5, 7, 9, 42],
        "TS2Vec": [3, 5, 7, 9, 42],
        "TS2Vec_S3": [3, 5, 7, 9, 42],
        "InfoTS": [3, 5, 7, 9, 42],
        "InfoTS_S3": [3, 5, 7, 9, 42]
    }
    
    # Convert methods to match the expected naming convention
    method_display_names = {
        "TSTCC": "TSTCC",
        "TSTCC_S3": "TSTCC+S3", 
        "SimCLR": "SimCLR",
        "SimCLR_S3": "SimCLR+S3",
        "TS2Vec": "TS2Vec",
        "TS2Vec_S3": "TS2Vec+S3",
        "InfoTS": "InfoTS",
        "InfoTS_S3": "InfoTS+S3"
    }
    
    ssl_results = {}
    
    for method, seeds in methods_seeds.items():
        method_data = []
        
        for seed in seeds:
            # Construct the path to the JSON files
            base_path = os.path.join(RESULTS_PATH, "ECG", method, "logistic_regression", str(seed), "0.1", "10", "5", "1.0")

            memory_file = os.path.join(base_path, "peak_memory_consumption_epochs.json")
            runtime_file = os.path.join(base_path, "runtime_per_epoch.json")
            
            if os.path.exists(memory_file) and os.path.exists(runtime_file):
                try:
                    # Load memory data
                    with open(memory_file, 'r') as f:
                        memory_data = json.load(f)
                    
                    # Load runtime data
                    with open(runtime_file, 'r') as f:
                        runtime_data = json.load(f)
                    
                    # Calculate averages across epochs for this seed
                    memory_values = list(memory_data.values())
                    runtime_values = list(runtime_data.values())
                    
                    avg_memory = np.mean(memory_values)
                    avg_runtime = np.mean(runtime_values)
                    
                    method_data.append({
                        'seed': seed,
                        'avg_memory': avg_memory,
                        'avg_runtime': avg_runtime
                    })
                    
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Warning: Could not load data for {method} seed {seed}: {e}")
                    continue
            else:
                print(f"Warning: Files not found for {method} seed {seed}")
                continue
        
        if method_data:
            # Average across all seeds for this method
            avg_memory_across_seeds = np.mean([d['avg_memory'] for d in method_data])
            avg_runtime_across_seeds = np.mean([d['avg_runtime'] for d in method_data])
            
            display_name = method_display_names[method]
            ssl_results[display_name] = {
                'avg_memory': avg_memory_across_seeds,
                'avg_runtime': avg_runtime_across_seeds,
                'num_seeds': len(method_data)
            }
        else:
            print(f"Warning: No data found for method {method}")
    
    return ssl_results


def create_ssl_runtime_plot(legend_style='in_labels', save_path=None, figsize=(10, 8),
                            results_base_path="results/ECG", label_fraction=0.1):
    """Create SSL runtime analysis plot showing AUROC vs Runtime with memory consumption as bubble size.

    Args:
        legend_style: str, Legend display style options:
            - 'in_labels': Include memory values in method labels (recommended)
            - 'below': Show legend below the plot
            - 'side': Show legend on the right side
            - 'none': No legend, just note about bubble sizes
        save_path: str, Path to save the plot (without extension, saves both PDF and PNG)
        figsize: tuple, Figure size (width, height)
        results_base_path: str, Base path to the results directory
        label_fraction: float, What label fraction to plot

    Returns:
        fig, ax: matplotlib figure and axes objects
    """

    # Load actual data from JSON files
    print(f"Loading runtime and memory data from {results_base_path}...")
    ssl_data = load_ssl_runtime_data(results_base_path)
    
    if not ssl_data:
        raise ValueError("No data could be loaded. Please check the results directory path.")
    
    print(f"Successfully loaded data for {len(ssl_data)} methods")

    auroc_values = {}
    for method in ssl_data.keys():
        method_key = method.replace("+S3", "_S3")  # Convert display name back to directory name
        auroc_list = []
        
        # Try to load from multiple seeds
        seeds = [3, 5, 7, 9, 42]
        for seed in seeds:
            results_file = os.path.join(
                RESULTS_PATH, "ECG", method_key, "logistic_regression", str(seed),
                str(label_fraction), "10", "5", "1.0", "test_results.json"
            )

            if os.path.exists(results_file):
                try:
                    with open(results_file, 'r') as f:
                        test_results = json.load(f)
                    if 'test_metrics' in test_results:
                        auroc_list.append(test_results['test_metrics']['auroc'])
                except (json.JSONDecodeError, KeyError):
                    continue
        
        if auroc_list:
            auroc_values[method] = np.mean(auroc_list)
        else:
            # Fallback values from performance table
            raise ValueError("We could not find the values!")

    # Create method averages dataframe
    method_data = []
    for method, data in ssl_data.items():
        method_data.append({
            'method': method,
            'avg_memory': data['avg_memory'],
            'avg_runtime': data['avg_runtime'],
            'auroc': auroc_values.get(method, 50.0),  # Default AUROC if not found
            'num_seeds': data['num_seeds']
        })

    method_averages = pd.DataFrame(method_data)

    # Adjust figure size based on legend style
    if legend_style == 'below':
        figsize = (figsize[0], figsize[1] + 1)
    elif legend_style == 'side':
        figsize = (figsize[0] + 2, figsize[1])

    # Set up the plotting style and create figure
    with plt.style.context(['ieee']):
        fig, ax = plt.subplots(figsize=figsize)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

        # Define colors for each method (matching provided color scheme)
        method_colors = {
            'TSTCC': '#0E9FEB',  # Light blue
            'TSTCC+S3': '#005377',  # Darker blue
            'SimCLR': '#DBABE0',  # Light purple/pink
            'SimCLR+S3': '#78248E',  # Darker purple
            'TS2Vec': '#D2B48C',  # Light tan/beige
            'TS2Vec+S3': '#944D0F',  # Darker brown
            'InfoTS': '#EBB952',  # Light orange/gold
            'InfoTS+S3': '#F08D00'  # Darker orange
        }

        # Create scatter plot with bubble sizes proportional to memory consumption
        for _, row in method_averages.iterrows():
            method = row['method']
            x = row['avg_runtime']
            y = row['auroc']

            # Use exponential scaling to make memory differences more pronounced
            size = (row['avg_memory'] ** 1.8) * 80
            print(f"{method}: Runtime: {x}, size: {size} and performance: {y}")

            color = method_colors.get(method, '#000000')

            ax.scatter(x, y, s=size, c=color, alpha=0.7)

        # Add method labels next to points
        for _, row in method_averages.iterrows():
            method = row['method']
            x = row['avg_runtime']
            y = row['auroc']
            memory = row['avg_memory']

            color = method_colors.get(method, '#000000')

            # Create label based on legend style
            if legend_style == 'in_labels':
                label_text = f"{method} [{memory:.2f} GB]"
            else:
                label_text = method

            # Calculate dynamic offset based on method position to avoid overlap
            # Use larger offsets and vary position based on method
            offset_map = {
                'TSTCC': (25, 25),
                'TSTCC+S3': (25, -25),
                'SimCLR': (25, 25), 
                'SimCLR+S3': (25, -30),
                'TS2Vec': (-120, 25),
                'TS2Vec+S3': (-120, -30),
                'InfoTS': (-100, 50),
                'InfoTS+S3': (-100, -60)
            }
            
            offset_x, offset_y = offset_map.get(method, (25, 25))
            
            ax.annotate(label_text, (x, y),
                        xytext=(offset_x, offset_y), textcoords='offset points',
                        fontsize=20, fontweight='bold',
                        color=color)

        # Customize the plot
        ax.set_xlabel('Runtime per Epoch (seconds)', fontsize=20)
        ax.set_ylabel('AUROC', fontsize=20)
        ax.set_ylim(0.45, 0.80)  # Set for decimal AUROC values (0.0 to 1.0)

        # Set tick label font sizes
        ax.tick_params(axis='both', which='major', labelsize=20)

        # Remove top and right spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)

        # Set grid to be behind other elements
        ax.set_axisbelow(True)

        # Handle legend based on style
        size_legend = None
        if legend_style in ['below', 'side']:
            memory_values = [2.9, 3.6, 4.6]
            legend_elements = []

            for mem_val in memory_values:
                size = (mem_val ** 1.8) * 80
                legend_elements.append(
                    plt.scatter([], [], s=size, c='gray', alpha=0.7,
                                label=f'{mem_val:.2f} GB'))

            if legend_style == 'below':
                size_legend = ax.legend(handles=legend_elements,
                                        loc='upper center',
                                        bbox_to_anchor=(0.5, -0.08),
                                        ncol=3,
                                        title='Peak GPU Memory (GB)',
                                        fontsize=12,
                                        title_fontsize=12,
                                        frameon=False,
                                        fancybox=False,
                                        shadow=False,
                                        columnspacing=3.0,
                                        handletextpad=1.5)
            else:  # side
                size_legend = ax.legend(handles=legend_elements,
                                        loc='center left',
                                        bbox_to_anchor=(1.05, 0.5),
                                        title='Peak GPU Memory (GB)',
                                        fontsize=12,
                                        title_fontsize=12,
                                        frameon=False,
                                        fancybox=False,
                                        shadow=False)

        elif legend_style == 'none':
            # Add a note about bubble sizes
            ax.text(0.02, 0.98, 'Bubble size ∝ Peak GPU Memory',
                    transform=ax.transAxes, fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        # Adjust layout
        if legend_style == 'below':
            plt.subplots_adjust(bottom=0.2)
        else:
            plt.tight_layout()

        # Save the plots
        save_name_pdf = 'ssl_runtime_analysis.pdf'
        save_name_png = 'ssl_runtime_analysis.png'
        if save_path:
            # Save with proper bbox handling for external legends
            save_kwargs = {'dpi': 500, 'bbox_inches': 'tight', 'facecolor': 'white'}
            if size_legend is not None:
                save_kwargs['bbox_extra_artists'] = (size_legend,)

            plt.savefig(f'{os.path.join(save_path, save_name_pdf)}', **save_kwargs)
            plt.savefig(f'{os.path.join(save_path, save_name_png)}', **save_kwargs)
            print(f"Plot saved to {save_path}.pdf and {save_path}.png")
        else:
            # Default save paths
            save_kwargs = {'dpi': 500, 'bbox_inches': 'tight', 'facecolor': 'white'}
            if size_legend is not None:
                save_kwargs['bbox_extra_artists'] = (size_legend,)

            plt.savefig(save_name_pdf, **save_kwargs)
            plt.savefig(save_name_png, **save_kwargs)
            print("Plot saved as ssl_runtime_analysis.pdf and ssl_runtime_analysis.png")

        print(f"Legend style: {legend_style}")
        print("\nData summary:")
        for _, row in method_averages.iterrows():
            print(f"{row['method']}: Runtime={row['avg_runtime']:.2f}s, "
                  f"Memory={row['avg_memory']:.2f} GB, AUROC={row['auroc']:.3f} (from {row['num_seeds']} seeds)")

        plt.show()

    return fig, ax


# Example usage
if __name__ == "__main__":
    # Create the plot with memory values in labels (recommended)

    # Create figures dictionary (if not yet created)
    create_directory(FIGURES_PATH)
    fig, ax = create_ssl_runtime_plot(legend_style='in_labels', save_path=FIGURES_PATH)



