#!/usr/bin/env python3
"""
ECG Data Visualization with UMAP
=================================

This script visualizes ECG data from three datasets (ECG, WESAD, STRESSID) using UMAP dimensionality reduction.
"""

import os
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP
from typing import Dict, List, Tuple, Optional
import argparse
import warnings

from utils.helper_paths import DATA_PATH, FIGURES_PATH
from utils.torch_utilities import create_directory
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ECGDataLoader:
    """Class to load and preprocess ECG data from different datasets."""
    
    def __init__(self):
        self.dataset_configs = {
            'Source [van der Mee et al. (2021)]': {
                'path': os.path.join(DATA_PATH, 'interim/ECG/500/10/5/windowed_data.h5'),
                'color': '#57B4BA',  # Blue
                'sampling_rate': 500
            },
            'WESAD': {
                'path': os.path.join(DATA_PATH, 'interim/WESAD/ECG/500/10/5/windowed_data.h5'),
                'color': '#015551',  # Orange
                'sampling_rate': 500
            },
            'StressID': {
                'path': os.path.join(DATA_PATH, 'interim/STRESSID/ECG/500/10/5/windowed_data.h5'),
                'color': '#FE4F2D',  # Green
                'sampling_rate': 500
            }
        }
    
    def load_dataset(self, dataset_name: str, focus_mental_stress: bool = True, 
                    samples_per_participant: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load ECG data from a specific dataset.
        
        Args:
            dataset_name: Name of the dataset ('ECG', 'WESAD', 'STRESSID')
            focus_mental_stress: If True, only load mental_stress=1 and baseline=0 data
            samples_per_participant: Number of samples to take per participant (None for all)
            
        Returns:
            Tuple of (data, labels, participant_ids)
        """
        config = self.dataset_configs[dataset_name]
        filepath = config['path']
        
        all_data = []
        all_labels = []
        all_participant_ids = []
        
        print(f"Loading {dataset_name} dataset from {filepath}")
        
        with h5py.File(filepath, 'r') as f:
            participants = list(f.keys())
            print(f"Found {len(participants)} participants")
            
            for participant_id in participants:
                participant_group = f[participant_id]
                participant_data = []
                participant_labels = []
                
                # Handle different dataset structures
                if focus_mental_stress:
                    # Load mental_stress (label=1) and baseline (label=0)
                    conditions = ['mental_stress', 'baseline']
                    condition_labels = [1, 0]
                else:
                    # Load all available conditions
                    conditions = list(participant_group.keys())
                    condition_labels = list(range(len(conditions)))
                
                for condition, label in zip(conditions, condition_labels):
                    if condition in participant_group:
                        condition_group = participant_group[condition]
                        
                        # Iterate through segments
                        for segment_name in condition_group.keys():
                            segment_data = condition_group[segment_name][:]
                            
                            # segment_data shape: (n_windows, sequence_length)
                            for window in segment_data:
                                participant_data.append(window)
                                participant_labels.append(label)
                
                # Subsample if requested
                if samples_per_participant is not None and len(participant_data) > samples_per_participant:
                    indices = np.random.choice(len(participant_data), samples_per_participant, replace=False)
                    participant_data = [participant_data[i] for i in indices]
                    participant_labels = [participant_labels[i] for i in indices]
                
                # Add to global lists
                all_data.extend(participant_data)
                all_labels.extend(participant_labels)
                all_participant_ids.extend([participant_id] * len(participant_data))
        
        print(f"Loaded {len(all_data)} samples from {dataset_name}")
        return np.array(all_data), np.array(all_labels), all_participant_ids

    def load_all_datasets(self, focus_mental_stress: bool = True, 
                         samples_per_participant: Optional[int] = None) -> Dict:
        """Load data from all three datasets."""
        all_datasets = {}
        
        for dataset_name in self.dataset_configs.keys():
            try:
                data, labels, participant_ids = self.load_dataset(
                    dataset_name, focus_mental_stress, samples_per_participant
                )
                all_datasets[dataset_name] = {
                    'data': data,
                    'labels': labels,
                    'participant_ids': participant_ids,
                    'config': self.dataset_configs[dataset_name]
                }
            except Exception as e:
                print(f"Error loading {dataset_name}: {e}")
                continue
        
        return all_datasets

class ECGVisualizer:
    """Class for creating ECG visualizations."""
    
    def __init__(self, datasets: Dict):
        self.datasets = datasets
        
    def plot_raw_ecg_samples(self, n_samples: int = 3, figsize: Tuple[int, int] = (15, 10)):
        """Plot raw ECG signals from each dataset."""
        fig, axes = plt.subplots(len(self.datasets), n_samples, figsize=figsize)
        if len(self.datasets) == 1:
            axes = axes.reshape(1, -1)
        
        for i, (dataset_name, dataset_info) in enumerate(self.datasets.items()):
            data = dataset_info['data']
            labels = dataset_info['labels']
            color = dataset_info['config']['color']
            sampling_rate = dataset_info['config']['sampling_rate']
            
            # Sample some signals
            stress_indices = np.where(labels == 1)[0]
            baseline_indices = np.where(labels == 0)[0]
            
            sample_indices = []
            if len(stress_indices) > 0:
                sample_indices.extend(np.random.choice(stress_indices, min(n_samples//2, len(stress_indices)), replace=False))
            if len(baseline_indices) > 0:
                sample_indices.extend(np.random.choice(baseline_indices, min(n_samples//2, len(baseline_indices)), replace=False))
            
            # Fill remaining samples if needed
            while len(sample_indices) < n_samples and len(sample_indices) < len(data):
                remaining_indices = set(range(len(data))) - set(sample_indices)
                if remaining_indices:
                    sample_indices.append(np.random.choice(list(remaining_indices)))
                else:
                    break
            
            for j, idx in enumerate(sample_indices[:n_samples]):
                signal = data[idx]
                label = labels[idx]
                time_axis = np.arange(len(signal)) / sampling_rate
                
                axes[i, j].plot(time_axis, signal, color=color, linewidth=0.8)
                axes[i, j].set_title(f'{dataset_name} - {"Stress" if label == 1 else "Baseline"}')
                axes[i, j].set_xlabel('Time (s)')
                axes[i, j].set_ylabel('ECG Amplitude')
                axes[i, j].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle('Raw ECG Signals from Different Datasets', y=1.02, fontsize=16)
        return fig
    
    def create_umap_embedding(self, n_components: int = 2, n_neighbors: int = 15, 
                             min_dist: float = 0.1, metric: str = 'euclidean',
                             random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create UMAP embedding of all datasets."""
        print("Creating UMAP embedding...")
        
        # Find minimum sequence length across all datasets
        min_length = float('inf')
        for dataset_name, dataset_info in self.datasets.items():
            data = dataset_info['data']
            current_length = data.shape[1]
            min_length = min(min_length, current_length)
            print(f"{dataset_name}: {data.shape[0]} samples, {current_length} time points")
        
        print(f"Using minimum length: {min_length}")
        
        # Combine all data with consistent length
        all_data = []
        dataset_labels = []
        stress_labels = []
        
        for dataset_name, dataset_info in self.datasets.items():
            data = dataset_info['data']
            labels = dataset_info['labels']
            
            # Truncate to minimum length
            truncated_data = data[:, :min_length]
            all_data.append(truncated_data)
            dataset_labels.extend([dataset_name] * len(data))
            stress_labels.extend(labels)
        
        combined_data = np.vstack(all_data)
        print(f"Combined data shape: {combined_data.shape}")
        
        # Create UMAP embedding
        umap_reducer = UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
            verbose=True
        )
        
        embedding = umap_reducer.fit_transform(combined_data)
        print(f"UMAP embedding shape: {embedding.shape}")
        
        return embedding, np.array(dataset_labels), np.array(stress_labels)
    
    def plot_umap_results(self, embedding: np.ndarray, dataset_labels: np.ndarray, 
                         stress_labels: np.ndarray, figsize: Tuple[int, int] = (10, 8)):
        """Plot UMAP results with different visualizations."""

        with plt.style.context(['ieee']):
            fig, axes = plt.subplots(1, 1, figsize=figsize)
            # plt.style.use('default')
            axes.set_facecolor('white')
            fig.patch.set_facecolor('white')
            
            # Set axis colors to black
            axes.tick_params(colors='black')
            axes.xaxis.label.set_color('black')
            axes.yaxis.label.set_color('black')
            for spine in axes.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(1.5)

            # Plot 1: Color by dataset
            unique_datasets = np.unique(dataset_labels)
            colors = [self.datasets[ds]['config']['color'] for ds in unique_datasets]

            for i, dataset in enumerate(unique_datasets):
                mask = dataset_labels == dataset
                axes.scatter(embedding[mask, 0], embedding[mask, 1],
                              c=colors[i], label=dataset, alpha=0.7, s=8)

            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            axes.set_xlabel('UMAP 1', fontsize=20)
            axes.set_ylabel('UMAP 2', fontsize=20)
            legend = axes.legend(loc='upper left', frameon=True, fancybox=True, shadow=False,
                      fontsize=20, markerscale=4)
            legend.get_frame().set_facecolor('white')
            axes.grid(True, alpha=0.7)

            # Improve overall appearance
            axes.spines['top'].set_visible(False)
            axes.spines['right'].set_visible(False)
            axes.spines['left'].set_linewidth(1.5)
            axes.spines['bottom'].set_linewidth(1.5)
            plt.tight_layout()
        return fig

def main():
    """Main function to run the UMAP visualization."""
    parser = argparse.ArgumentParser(description='Visualize ECG data using UMAP')
    parser.add_argument('--samples-per-participant', type=int, default=100,
                       help='Number of samples to take per participant (None for all)')
    parser.add_argument('--focus-mental-stress', action='store_true', default=True,
                       help='Focus only on mental stress (1) and baseline (0) labels')
    parser.add_argument('--n-neighbors', type=int, default=15, #15 is good
                       help='UMAP n_neighbors parameter')
    parser.add_argument('--min-dist', type=float, default=0.1,
                       help='UMAP min_dist parameter')
    parser.add_argument('--metric', type=str, default='euclidean',
                       help='UMAP distance metric')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--save-plots', action='store_true', default=True,
                       help='Save plots to files')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.random_seed)

    print("ECG Data Visualization with UMAP")
    print("=" * 40)
    
    # Load data
    loader = ECGDataLoader()
    datasets = loader.load_all_datasets(
        focus_mental_stress=args.focus_mental_stress,
        samples_per_participant=args.samples_per_participant
    )
    create_directory(FIGURES_PATH)
    
    if not datasets:
        print("No datasets loaded successfully!")
        return
    
    print(f"Loaded {len(datasets)} datasets")
    for name, info in datasets.items():
        print(f"  {name}: {len(info['data'])} samples")
    
    # Create visualizer
    visualizer = ECGVisualizer(datasets)
    
    # Plot raw ECG samples
    print("\nCreating raw ECG plots...")
    fig_raw = visualizer.plot_raw_ecg_samples()
    if args.save_plots:
        fig_raw.savefig(os.path.join(FIGURES_PATH,'ecg_raw_signals.png'), dpi=500, bbox_inches='tight')
        print("Saved raw ECG plot to ecg_raw_signals.png")
    
    # Create UMAP embedding
    embedding, dataset_labels, stress_labels = visualizer.create_umap_embedding(
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        random_state=args.random_seed
    )
    
    # Plot UMAP results
    print("\nCreating UMAP plots...")
    fig_umap = visualizer.plot_umap_results(embedding, dataset_labels, stress_labels)
    if args.save_plots:
        fig_umap.savefig(os.path.join(FIGURES_PATH, 'ecg_umap_embedding.png'), dpi=500, bbox_inches='tight')
        print("Saved UMAP plot to ecg_umap_embedding.png")
    
    print("\nVisualization complete!")
    # Clean up figures
    plt.close('all')

if __name__ == '__main__':
    main()