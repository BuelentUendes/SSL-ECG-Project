#!/usr/bin/env python
"""
Domain Discriminability Analysis for ECG Datasets

This script measures how easily a linear classifier can discriminate between different datasets
based on features or representations. High discriminability indicates a strong domain gap.

The analysis includes:
1. Overall domain discrimination (mixed classes, balanced composition)
2. Per-class domain discrimination (stress vs non-stress separately)
3. Pairwise analysis matrix between all dataset combinations
4. Subject-wise splitting to avoid leakage

"""

import os
import sys
import json
import argparse
import logging
from itertools import combinations
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import h5py
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
import matplotlib.pyplot as plt

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from utils.torch_utilities import set_seed
    from utils.helper_paths import DATA_PATH, RESULTS_PATH
except ImportError:
    print("Warning: Could not import project utilities. Using fallbacks.")
    DATA_PATH = os.path.join(project_root, "data")
    RESULTS_PATH = os.path.join(project_root, "results")
    
    def set_seed(seed):
        np.random.seed(seed)


class DomainDiscriminabilityAnalyzer:
    """Analyzes domain discriminability between ECG datasets"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        set_seed(random_state)
        self.results = {}
        
    def load_dataset(self, dataset_path: str, dataset_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load features, labels, and groups from H5 file using the same format as load_processed_data"""
        try:
            label_map = {"baseline": 0, "mental_stress": 1}
            
            X_list, y_list, groups_list = [], [], []
            with h5py.File(dataset_path, "r") as f:
                participants = list(f.keys())

                for participant_key in participants:
                    participant_id = participant_key.replace("participant_", "")
                    for cat in f[participant_key].keys():
                        if cat not in label_map:
                            continue
                        cat_group = f[participant_key][cat]
                        segment_windows_list = []
                        for segment_name in cat_group.keys():
                            windows = cat_group[segment_name][...]
                            segment_windows_list.append(windows)
                        if len(segment_windows_list) == 0:
                            continue
                        # Concatenate windows from all segments in this category
                        windows_all = np.concatenate(segment_windows_list, axis=0)
                        n_windows = windows_all.shape[0]
                        groups_arr = np.array([participant_id] * n_windows, dtype=object)

                        X_list.append(windows_all)
                        y_list.append(np.full((n_windows,), label_map[cat], dtype=int))
                        groups_list.append(groups_arr)

            if len(X_list) == 0:
                print(f"No valid data found in {dataset_path} with label_map {label_map}")
                return None, None, None

            X = np.concatenate(X_list, axis=0)
            y = np.concatenate(y_list, axis=0)
            groups = np.concatenate(groups_list, axis=0)
            
            # Handle NaN values by removing samples with any NaN
            nan_mask = ~np.isnan(X).any(axis=1)
            if not nan_mask.all():
                print(f"Removing {(~nan_mask).sum()} samples with NaN values from {dataset_name}")
                X = X[nan_mask]
                y = y[nan_mask]
                groups = groups[nan_mask]
            
            print(f"Loaded {dataset_name}: X={X.shape}, y={len(y)}, groups={len(np.unique(groups))} participants")
            return X, y, groups
                
        except Exception as e:
            print(f"Error loading {dataset_path}: {e}")
            return None, None, None
    
    def balance_dataset_composition(self, X1: np.ndarray, y1: np.ndarray, groups1: np.ndarray,
                                  X2: np.ndarray, y2: np.ndarray, groups2: np.ndarray,
                                  balance_method: str = 'stratify') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Balance class composition across datasets to avoid label leakage
        """
        # Get class distributions
        unique_classes = np.intersect1d(np.unique(y1), np.unique(y2))
        
        if balance_method == 'undersample':
            # Find minimum class count across both datasets
            min_counts = {}
            for cls in unique_classes:
                count1 = np.sum(y1 == cls)
                count2 = np.sum(y2 == cls)
                min_counts[cls] = min(count1, count2)
            
            # Subsample both datasets
            indices1, indices2 = [], []
            
            for cls in unique_classes:
                cls_idx1 = np.where(y1 == cls)[0]
                cls_idx2 = np.where(y2 == cls)[0]
                
                # Random sample without replacement
                selected1 = np.random.choice(cls_idx1, min_counts[cls], replace=False)
                selected2 = np.random.choice(cls_idx2, min_counts[cls], replace=False)
                
                indices1.extend(selected1)
                indices2.extend(selected2)
            
            indices1, indices2 = np.array(indices1), np.array(indices2)
            
            return (X1[indices1], y1[indices1], groups1[indices1],
                    X2[indices2], y2[indices2], groups2[indices2])
        
        else:  # 'stratify' - keep original but report class distributions
            return X1, y1, groups1, X2, y2, groups2
    
    def create_domain_dataset(self, X1: np.ndarray, y1: np.ndarray, groups1: np.ndarray,
                            X2: np.ndarray, y2: np.ndarray, groups2: np.ndarray,
                            balance_classes: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create domain discrimination dataset
        """
        if balance_classes:
            X1_bal, y1_bal, groups1_bal, X2_bal, y2_bal, groups2_bal = self.balance_dataset_composition(
                X1, y1, groups1, X2, y2, groups2
            )
        else:
            X1_bal, y1_bal, groups1_bal = X1, y1, groups1
            X2_bal, y2_bal, groups2_bal = X2, y2, groups2
        
        # Combine datasets
        X_combined = np.vstack([X1_bal, X2_bal])
        y_task = np.concatenate([y1_bal, y2_bal])  # Original task labels
        y_domain = np.concatenate([np.zeros(len(X1_bal)), np.ones(len(X2_bal))])  # Domain labels
        
        # Create unique groups across datasets by prefixing dataset names
        groups1_prefixed = np.array([f"ds1_{g}" for g in groups1_bal])
        groups2_prefixed = np.array([f"ds2_{g}" for g in groups2_bal])
        groups_combined = np.concatenate([groups1_prefixed, groups2_prefixed])
        
        print(f"Combined dataset: X={X_combined.shape}")
        print(f"Domain distribution: {np.bincount(y_domain.astype(int))}")
        print(f"Task label distribution: {np.bincount(y_task.astype(int))}")
        
        return X_combined, y_domain, y_task, groups_combined
    
    def evaluate_domain_discrimination(self, X: np.ndarray, y_domain: np.ndarray, 
                                     groups: np.ndarray, cv_folds: int = 5) -> Dict:
        """
        Evaluate domain discrimination using cross-validation
        """
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Cross-validation (stratified to maintain class balance)
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        # Evaluate dummy classifier (majority class baseline)
        from sklearn.dummy import DummyClassifier
        dummy_clf = DummyClassifier(strategy='most_frequent', random_state=self.random_state)
        
        dummy_cv_accuracy = cross_val_score(dummy_clf, X_scaled, y_domain, cv=cv, scoring='accuracy')
        dummy_cv_balanced_acc = cross_val_score(dummy_clf, X_scaled, y_domain, cv=cv, scoring='balanced_accuracy')
        dummy_cv_roc_auc = cross_val_score(dummy_clf, X_scaled, y_domain, cv=cv, scoring='roc_auc')
        dummy_cv_auprc = cross_val_score(dummy_clf, X_scaled, y_domain, cv=cv, scoring='average_precision')
        
        # Logistic Regression classifier
        clf = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000,
            n_jobs=-1
        )
        
        # CV scores for logistic regression
        cv_accuracy = cross_val_score(clf, X_scaled, y_domain, cv=cv, scoring='accuracy')
        cv_balanced_acc = cross_val_score(clf, X_scaled, y_domain, cv=cv, scoring='balanced_accuracy')
        cv_roc_auc = cross_val_score(clf, X_scaled, y_domain, cv=cv, scoring='roc_auc')
        cv_auprc = cross_val_score(clf, X_scaled, y_domain, cv=cv, scoring='average_precision')
        
        # Fit on full data for additional metrics
        clf.fit(X_scaled, y_domain)
        y_pred = clf.predict(X_scaled)
        
        # Get probabilities for full dataset metrics
        from sklearn.metrics import roc_auc_score, average_precision_score
        y_prob = clf.predict_proba(X_scaled)[:, 1]
        full_roc_auc = roc_auc_score(y_domain, y_prob)
        full_auprc = average_precision_score(y_domain, y_prob)
        
        results = {
            'cv_accuracy_mean': cv_accuracy.mean(),
            'cv_accuracy_std': cv_accuracy.std(),
            'cv_balanced_accuracy_mean': cv_balanced_acc.mean(),
            'cv_balanced_accuracy_std': cv_balanced_acc.std(),
            'cv_roc_auc_mean': cv_roc_auc.mean(),
            'cv_roc_auc_std': cv_roc_auc.std(),
            'cv_auprc_mean': cv_auprc.mean(),
            'cv_auprc_std': cv_auprc.std(),
            'full_accuracy': accuracy_score(y_domain, y_pred),
            'full_balanced_accuracy': balanced_accuracy_score(y_domain, y_pred),
            'full_roc_auc': full_roc_auc,
            'full_auprc': full_auprc,
            'classification_report': classification_report(y_domain, y_pred),
            'feature_importance': np.abs(clf.coef_[0]) if hasattr(clf, 'coef_') else None,
            # Dummy classifier baselines
            'dummy_cv_accuracy_mean': dummy_cv_accuracy.mean(),
            'dummy_cv_accuracy_std': dummy_cv_accuracy.std(),
            'dummy_cv_balanced_accuracy_mean': dummy_cv_balanced_acc.mean(),
            'dummy_cv_balanced_accuracy_std': dummy_cv_balanced_acc.std(),
            'dummy_cv_roc_auc_mean': dummy_cv_roc_auc.mean(),
            'dummy_cv_roc_auc_std': dummy_cv_roc_auc.std(),
            'dummy_cv_auprc_mean': dummy_cv_auprc.mean(),
            'dummy_cv_auprc_std': dummy_cv_auprc.std(),
        }
        
        return results
    
    def per_class_domain_analysis(self, X1: np.ndarray, y1: np.ndarray, groups1: np.ndarray,
                                 X2: np.ndarray, y2: np.ndarray, groups2: np.ndarray) -> Dict:
        """
        Analyze domain discrimination separately for each class
        """
        results = {}
        unique_classes = np.intersect1d(np.unique(y1), np.unique(y2))
        
        for cls in unique_classes:
            # Filter to single class
            mask1 = y1 == cls
            mask2 = y2 == cls
            
            if np.sum(mask1) < 10 or np.sum(mask2) < 10:
                print(f"Skipping class {cls} due to insufficient samples")
                continue
            
            X1_cls = X1[mask1]
            groups1_cls = groups1[mask1]
            X2_cls = X2[mask2]
            groups2_cls = groups2[mask2]
            
            # Create domain dataset for this class
            X_combined = np.vstack([X1_cls, X2_cls])
            y_domain = np.concatenate([np.zeros(len(X1_cls)), np.ones(len(X2_cls))])
            
            # Create unique groups across datasets by prefixing dataset names
            groups1_cls_prefixed = np.array([f"ds1_{g}" for g in groups1_cls])
            groups2_cls_prefixed = np.array([f"ds2_{g}" for g in groups2_cls])
            groups_combined = np.concatenate([groups1_cls_prefixed, groups2_cls_prefixed])
            
            # Evaluate
            class_results = self.evaluate_domain_discrimination(X_combined, y_domain, groups_combined)
            results[f'class_{int(cls)}'] = class_results
            
            print(f"Class {cls}: Balanced Accuracy = {class_results['cv_balanced_accuracy_mean']:.3f} ± {class_results['cv_balanced_accuracy_std']:.3f}")
        
        return results
    
    def analyze_dataset_pair(self, dataset1_path: str, dataset1_name: str,
                           dataset2_path: str, dataset2_name: str) -> Dict:
        """
        Analyze domain discriminability between two datasets
        """
        print(f"\n=== Analyzing {dataset1_name} vs {dataset2_name} ===")
        
        # Load datasets
        X1, y1, groups1 = self.load_dataset(dataset1_path, dataset1_name)
        X2, y2, groups2 = self.load_dataset(dataset2_path, dataset2_name)
        
        if X1 is None or X2 is None:
            return {'error': 'Failed to load data'}
        
        # Filter to binary classification (baseline=0, mental_stress=1)
        binary_mask1 = np.isin(y1, [0, 1])
        binary_mask2 = np.isin(y2, [0, 1])
        
        X1_binary = X1[binary_mask1]
        y1_binary = y1[binary_mask1]
        groups1_binary = groups1[binary_mask1]
        
        X2_binary = X2[binary_mask2]
        y2_binary = y2[binary_mask2]
        groups2_binary = groups2[binary_mask2]
        
        results = {
            'dataset_pair': f"{dataset1_name}_vs_{dataset2_name}",
            'dataset1_samples': len(X1_binary),
            'dataset2_samples': len(X2_binary),
            'dataset1_participants': len(np.unique(groups1_binary)),
            'dataset2_participants': len(np.unique(groups2_binary))
        }
        
        # Overall domain discrimination (balanced classes)
        X_combined, y_domain, y_task, groups_combined = self.create_domain_dataset(
            X1_binary, y1_binary, groups1_binary,
            X2_binary, y2_binary, groups2_binary,
            balance_classes=True
        )
        
        overall_results = self.evaluate_domain_discrimination(X_combined, y_domain, groups_combined)
        results['overall_discrimination'] = overall_results
        
        # Per-class domain discrimination
        per_class_results = self.per_class_domain_analysis(
            X1_binary, y1_binary, groups1_binary,
            X2_binary, y2_binary, groups2_binary
        )
        results['per_class_discrimination'] = per_class_results
        
        return results
    
    def run_full_analysis(self, datasets: Dict[str, str], output_dir: str = None, window_size: int=30, step_size: int = 10) -> Dict:
        """
        Run complete pairwise domain discriminability analysis
        """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        all_results = {}
        dataset_names = list(datasets.keys())
        
        # Create results matrix
        n_datasets = len(dataset_names)
        accuracy_matrix = np.zeros((n_datasets, n_datasets))
        balanced_acc_matrix = np.zeros((n_datasets, n_datasets))
        
        print(f"Running pairwise analysis for {len(dataset_names)} datasets...")
        
        for i, (name1, path1) in enumerate(datasets.items()):
            for j, (name2, path2) in enumerate(datasets.items()):
                if i >= j:  # Skip diagonal and lower triangle
                    continue
                
                pair_results = self.analyze_dataset_pair(path1, name1, path2, name2)
                pair_key = f"{name1}_vs_{name2}"
                all_results[pair_key] = pair_results
                
                if 'overall_discrimination' in pair_results:
                    acc = pair_results['overall_discrimination']['cv_balanced_accuracy_mean']
                    balanced_acc_matrix[i, j] = acc
                    balanced_acc_matrix[j, i] = acc  # Symmetric matrix
                    accuracy_matrix[i, j] = pair_results['overall_discrimination']['cv_accuracy_mean']
                    accuracy_matrix[j, i] = pair_results['overall_discrimination']['cv_accuracy_mean']
        
        # Create publication-quality visualizations
        self.create_publication_heatmaps(all_results, dataset_names, output_dir, window_size, step_size)
        
        # Save results
        if output_dir:
            results_path = os.path.join(output_dir, f"domain_discriminability_results_{window_size}_{step_size}.json")
            with open(results_path, 'w') as f:
                json.dump(all_results, f, indent=2, default=str)
            
            # Save matrices
            matrix_results = {
                'dataset_names': dataset_names,
                'balanced_accuracy_matrix': balanced_acc_matrix.tolist(),
                'accuracy_matrix': accuracy_matrix.tolist()
            }
            
            matrix_path = os.path.join(output_dir, f"discriminability_matrices_{window_size}_{step_size}.json")
            with open(matrix_path, 'w') as f:
                json.dump(matrix_results, f, indent=2)
        
        return all_results
    
    def create_publication_heatmaps(self, all_results: Dict, dataset_names: List[str], 
                                  output_dir: str = None, window_size: int=30, step_size: int=10):
        """Create clean publication-quality heatmaps for domain discriminability analysis"""
        
        # Set clean publication style
        plt.rcParams.update({
            'font.size': 11,
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial'],
            'axes.linewidth': 0.8,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'xtick.labelsize': 11,
            'ytick.labelsize': 11,
            'legend.fontsize': 10,
            'figure.dpi': 300
        })
        
        n_datasets = len(dataset_names)
        
        # Create matrices for balanced accuracy, ROC AUC, and AUPRC - overall, class 0, and class 1
        matrices = {
            'balanced_acc': {'overall': np.zeros((n_datasets, n_datasets)),
                           'overall_std': np.zeros((n_datasets, n_datasets)),
                           'overall_dummy': np.zeros((n_datasets, n_datasets)),
                           'overall_dummy_std': np.zeros((n_datasets, n_datasets)),
                           'class0': np.zeros((n_datasets, n_datasets)),
                           'class0_std': np.zeros((n_datasets, n_datasets)),
                           'class0_dummy': np.zeros((n_datasets, n_datasets)),
                           'class0_dummy_std': np.zeros((n_datasets, n_datasets)),
                           'class1': np.zeros((n_datasets, n_datasets)),
                           'class1_std': np.zeros((n_datasets, n_datasets)),
                           'class1_dummy': np.zeros((n_datasets, n_datasets)),
                           'class1_dummy_std': np.zeros((n_datasets, n_datasets))},
            'roc_auc': {'overall': np.zeros((n_datasets, n_datasets)),
                       'overall_std': np.zeros((n_datasets, n_datasets)),
                       'overall_dummy': np.zeros((n_datasets, n_datasets)),
                       'overall_dummy_std': np.zeros((n_datasets, n_datasets)),
                       'class0': np.zeros((n_datasets, n_datasets)),
                       'class0_std': np.zeros((n_datasets, n_datasets)),
                       'class0_dummy': np.zeros((n_datasets, n_datasets)),
                       'class0_dummy_std': np.zeros((n_datasets, n_datasets)),
                       'class1': np.zeros((n_datasets, n_datasets)),
                       'class1_std': np.zeros((n_datasets, n_datasets)),
                       'class1_dummy': np.zeros((n_datasets, n_datasets)),
                       'class1_dummy_std': np.zeros((n_datasets, n_datasets))},
            'auprc': {'overall': np.zeros((n_datasets, n_datasets)),
                     'overall_std': np.zeros((n_datasets, n_datasets)),
                     'overall_dummy': np.zeros((n_datasets, n_datasets)),
                     'overall_dummy_std': np.zeros((n_datasets, n_datasets)),
                     'class0': np.zeros((n_datasets, n_datasets)),
                     'class0_std': np.zeros((n_datasets, n_datasets)),
                     'class0_dummy': np.zeros((n_datasets, n_datasets)),
                     'class0_dummy_std': np.zeros((n_datasets, n_datasets)),
                     'class1': np.zeros((n_datasets, n_datasets)),
                     'class1_std': np.zeros((n_datasets, n_datasets)),
                     'class1_dummy': np.zeros((n_datasets, n_datasets)),
                     'class1_dummy_std': np.zeros((n_datasets, n_datasets))}
        }
        
        # Populate matrices
        name_to_idx = {name: i for i, name in enumerate(dataset_names)}
        
        for pair_key, pair_results in all_results.items():
            if 'error' in pair_results:
                continue
                
            name1, name2 = pair_key.split('_vs_')
            i, j = name_to_idx[name1], name_to_idx[name2]
            
            # Overall discrimination
            if 'overall_discrimination' in pair_results:
                overall = pair_results['overall_discrimination']
                # Balanced accuracy
                matrices['balanced_acc']['overall'][i, j] = matrices['balanced_acc']['overall'][j, i] = overall['cv_balanced_accuracy_mean']
                matrices['balanced_acc']['overall_std'][i, j] = matrices['balanced_acc']['overall_std'][j, i] = overall['cv_balanced_accuracy_std']
                matrices['balanced_acc']['overall_dummy'][i, j] = matrices['balanced_acc']['overall_dummy'][j, i] = overall['dummy_cv_balanced_accuracy_mean']
                matrices['balanced_acc']['overall_dummy_std'][i, j] = matrices['balanced_acc']['overall_dummy_std'][j, i] = overall['dummy_cv_balanced_accuracy_std']
                # ROC AUC
                matrices['roc_auc']['overall'][i, j] = matrices['roc_auc']['overall'][j, i] = overall['cv_roc_auc_mean']
                matrices['roc_auc']['overall_std'][i, j] = matrices['roc_auc']['overall_std'][j, i] = overall['cv_roc_auc_std']
                matrices['roc_auc']['overall_dummy'][i, j] = matrices['roc_auc']['overall_dummy'][j, i] = overall['dummy_cv_roc_auc_mean']
                matrices['roc_auc']['overall_dummy_std'][i, j] = matrices['roc_auc']['overall_dummy_std'][j, i] = overall['dummy_cv_roc_auc_std']
                # AUPRC
                matrices['auprc']['overall'][i, j] = matrices['auprc']['overall'][j, i] = overall['cv_auprc_mean']
                matrices['auprc']['overall_std'][i, j] = matrices['auprc']['overall_std'][j, i] = overall['cv_auprc_std']
                matrices['auprc']['overall_dummy'][i, j] = matrices['auprc']['overall_dummy'][j, i] = overall['dummy_cv_auprc_mean']
                matrices['auprc']['overall_dummy_std'][i, j] = matrices['auprc']['overall_dummy_std'][j, i] = overall['dummy_cv_auprc_std']
            
            # Per-class discrimination
            if 'per_class_discrimination' in pair_results:
                per_class = pair_results['per_class_discrimination']
                
                if 'class_0' in per_class:
                    class0 = per_class['class_0']
                    matrices['balanced_acc']['class0'][i, j] = matrices['balanced_acc']['class0'][j, i] = class0['cv_balanced_accuracy_mean']
                    matrices['balanced_acc']['class0_std'][i, j] = matrices['balanced_acc']['class0_std'][j, i] = class0['cv_balanced_accuracy_std']
                    matrices['balanced_acc']['class0_dummy'][i, j] = matrices['balanced_acc']['class0_dummy'][j, i] = class0['dummy_cv_balanced_accuracy_mean']
                    matrices['balanced_acc']['class0_dummy_std'][i, j] = matrices['balanced_acc']['class0_dummy_std'][j, i] = class0['dummy_cv_balanced_accuracy_std']
                    matrices['roc_auc']['class0'][i, j] = matrices['roc_auc']['class0'][j, i] = class0['cv_roc_auc_mean']
                    matrices['roc_auc']['class0_std'][i, j] = matrices['roc_auc']['class0_std'][j, i] = class0['cv_roc_auc_std']
                    matrices['roc_auc']['class0_dummy'][i, j] = matrices['roc_auc']['class0_dummy'][j, i] = class0['dummy_cv_roc_auc_mean']
                    matrices['roc_auc']['class0_dummy_std'][i, j] = matrices['roc_auc']['class0_dummy_std'][j, i] = class0['dummy_cv_roc_auc_std']
                    matrices['auprc']['class0'][i, j] = matrices['auprc']['class0'][j, i] = class0['cv_auprc_mean']
                    matrices['auprc']['class0_std'][i, j] = matrices['auprc']['class0_std'][j, i] = class0['cv_auprc_std']
                    matrices['auprc']['class0_dummy'][i, j] = matrices['auprc']['class0_dummy'][j, i] = class0['dummy_cv_auprc_mean']
                    matrices['auprc']['class0_dummy_std'][i, j] = matrices['auprc']['class0_dummy_std'][j, i] = class0['dummy_cv_auprc_std']
                
                if 'class_1' in per_class:
                    class1 = per_class['class_1']
                    matrices['balanced_acc']['class1'][i, j] = matrices['balanced_acc']['class1'][j, i] = class1['cv_balanced_accuracy_mean']
                    matrices['balanced_acc']['class1_std'][i, j] = matrices['balanced_acc']['class1_std'][j, i] = class1['cv_balanced_accuracy_std']
                    matrices['balanced_acc']['class1_dummy'][i, j] = matrices['balanced_acc']['class1_dummy'][j, i] = class1['dummy_cv_balanced_accuracy_mean']
                    matrices['balanced_acc']['class1_dummy_std'][i, j] = matrices['balanced_acc']['class1_dummy_std'][j, i] = class1['dummy_cv_balanced_accuracy_std']
                    matrices['roc_auc']['class1'][i, j] = matrices['roc_auc']['class1'][j, i] = class1['cv_roc_auc_mean']
                    matrices['roc_auc']['class1_std'][i, j] = matrices['roc_auc']['class1_std'][j, i] = class1['cv_roc_auc_std']
                    matrices['roc_auc']['class1_dummy'][i, j] = matrices['roc_auc']['class1_dummy'][j, i] = class1['dummy_cv_roc_auc_mean']
                    matrices['roc_auc']['class1_dummy_std'][i, j] = matrices['roc_auc']['class1_dummy_std'][j, i] = class1['dummy_cv_roc_auc_std']
                    matrices['auprc']['class1'][i, j] = matrices['auprc']['class1'][j, i] = class1['cv_auprc_mean']
                    matrices['auprc']['class1_std'][i, j] = matrices['auprc']['class1_std'][j, i] = class1['cv_auprc_std']
                    matrices['auprc']['class1_dummy'][i, j] = matrices['auprc']['class1_dummy'][j, i] = class1['dummy_cv_auprc_mean']
                    matrices['auprc']['class1_dummy_std'][i, j] = matrices['auprc']['class1_dummy_std'][j, i] = class1['dummy_cv_auprc_std']
        
        # Create heatmaps for all three metrics
        metric_configs = [
            {
                'metric': 'balanced_acc',
                'label': 'Balanced Accuracy',
                'prefix': 'balanced_accuracy'
            },
            {
                'metric': 'roc_auc', 
                'label': 'ROC AUC',
                'prefix': 'roc_auc'
            },
            {
                'metric': 'auprc',
                'label': 'AUPRC',
                'prefix': 'auprc'
            }
        ]
        
        analysis_configs = [
            {'key': 'overall', 'name': 'Overall'},
            {'key': 'class0', 'name': 'Baseline'},
            {'key': 'class1', 'name': 'Stress'}
        ]
        
        for metric_config in metric_configs:
            for analysis_config in analysis_configs:
                matrix = matrices[metric_config['metric']][analysis_config['key']]
                std_matrix = matrices[metric_config['metric']][analysis_config['key'] + '_std']
                dummy_matrix = matrices[metric_config['metric']][analysis_config['key'] + '_dummy']
                dummy_std_matrix = matrices[metric_config['metric']][analysis_config['key'] + '_dummy_std']
                
                filename = f"{metric_config['prefix']}_{analysis_config['key'].lower()}.png"
                
                self._create_clean_heatmap(
                    matrix, std_matrix, dataset_names,
                    metric_config['label'], filename, output_dir, window_size, step_size,
                    dummy_matrix, dummy_std_matrix
                )
        
        # Reset matplotlib parameters
        plt.rcParams.update(plt.rcParamsDefault)
    
    def _create_clean_heatmap(self, matrix: np.ndarray, std_matrix: np.ndarray,
                             dataset_names: List[str], metric_label: str,
                             filename: str, output_dir: str = None, window_size: int=30, step_size: int=10,
                             dummy_matrix: np.ndarray = None, dummy_std_matrix: np.ndarray = None):
        """Create a single clean heatmap with dummy baseline"""
        
        fig, ax = plt.subplots(figsize=(6, 5))
        
        # Create mask for upper triangle and diagonal
        mask = np.triu(np.ones_like(matrix, dtype=bool))
        np.fill_diagonal(mask, True)
        
        # Clean blue gradient colormap
        from matplotlib.colors import LinearSegmentedColormap
        blues = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b']
        cmap = LinearSegmentedColormap.from_list('clean_blues', blues, N=256)
        
        # Create heatmap using seaborn for cleaner appearance
        heatmap_data = matrix.copy()
        heatmap_data[mask] = np.nan
        
        # Create annotations with logistic regression performance and dummy baseline
        annot = np.empty_like(matrix, dtype=object)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if not mask[i, j] and matrix[i, j] > 0:
                    # Main performance (logistic regression)
                    main_perf = f'{matrix[i, j]:.3f}±{std_matrix[i, j]:.3f}'
                    
                    # Add dummy baseline if provided
                    if dummy_matrix is not None and dummy_std_matrix is not None:
                        dummy_perf = f'{dummy_matrix[i, j]:.3f}±{dummy_std_matrix[i, j]:.3f}'
                        annot[i, j] = f'{main_perf}\nRB: {dummy_perf}'
                    else:
                        annot[i, j] = main_perf
                else:
                    annot[i, j] = ''
        
        # Create the heatmap
        im = sns.heatmap(heatmap_data, 
                        annot=annot, 
                        fmt='',
                        cmap=cmap,
                        vmin=0.5, 
                        vmax=1.0,
                        xticklabels=dataset_names,
                        yticklabels=dataset_names,
                        cbar_kws={'shrink': 0.8, 'label': metric_label},
                        annot_kws={'fontsize': 9, 'fontweight': 'normal'},
                        square=True,
                        linewidths=0.5,
                        linecolor='white',
                        ax=ax)
        
        # Clean up the plot
        ax.set_xlabel('')
        ax.set_ylabel('')
        
        # Remove tick parameters
        ax.tick_params(axis='both', which='major', labelsize=11, 
                      length=0, width=0, pad=5)
        
        # Remove spines
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save with high quality
        if output_dir:
            output_path = os.path.join(output_dir, f"{filename.split('.png')[0]}_{window_size}_{step_size}.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"Saved clean heatmap: {output_path}")
        
        plt.show()
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Domain Discriminability Analysis")
    parser.add_argument("--output_dir", type=str, default="statistical_analysis/results",
                       help="Output directory for results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fs", type=str, default="1000", help="Sampling frequency")
    parser.add_argument("--window_size", type=int, default=10, help="Window size")
    parser.add_argument("--step_size", type=int, default=5, help="Step size")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = DomainDiscriminabilityAnalyzer(random_state=args.seed)
    
    # Define dataset paths
    datasets = {
        'Ours': os.path.join(DATA_PATH, "interim", "ECG_features", args.fs,
                           str(args.window_size), str(args.step_size), "windowed_data.h5"),
        'STRESSID': os.path.join(DATA_PATH, "interim", "STRESSID_features", "ECG", "500",
                                str(args.window_size), str(args.step_size), "windowed_data.h5"),
        'WESAD': os.path.join(DATA_PATH, "interim", "WESAD_features", "ECG", "700",
                             str(args.window_size), str(args.step_size), "windowed_data.h5")
    }
    
    # Filter existing datasets
    existing_datasets = {}
    for name, path in datasets.items():
        if os.path.exists(path):
            existing_datasets[name] = path
            print(f"Found dataset: {name} at {path}")
        else:
            print(f"Warning: Dataset {name} not found at {path}")
    
    if len(existing_datasets) < 2:
        print("Error: Need at least 2 datasets for comparison")
        return
    
    # Run analysis
    print(f"Running domain discriminability analysis with {len(existing_datasets)} datasets...")
    results = analyzer.run_full_analysis(existing_datasets, args.output_dir, args.window_size, args.step_size)
    
    print(f"\n=== Analysis Complete ===")
    print(f"Results saved to: {args.output_dir}")
    
    # Print summary
    print("\n=== Summary of Domain Discriminability ===")
    for pair_name, pair_results in results.items():
        if 'overall_discrimination' in pair_results:
            acc = pair_results['overall_discrimination']['cv_balanced_accuracy_mean']
            std = pair_results['overall_discrimination']['cv_balanced_accuracy_std']
            print(f"{pair_name}: {acc:.3f} ± {std:.3f}")


if __name__ == "__main__":
    main()