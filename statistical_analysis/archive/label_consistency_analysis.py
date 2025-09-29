#!/usr/bin/env python
"""
Domain Discriminability Analysis for ECG Datasets

This script measures how easily a linear classifier can discriminate between different datasets
based on features or representations. High discriminability indicates a strong domain gap.

The analysis includes:
1. Overall domain discrimination (mixed classes, balanced composition)
2. Per-class domain discrimination (stress vs non-stress separately)
3. Pairwise analysis matrix between all dataset combinations
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import h5py
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from utils.torch_utilities import set_seed, split_indices_by_participant_groups, get_participant_cv_splitter
    from utils.helper_paths import DATA_PATH, RESULTS_PATH
except ImportError:
    print("Warning: Could not import project utilities. Using fallbacks.")
    DATA_PATH = os.path.join(project_root, "data")
    RESULTS_PATH = os.path.join(project_root, "results")
    
    def set_seed(seed):
        np.random.seed(seed)
    
    def split_indices_by_participant_groups(groups, train_ratio=0.8, label_fraction=1.0, seed=42):
        """Fallback implementation for participant-based splitting"""
        np.random.seed(seed)
        unique_participants = np.unique(groups)
        n_train = int(len(unique_participants) * train_ratio)
        
        train_participants = np.random.choice(unique_participants, n_train, replace=False)
        test_participants = np.setdiff1d(unique_participants, train_participants)
        
        train_idx = np.isin(groups, train_participants)
        test_idx = np.isin(groups, test_participants)
        
        return np.where(train_idx)[0], train_participants, np.where(test_idx)[0], test_participants
    
    def get_participant_cv_splitter(groups, min_participants_for_kfold=5, k=5):
        """Fallback implementation for CV splitter"""
        from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
        unique_participants = np.unique(groups)
        
        if len(unique_participants) >= min_participants_for_kfold:
            cv_splitter = GroupKFold(n_splits=min(k, len(unique_participants)))
            n_splits = min(k, len(unique_participants))
        else:
            cv_splitter = LeaveOneGroupOut()
            n_splits = len(unique_participants)
        
        return cv_splitter, n_splits


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
        print(f"Domain distribution: {np.bincount(y_domain.astype(int))}. First entry is dataset 0, second one is dataset 1")
        
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
    
    def evaluate_domain_discrimination_with_holdout(self, X: np.ndarray, y_domain: np.ndarray, 
                                                   groups: np.ndarray, train_ratio: float = 0.8, 
                                                   cv_folds: int = 5, min_participants_for_kfold: int = 5) -> Dict:
        """
        Evaluate domain discrimination using 80/20 train-test split with participant-based CV on training data
        
        This method follows the pattern from train_simple_classifiers.py:
        1. Split data by participants into 80% train, 20% test
        2. Perform CV on the 80% training data for model selection
        3. Fit scaler on training data and apply to test data
        4. Train final model on full training data and evaluate on test data
        """
        # Split by participants into train/test
        train_idx, train_participants, test_idx, test_participants = split_indices_by_participant_groups(
            groups, train_ratio=train_ratio, label_fraction=1.0, seed=self.random_state
        )
        
        X_train = X[train_idx]
        y_train = y_domain[train_idx] 
        groups_train = groups[train_idx]
        
        X_test = X[test_idx]
        y_test = y_domain[test_idx]
        
        print(f"Holdout split: Train={len(X_train)} samples from {len(train_participants)} participants, "
              f"Test={len(X_test)} samples from {len(test_participants)} participants")
        
        # Set up CV splitter for training data
        cv_splitter, n_splits = get_participant_cv_splitter(
            groups_train, min_participants_for_kfold=min_participants_for_kfold, k=cv_folds
        )
        
        # Standardize features - fit on train, apply on test
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Cross-validation on training data for model selection
        clf = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000,
            n_jobs=-1
        )
        
        # CV scores on training data
        cv_accuracy = cross_val_score(clf, X_train_scaled, y_train, cv=cv_splitter, scoring='accuracy', groups=groups_train)
        cv_balanced_acc = cross_val_score(clf, X_train_scaled, y_train, cv=cv_splitter, scoring='balanced_accuracy', groups=groups_train)
        cv_roc_auc = cross_val_score(clf, X_train_scaled, y_train, cv=cv_splitter, scoring='roc_auc', groups=groups_train)
        cv_auprc = cross_val_score(clf, X_train_scaled, y_train, cv=cv_splitter, scoring='average_precision', groups=groups_train)
        
        # Fit final model on full training data
        clf.fit(X_train_scaled, y_train)
        
        # Evaluate on test data
        y_test_pred = clf.predict(X_test_scaled)
        y_test_prob = clf.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate test metrics
        from sklearn.metrics import roc_auc_score, average_precision_score
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_balanced_accuracy = balanced_accuracy_score(y_test, y_test_pred)
        test_roc_auc = roc_auc_score(y_test, y_test_prob)
        test_auprc = average_precision_score(y_test, y_test_prob)
        
        # Dummy classifier baseline on test data
        from sklearn.dummy import DummyClassifier
        dummy_clf = DummyClassifier(strategy='most_frequent', random_state=self.random_state)
        dummy_clf.fit(X_train_scaled, y_train)
        dummy_test_pred = dummy_clf.predict(X_test_scaled)
        dummy_test_prob = dummy_clf.predict_proba(X_test_scaled)[:, 1]
        
        dummy_test_accuracy = accuracy_score(y_test, dummy_test_pred)
        dummy_test_balanced_accuracy = balanced_accuracy_score(y_test, dummy_test_pred)
        dummy_test_roc_auc = roc_auc_score(y_test, dummy_test_prob)
        dummy_test_auprc = average_precision_score(y_test, dummy_test_prob)
        
        results = {
            # CV results on training data
            'cv_accuracy_mean': cv_accuracy.mean(),
            'cv_accuracy_std': cv_accuracy.std(),
            'cv_balanced_accuracy_mean': cv_balanced_acc.mean(),
            'cv_balanced_accuracy_std': cv_balanced_acc.std(),
            'cv_roc_auc_mean': cv_roc_auc.mean(),
            'cv_roc_auc_std': cv_roc_auc.std(),
            'cv_auprc_mean': cv_auprc.mean(),
            'cv_auprc_std': cv_auprc.std(),
            'n_cv_splits': n_splits,
            
            # Test set results
            'test_accuracy': test_accuracy,
            'test_balanced_accuracy': test_balanced_accuracy,
            'test_roc_auc': test_roc_auc,
            'test_auprc': test_auprc,
            
            # Dummy baseline test results
            'dummy_test_accuracy': dummy_test_accuracy,
            'dummy_test_balanced_accuracy': dummy_test_balanced_accuracy,
            'dummy_test_roc_auc': dummy_test_roc_auc,
            'dummy_test_auprc': dummy_test_auprc,
            
            # Split information
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'train_participants': len(train_participants),
            'test_participants': len(test_participants),
            
            'classification_report': classification_report(y_test, y_test_pred),
            'feature_importance': np.abs(clf.coef_[0]) if hasattr(clf, 'coef_') else None,
        }
        
        return results
    
    def analyze_dataset_pair(self, dataset1_path: str, dataset1_name: str,
                           dataset2_path: str, dataset2_name: str, use_holdout: bool = False) -> Dict:
        """
        Analyze domain discriminability between two datasets
        
        Args:
            dataset1_path: Path to first dataset
            dataset1_name: Name of first dataset  
            dataset2_path: Path to second dataset
            dataset2_name: Name of second dataset
            use_holdout: If True, use 80/20 holdout split with participant-based CV, otherwise use regular CV
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
            'dataset2_participants': len(np.unique(groups2_binary)),
            'evaluation_method': 'holdout_80_20' if use_holdout else 'cross_validation'
        }
        
        # Overall domain discrimination (balanced classes)
        X_combined, y_domain, y_task, groups_combined = self.create_domain_dataset(
            X1_binary, y1_binary, groups1_binary,
            X2_binary, y2_binary, groups2_binary,
            balance_classes=False,
        )
        
        if use_holdout:
            # Use 80/20 holdout evaluation with participant-based CV on training data
            overall_results = self.evaluate_domain_discrimination_with_holdout(
                X_combined, y_domain, groups_combined
            )
        else:
            # Use regular cross-validation
            overall_results = self.evaluate_domain_discrimination(X_combined, y_domain, groups_combined)
        
        results['overall_discrimination'] = overall_results
        
        # Per-class domain discrimination (only for regular CV for now to keep it manageable)
        if not use_holdout:
            per_class_results = self.per_class_domain_analysis(
                X1_binary, y1_binary, groups1_binary,
                X2_binary, y2_binary, groups2_binary
            )
            results['per_class_discrimination'] = per_class_results
        
        return results
    
    def run_full_analysis(
            self, datasets: Dict[str, str], output_dir: str = None, window_size: int=30, step_size: int = 10, use_holdout: bool = False
    ) -> Dict:
        """
        Run complete pairwise domain discriminability analysis
        
        Args:
            datasets: Dictionary mapping dataset names to file paths
            output_dir: Output directory for results
            window_size: Window size parameter
            step_size: Step size parameter  
            use_holdout: If True, use 80/20 holdout split with participant-based CV
        """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        all_results = {}
        dataset_names = list(datasets.keys())
        
        # Create results matrix
        n_datasets = len(dataset_names)
        accuracy_matrix = np.zeros((n_datasets, n_datasets))
        balanced_acc_matrix = np.zeros((n_datasets, n_datasets))
        
        evaluation_method = "holdout_80_20" if use_holdout else "cross_validation"
        print(f"Running pairwise analysis for {len(dataset_names)} datasets using {evaluation_method}...")
        
        for i, (name1, path1) in enumerate(datasets.items()):
            for j, (name2, path2) in enumerate(datasets.items()):
                if i >= j:  # Skip diagonal and lower triangle
                    continue
                
                pair_results = self.analyze_dataset_pair(path1, name1, path2, name2, use_holdout=use_holdout)
                pair_key = f"{name1}_vs_{name2}"
                all_results[pair_key] = pair_results
                
                if 'overall_discrimination' in pair_results:
                    acc = pair_results['overall_discrimination']['cv_balanced_accuracy_mean']
                    balanced_acc_matrix[i, j] = acc
                    balanced_acc_matrix[j, i] = acc  # Symmetric matrix
                    accuracy_matrix[i, j] = pair_results['overall_discrimination']['cv_accuracy_mean']
                    accuracy_matrix[j, i] = pair_results['overall_discrimination']['cv_accuracy_mean']
        
        # Create publication-quality visualizations
        # self.create_publication_heatmaps(all_results, dataset_names, output_dir, window_size, step_size)
        
        # Save results
        if output_dir:
            suffix = f"_{evaluation_method}" if use_holdout else ""
            results_path = os.path.join(output_dir, f"domain_discriminability_results_{window_size}_{step_size}{suffix}.json")
            with open(results_path, 'w') as f:
                json.dump(all_results, f, indent=2, default=str)
            
            # Save matrices
            matrix_results = {
                'dataset_names': dataset_names,
                'balanced_accuracy_matrix': balanced_acc_matrix.tolist(),
                'accuracy_matrix': accuracy_matrix.tolist(),
                'evaluation_method': evaluation_method
            }
            
            matrix_path = os.path.join(output_dir, f"discriminability_matrices_{window_size}_{step_size}{suffix}.json")
            with open(matrix_path, 'w') as f:
                json.dump(matrix_results, f, indent=2)
        
        return all_results
    

def main():
    parser = argparse.ArgumentParser(description="Domain Discriminability Analysis")
    parser.add_argument("--output_dir", type=str, default="statistical_analysis/results",
                       help="Output directory for results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fs", type=str, default="1000", help="Sampling frequency")
    parser.add_argument("--window_size", type=int, default=10, help="Window size")
    parser.add_argument("--step_size", type=int, default=5, help="Step size")
    parser.add_argument("--use_holdout", action="store_true",
                       help="Use 80/20 train-test split with participant-based CV on training data "
                            "(following train_simple_classifiers.py pattern)")
    
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
    evaluation_method = "80/20 holdout split with participant-based CV" if args.use_holdout else "cross-validation"
    print(f"Running domain discriminability analysis with {len(existing_datasets)} datasets using {evaluation_method}...")
    results = analyzer.run_full_analysis(existing_datasets, args.output_dir, args.window_size, args.step_size,
                                         use_holdout=args.use_holdout)
    
    print(f"\n=== Analysis Complete ===")
    print(f"Results saved to: {args.output_dir}")
    print(f"Evaluation method: {evaluation_method}")
    
    # Print summary
    print("\n=== Summary of Domain Discriminability ===")
    for pair_name, pair_results in results.items():
        if 'overall_discrimination' in pair_results:
            if args.use_holdout:
                # For holdout, show test performance
                test_acc = pair_results['overall_discrimination']['test_balanced_accuracy']
                cv_acc = pair_results['overall_discrimination']['cv_balanced_accuracy_mean']
                cv_std = pair_results['overall_discrimination']['cv_balanced_accuracy_std']
                print(f"{pair_name}: CV={cv_acc:.3f}±{cv_std:.3f}, Test={test_acc:.3f}")
            else:
                # For CV, show CV performance
                acc = pair_results['overall_discrimination']['cv_balanced_accuracy_mean']
                std = pair_results['overall_discrimination']['cv_balanced_accuracy_std']
                print(f"{pair_name}: {acc:.3f} ± {std:.3f}")


if __name__ == "__main__":
    main()