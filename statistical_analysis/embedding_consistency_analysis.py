#!/usr/bin/env python

import os
import sys
import json
import argparse
from typing import Dict, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, accuracy_score, roc_auc_score, average_precision_score
from sklearn.model_selection import GroupKFold
from sklearn.dummy import DummyClassifier
from utils.torch_utilities import set_seed
from utils.helper_paths import DATA_PATH

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


class SimplifiedEmbeddingDomainAnalyzer:
    """Simplified embedding domain discriminability analyzer with participant-level CV"""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        set_seed(random_state)

    def load_embeddings(self, embedding_path: str, dataset_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load embeddings, labels, and participant groups from NPZ file"""
        try:
            if not os.path.exists(embedding_path):
                print(f"Embedding file not found at {embedding_path}")
                return None, None, None

            # Load embeddings from NPZ file
            data = np.load(embedding_path, allow_pickle=True)
            embeddings = data['array1']  # X embeddings
            y = data['array_2']  # y labels
            groups = data['array_3']

            print(f"Loaded embeddings for {dataset_name}: X={embeddings.shape}, labels={len(y)}, groups={len(groups)}")

            #ToDo: Get the participant ids for each of the train idx
            # This is wrong! Here we have the groups now!

            # Handle NaN values in embeddings
            nan_mask = ~np.isnan(embeddings).any(axis=1)
            if not nan_mask.all():
                print(f"Removing {(~nan_mask).sum()} samples with NaN values from {dataset_name}")
                embeddings = embeddings[nan_mask]
                y = y[nan_mask]
                groups = groups[nan_mask]

            print(f"Final {dataset_name}: X={embeddings.shape}, participants={len(np.unique(groups))}")
            return embeddings, y, groups

        except Exception as e:
            print(f"Error loading embeddings from {embedding_path}: {e}")
            return None, None, None

    def create_combined_dataset(self, X1: np.ndarray, y1: np.ndarray, groups1: np.ndarray,
                               X2: np.ndarray, y2: np.ndarray, groups2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create combined dataset for domain discrimination"""
        # Combine datasets
        X_combined = np.vstack([X1, X2])
        y_domain = np.concatenate([np.zeros(len(X1)), np.ones(len(X2))])

        # Create unique groups across datasets
        groups1_prefixed = np.array([f"ds1_{g}" for g in groups1])
        groups2_prefixed = np.array([f"ds2_{g}" for g in groups2])
        groups_combined = np.concatenate([groups1_prefixed, groups2_prefixed])

        print(f"Combined embedding dataset: X={X_combined.shape}, domain distribution={np.bincount(y_domain.astype(int))}")
        return X_combined, y_domain, groups_combined

    def evaluate_with_participant_cv(
            self,
            X: np.ndarray,
            y_domain: np.ndarray,
            groups: np.ndarray,
            k_folds: int = 5,
            analysis_type: str = "overall"
    ) -> Dict:
        """Evaluate domain discrimination using participant-level k-fold CV"""
        unique_participants = np.unique(groups)
        n_participants = len(unique_participants)

        if n_participants < k_folds:
            print(f"Warning: Only {n_participants} participants, using {n_participants}-fold CV")
            k_folds = n_participants

        # Use GroupKFold for participant-level splits
        cv_splitter = GroupKFold(n_splits=k_folds)

        fold_results = {
            'accuracy': [],
            'balanced_accuracy': [],
            'roc_auc': [],
            'pr_auc': [],
            'dummy_accuracy': [],
            'dummy_balanced_accuracy': [],
            'dummy_roc_auc': [],
            'dummy_pr_auc': []
        }

        print(f"Running {k_folds}-fold participant-level CV for {analysis_type} analysis...")

        for fold_idx, (train_idx, test_idx) in enumerate(cv_splitter.split(X, y_domain, groups)):
            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y_domain[train_idx], y_domain[test_idx]
            groups_train, groups_test = groups[train_idx], groups[test_idx]

            # VERIFY PARTICIPANT-LEVEL SPLIT: No participant should appear in both train and test
            train_participants = set(groups_train)
            test_participants = set(groups_test)
            overlap_participants = train_participants.intersection(test_participants)

            if len(overlap_participants) > 0:
                print(f"ERROR: Participant overlap detected in fold {fold_idx+1}: {overlap_participants}")
                print("This indicates a data leakage issue!")
                continue

            print(f"Fold {fold_idx+1}: "
                  f"Train participants: {len(train_participants)}, "
                  f"Test participants: {len(test_participants)}, "
                  f"No overlap: ✓")

            # Check if we have both classes in train and test sets
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                print(f"Warning: Fold {fold_idx+1} has insufficient class diversity, skipping")
                continue

            # No standardization needed - embeddings are already normalized
            # Use embeddings directly
            X_train_processed = X_train
            X_test_processed = X_test

            # Train dummy classifier (majority class baseline)
            dummy_clf = DummyClassifier(strategy='most_frequent', random_state=self.random_state)
            dummy_clf.fit(X_train_processed, y_train)

            # Evaluate dummy classifier
            dummy_y_pred = dummy_clf.predict(X_test_processed)
            dummy_y_prob = dummy_clf.predict_proba(X_test_processed)[:, 1]

            dummy_accuracy = accuracy_score(y_test, dummy_y_pred)
            dummy_balanced_acc = balanced_accuracy_score(y_test, dummy_y_pred)
            dummy_roc_auc = roc_auc_score(y_test, dummy_y_prob)
            dummy_pr_auc = average_precision_score(y_test, dummy_y_prob)

            # Train logistic regression
            clf = LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                n_jobs=-1
            )
            clf.fit(X_train_processed, y_train)

            # Evaluate logistic regression
            y_pred = clf.predict(X_test_processed)
            y_prob = clf.predict_proba(X_test_processed)[:, 1]

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            balanced_acc = balanced_accuracy_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_prob)
            pr_auc = average_precision_score(y_test, y_prob)

            # Store results
            fold_results['accuracy'].append(accuracy)
            fold_results['balanced_accuracy'].append(balanced_acc)
            fold_results['roc_auc'].append(roc_auc)
            fold_results['pr_auc'].append(pr_auc)

            fold_results['dummy_accuracy'].append(dummy_accuracy)
            fold_results['dummy_balanced_accuracy'].append(dummy_balanced_acc)
            fold_results['dummy_roc_auc'].append(dummy_roc_auc)
            fold_results['dummy_pr_auc'].append(dummy_pr_auc)

            print(f"Fold {fold_idx+1}/{k_folds}: LogReg Bal.Acc={balanced_acc:.3f}, "
                  f"Dummy Bal.Acc={dummy_balanced_acc:.3f}, LogReg ROC-AUC={roc_auc:.3f}, PR-AUC={pr_auc:.3f}")

        # Calculate mean and std across folds
        results = {}
        for metric, values in fold_results.items():
            if len(values) > 0:
                values = np.array(values)
                results[f'{metric}_mean'] = values.mean()
                results[f'{metric}_std'] = values.std()
            else:
                results[f'{metric}_mean'] = np.nan
                results[f'{metric}_std'] = np.nan

        results['n_folds'] = len(fold_results['accuracy'])
        results['n_participants'] = n_participants
        results['analysis_type'] = analysis_type

        return results

    def create_class_specific_dataset(self, X1: np.ndarray, y1: np.ndarray, groups1: np.ndarray,
                                     X2: np.ndarray, y2: np.ndarray, groups2: np.ndarray,
                                     target_class: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create dataset for class-specific domain discrimination"""
        # Filter to specific class
        mask1 = y1 == target_class
        mask2 = y2 == target_class

        if np.sum(mask1) < 10 or np.sum(mask2) < 10:
            print(f"Warning: Insufficient samples for class {target_class} analysis")
            return None, None, None

        X1_class = X1[mask1]
        groups1_class = groups1[mask1]
        X2_class = X2[mask2]
        groups2_class = groups2[mask2]

        # Combine datasets for this class
        X_combined = np.vstack([X1_class, X2_class])
        y_domain = np.concatenate([np.zeros(len(X1_class)), np.ones(len(X2_class))])

        # Create unique groups across datasets
        groups1_prefixed = np.array([f"ds1_{g}" for g in groups1_class])
        groups2_prefixed = np.array([f"ds2_{g}" for g in groups2_class])
        groups_combined = np.concatenate([groups1_prefixed, groups2_prefixed])

        print(f"Class {target_class} embedding dataset: X={X_combined.shape}, domain distribution={np.bincount(y_domain.astype(int))}")
        return X_combined, y_domain, groups_combined

    def analyze_dataset_pair(self, embedding1_path: str, dataset1_name: str,
                           embedding2_path: str, dataset2_name: str,
                           k_folds: int = 5) -> Dict:
        """Analyze domain discriminability between two datasets using embeddings"""
        print(f"\n=== Analyzing {dataset1_name} vs {dataset2_name} (Embeddings) ===")

        # Load embeddings
        X1, y1, groups1 = self.load_embeddings(embedding1_path, dataset1_name)
        X2, y2, groups2 = self.load_embeddings(embedding2_path, dataset2_name)

        if X1 is None or X2 is None:
            return {'error': 'Failed to load embeddings'}

        # Filter to binary classification only
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
            'embedding_dim': X1_binary.shape[1]
        }

        # 1. Overall domain discrimination (all classes mixed)
        print(f"\n--- Overall Domain Discrimination ---")
        X_combined_overall, y_domain_overall, groups_combined_overall = self.create_combined_dataset(
            X1_binary, y1_binary, groups1_binary, X2_binary, y2_binary, groups2_binary
        )

        overall_results = self.evaluate_with_participant_cv(
            X_combined_overall, y_domain_overall, groups_combined_overall, k_folds, "overall"
        )
        results['overall_discrimination'] = overall_results

        # 2. Per-class domain discrimination
        class_results = {}

        # Class 0 (baseline) discrimination
        print(f"\n--- Class 0 (Baseline) Domain Discrimination ---")
        X_class0, y_domain_class0, groups_class0 = self.create_class_specific_dataset(
            X1_binary, y1_binary, groups1_binary, X2_binary, y2_binary, groups2_binary, target_class=0
        )

        if X_class0 is not None:
            class0_results = self.evaluate_with_participant_cv(
                X_class0, y_domain_class0, groups_class0, k_folds, "class_0_baseline"
            )
            class_results['class_0_baseline'] = class0_results

        # Class 1 (mental stress) discrimination
        print(f"\n--- Class 1 (Mental Stress) Domain Discrimination ---")
        X_class1, y_domain_class1, groups_class1 = self.create_class_specific_dataset(
            X1_binary, y1_binary, groups1_binary, X2_binary, y2_binary, groups2_binary, target_class=1
        )

        if X_class1 is not None:
            class1_results = self.evaluate_with_participant_cv(
                X_class1, y_domain_class1, groups_class1, k_folds, "class_1_stress"
            )
            class_results['class_1_stress'] = class1_results

        results['per_class_discrimination'] = class_results

        return results

    def run_pairwise_analysis(self, datasets: Dict[str, str], k_folds: int = 5) -> Dict:
        """Run pairwise analysis for all dataset combinations"""
        results = {}
        dataset_names = list(datasets.keys())

        print(f"Running pairwise embedding analysis for {len(dataset_names)} datasets with {k_folds}-fold CV...")

        for i, (name1, path1) in enumerate(datasets.items()):
            for j, (name2, path2) in enumerate(datasets.items()):
                if i >= j:  # Skip diagonal and lower triangle
                    continue

                pair_results = self.analyze_dataset_pair(path1, name1, path2, name2, k_folds)
                pair_key = f"{name1}_vs_{name2}"
                results[pair_key] = pair_results

        return results


def main():
    parser = argparse.ArgumentParser(description="Simplified Embedding Consistency Analysis")
    parser.add_argument("--output_dir", type=str, default="statistical_analysis/results",
                       help="Output directory for results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--ssl_method", type=str, default="TSTCC",
                       choices=["TSTCC", "TS2Vec", "SimCLR", "SoftTSTCC", "SoftTS2Vec"],
                       help="SSL method for embedding extraction")
    parser.add_argument("--window_size", type=int, default=10, help="Window size")
    parser.add_argument("--step_size", type=int, default=5, help="Step size")
    parser.add_argument("--k_folds", type=int, default=5, help="Number of CV folds")

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = SimplifiedEmbeddingDomainAnalyzer(random_state=args.seed)

    # Define embedding paths based on SSL method
    # data/embeddings/<dataset>/<fs>/<ssl_method>/<seed>/<window_size>/<step_size>/x_y_embedding.npz
    datasets = {
        'Ours': os.path.join(DATA_PATH, "embeddings", "ECG", "1000", args.ssl_method,
                           str(args.seed), str(args.window_size), str(args.step_size), "x_y_groups_embedding.npz"),
        'STRESSID': os.path.join(DATA_PATH, "embeddings", "STRESSID", "500", args.ssl_method,
                                str(args.seed), str(args.window_size), str(args.step_size), "x_y_groups_embedding.npz"),
        'WESAD': os.path.join(DATA_PATH, "embeddings", "WESAD", "700", args.ssl_method,
                             str(args.seed), str(args.window_size), str(args.step_size), "x_y_groups_embedding.npz")
    }

    # Filter existing datasets
    existing_datasets = {}
    for name, path in datasets.items():
        if os.path.exists(path):
            existing_datasets[name] = path
            print(f"Found embeddings: {name}")
        else:
            print(f"Warning: Embeddings {name} not found at {path}")

    if len(existing_datasets) < 2:
        print("Error: Need at least 2 datasets for comparison")
        print("Please ensure you have run SSL training with --save_embeddings flag")
        return

    # Run analysis
    results = analyzer.run_pairwise_analysis(existing_datasets, args.k_folds)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(
        args.output_dir, f"embedding_consistency_{args.ssl_method}_{args.window_size}_{args.step_size}.json"
    )
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n=== Analysis Complete ===")
    print(f"Results saved to: {results_path}")

    # Print comprehensive summary
    print("\n=== Embedding Domain Discrimination Results ===")
    for pair_name, pair_results in results.items():
        if 'error' in pair_results:
            print(f"{pair_name}: ERROR - {pair_results['error']}")
            continue

        print(f"\n{pair_name}:")
        print(f"  Dataset sizes: {pair_results['dataset1_samples']} vs {pair_results['dataset2_samples']} samples")
        print(f"  Participants: {pair_results['dataset1_participants']} vs {pair_results['dataset2_participants']}")
        print(f"  Embedding dim: {pair_results['embedding_dim']}")

        # Overall discrimination results
        if 'overall_discrimination' in pair_results:
            overall = pair_results['overall_discrimination']
            print(f"  \n  OVERALL DISCRIMINATION:")
            print(f"    LOGISTIC REGRESSION:")
            print(f"      Balanced Accuracy: {overall['balanced_accuracy_mean']:.3f} ± {overall['balanced_accuracy_std']:.3f}")
            print(f"      ROC-AUC: {overall['roc_auc_mean']:.3f} ± {overall['roc_auc_std']:.3f}")
            print(f"      PR-AUC: {overall['pr_auc_mean']:.3f} ± {overall['pr_auc_std']:.3f}")
            print(f"      Accuracy: {overall['accuracy_mean']:.3f} ± {overall['accuracy_std']:.3f}")
            print(f"    DUMMY CLASSIFIER (Majority Class):")
            print(f"      Balanced Accuracy: {overall['dummy_balanced_accuracy_mean']:.3f} ± {overall['dummy_balanced_accuracy_std']:.3f}")
            print(f"      ROC-AUC: {overall['dummy_roc_auc_mean']:.3f} ± {overall['dummy_roc_auc_std']:.3f}")
            print(f"      PR-AUC: {overall['dummy_pr_auc_mean']:.3f} ± {overall['dummy_pr_auc_std']:.3f}")
            print(f"      Accuracy: {overall['dummy_accuracy_mean']:.3f} ± {overall['dummy_accuracy_std']:.3f}")
            print(f"    Folds completed: {overall['n_folds']}")

        # Per-class discrimination results
        if 'per_class_discrimination' in pair_results:
            per_class = pair_results['per_class_discrimination']

            if 'class_0_baseline' in per_class:
                class0 = per_class['class_0_baseline']
                print(f"  \n  CLASS 0 (BASELINE) DISCRIMINATION:")
                print(f"    LOGISTIC REGRESSION:")
                print(f"      Balanced Accuracy: {class0['balanced_accuracy_mean']:.3f} ± {class0['balanced_accuracy_std']:.3f}")
                print(f"      ROC-AUC: {class0['roc_auc_mean']:.3f} ± {class0['roc_auc_std']:.3f}")
                print(f"      PR-AUC: {class0['pr_auc_mean']:.3f} ± {class0['pr_auc_std']:.3f}")
                print(f"    DUMMY CLASSIFIER:")
                print(f"      Balanced Accuracy: {class0['dummy_balanced_accuracy_mean']:.3f} ± {class0['dummy_balanced_accuracy_std']:.3f}")
                print(f"      ROC-AUC: {class0['dummy_roc_auc_mean']:.3f} ± {class0['dummy_roc_auc_std']:.3f}")
                print(f"      PR-AUC: {class0['dummy_pr_auc_mean']:.3f} ± {class0['dummy_pr_auc_std']:.3f}")
                print(f"    Folds completed: {class0['n_folds']}")

            if 'class_1_stress' in per_class:
                class1 = per_class['class_1_stress']
                print(f"  \n  CLASS 1 (STRESS) DISCRIMINATION:")
                print(f"    LOGISTIC REGRESSION:")
                print(f"      Balanced Accuracy: {class1['balanced_accuracy_mean']:.3f} ± {class1['balanced_accuracy_std']:.3f}")
                print(f"      ROC-AUC: {class1['roc_auc_mean']:.3f} ± {class1['roc_auc_std']:.3f}")
                print(f"      PR-AUC: {class1['pr_auc_mean']:.3f} ± {class1['pr_auc_std']:.3f}")
                print(f"    DUMMY CLASSIFIER:")
                print(f"      Balanced Accuracy: {class1['dummy_balanced_accuracy_mean']:.3f} ± {class1['dummy_balanced_accuracy_std']:.3f}")
                print(f"      ROC-AUC: {class1['dummy_roc_auc_mean']:.3f} ± {class1['dummy_roc_auc_std']:.3f}")
                print(f"      PR-AUC: {class1['dummy_pr_auc_mean']:.3f} ± {class1['dummy_pr_auc_std']:.3f}")
                print(f"    Folds completed: {class1['n_folds']}")

    print(f"\n=== Summary ===")
    print("Higher scores indicate better ability to discriminate between datasets (larger domain gap)")
    print("Scores close to 0.5 indicate datasets are similar (small domain gap)")
    print("Analysis uses participant-level CV splits to prevent data leakage")
    print(f"SSL method used: {args.ssl_method}")


if __name__ == "__main__":
    main()