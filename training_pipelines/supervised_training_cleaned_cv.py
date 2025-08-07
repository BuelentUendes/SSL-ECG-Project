####
# This is the script of supervised_training with cross-validation
###

import os
import json
import gc
import argparse
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, average_precision_score
from sklearn.model_selection import GridSearchCV
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping as SkorchEarlyStopping

from torch.utils.data import DataLoader

from utils.helper_paths import SAVED_MODELS_PATH, DATA_PATH, RESULTS_PATH

from utils.torch_utilities import (
    load_processed_data,
    split_indices_by_participant_groups,  # Changed from split_indices_by_participant
    build_supervised_fingerprint,
    PhysiologicalDataset,
    set_seed,
    create_directory,
    train_one_epoch,
    test,
    EarlyStopping,
    get_participant_cv_splitter,  # Added CV splitter
)

from models.supervised import (
    Improved1DCNN_v2,
    TCNClassifier,
    TransformerECGClassifier,
)


def create_skorch_classifier(model_class, device, max_epochs=25, lr=1e-4, batch_size=32,
                             patience=20, scheduler_mode='min', scheduler_factor=0.5,
                             scheduler_patience=2, scheduler_min_lr=1e-9, weight_decay=3e-4):
    """Create a skorch NeuralNetClassifier wrapper for PyTorch models"""

    # Use CPU if device is not CUDA (skorch handles device differently)
    device_str = 'cuda' if device.type == 'cuda' else 'cpu'

    # For GridSearchCV, we simplify the callbacks since CV handles model selection
    net = NeuralNetClassifier(
        model_class,
        max_epochs=max_epochs,
        lr=lr,
        batch_size=batch_size,
        device=device_str,
        criterion=nn.BCEWithLogitsLoss,
        optimizer=torch.optim.AdamW,  # AdamW optimizer
        optimizer__weight_decay=weight_decay,  # AdamW weight decay
        iterator_train__shuffle=True,
        iterator_train__num_workers=min(4, os.cpu_count() or 2),
        iterator_valid__shuffle=False,
        iterator_valid__num_workers=min(4, os.cpu_count() or 2),
        train_split=None,  # No internal splitting - GridSearchCV handles this
        verbose=0,  # Reduce verbosity during grid search
    )

    return net


def run_sklearn_grid_search_cv(
        model_class, X_train, y_train, groups_train, X_test, y_test,
        cv_splitter, device, model_type, scheduler_mode='min', scheduler_factor=0.5,
        scheduler_patience=2, scheduler_min_lr=1e-9, verbose=False
):
    """Run sklearn GridSearchCV with skorch for deep learning models"""

    print("Setting up sklearn GridSearchCV with skorch...")

    # Create base skorch classifier with scheduler parameters
    net = create_skorch_classifier(
        model_class, device,
        scheduler_mode=scheduler_mode,
        scheduler_factor=scheduler_factor,
        scheduler_patience=scheduler_patience,
        scheduler_min_lr=scheduler_min_lr
    )

    # Define parameter grid - simplified since GridSearchCV handles model selection
    param_grid = {
        'lr': [1e-3, 1e-4],
        # 'batch_size': [16, 32, 64],
        # 'max_epochs': [25, 50, 75],
        # 'optimizer__weight_decay': [0.0, 1e-4, 3e-4],  # AdamW weight decay
    }

    print(f"Using AdamW optimizer (no internal callbacks - GridSearchCV handles selection)")
    print(f"Parameter grid: {param_grid}")
    total_combinations = np.prod([len(v) for v in param_grid.values()])
    print(f"Total combinations: {total_combinations}")

    # Set up GridSearchCV with custom CV splitter
    grid_search = GridSearchCV(
        estimator=net,
        param_grid=param_grid,
        cv=cv_splitter,
        scoring='roc_auc',
        n_jobs=1,  # PyTorch models don't parallelize well
        verbose=1 if verbose else 0,
        refit=True,  # Refit on full training set with best params
        return_train_score=True
    )

    # Convert data to float32 for PyTorch compatibility
    X_train_torch = X_train.astype(np.float32)
    y_train_torch = y_train.astype(np.float32)
    X_test_torch = X_test.astype(np.float32)
    y_test_torch = y_test.astype(np.float32)

    print("Running GridSearchCV...")

    if cv_splitter is None:
        print("Warning: cv_splitter is None, using default CV")
        grid_search.fit(X_train_torch, y_train_torch)
    else:
        print(f"Using custom CV splitter with {cv_splitter.get_n_splits()} folds")
        grid_search.fit(X_train_torch, y_train_torch, groups=groups_train)

    print("GridSearchCV complete!")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV F1 score: {grid_search.best_score_:.4f}")

    # Get the best model (already fitted on full training set)
    best_model = grid_search.best_estimator_

    # Evaluate on test set
    print("Evaluating on test set...")

    # Get predictions and probabilities
    y_pred = best_model.predict(X_test_torch)
    y_pred_proba = best_model.predict_proba(X_test_torch)[:, 1]  # Get positive class probabilities

    # Calculate metrics
    test_acc = accuracy_score(y_test_torch, y_pred)
    test_auroc = roc_auc_score(y_test_torch, y_pred_proba)
    test_f1 = f1_score(y_test_torch, y_pred)
    test_pr_auc = average_precision_score(y_test_torch, y_pred_proba)

    print(f"\n=== Test Set Results ===")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test AUROC: {test_auroc:.4f}")
    print(f"Test F1: {test_f1:.4f}")
    print(f"Test PR-AUC: {test_pr_auc:.4f}")

    # Extract the underlying PyTorch model
    pytorch_model = best_model.module_

    return {
        'best_params': grid_search.best_params_,
        'best_cv_score': grid_search.best_score_,
        'cv_results': grid_search.cv_results_,
        'test_metrics': {
            'accuracy': test_acc,
            'auroc': test_auroc,
            'f1': test_f1,
            'pr_auc': test_pr_auc
        },
        'model': pytorch_model,  # Return the PyTorch model
        'skorch_model': best_model,  # Also return the skorch wrapper
        'grid_search': grid_search,
        'best_threshold': 0.5  # Default threshold (skorch handles this internally)
    }


def main(
        mlflow_tracking_uri: str,
        fs: str,
        model_type: str = "cnn",
        gpu: int = 0,
        seed: int = 42,
        force_retraining: bool = True,
        lr: float = 1e-4,
        batch_size: int = 32,
        num_epochs: int = 25,
        patience: int = 5,
        scheduler_mode: str = "min",
        scheduler_factor: float = 0.1,
        scheduler_patience: int = 2,
        scheduler_min_lr: float = 1e-11,
        label_fraction: float = 1.0,
        k_folds: int = 5,
        min_participants_for_kfold: int = 5,
        verbose: bool = False,
):
    set_seed(seed)

    # device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    use_cuda = (device.type == "cuda")
    pin_memory = use_cuda

    # mlflow setup
    exp_map = {
        "cnn": "Supervised_CNN_CV",
        "tcn": "Supervised_TCN_CV",
        "transformer": "Supervised_Transformer_CV",
    }
    experiment_name = exp_map.get(model_type.lower())
    if experiment_name is None:
        raise ValueError(f"Unknown model_type '{model_type}'")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)
    run = mlflow.start_run(run_name=f"cv_{model_type}_{seed}_lf_{label_fraction}")

    # Check if directory for saving model parameters exist, otherwise create it
    create_directory(SAVED_MODELS_PATH)
    create_directory(RESULTS_PATH)

    # We save the model here via seeds, we create a separate folder for pretraining on all labels and on only task-related data
    model_save_path = os.path.join(SAVED_MODELS_PATH, "ECG", str(fs), f"{model_type}_cv", f"{seed}",
                                   f"{label_fraction}")
    results_save_path = os.path.join(RESULTS_PATH, "ECG", "Supervised_CV", model_type, f"{seed}", f"{label_fraction}")

    create_directory(model_save_path)
    create_directory(results_save_path)

    # Data path
    window_data_path = os.path.join(DATA_PATH, "interim", "ECG", str(fs), 'windowed_data.h5')

    # load data
    X, y, groups = load_processed_data(
        window_data_path,
        label_map={"baseline": 0, "mental_stress": 1},
    )
    y = y.astype(np.float32)
    n_features = X.shape[2]

    # Changed: train/test split using participant groups (similar to baseline classifier)
    train_idx, train_p, test_idx, test_p = split_indices_by_participant_groups(
        groups,
        train_ratio=0.8,
        label_fraction=label_fraction,
        seed=seed
    )

    print(f"windows: train {len(train_idx)}, test {len(test_idx)}")
    print(f"participants: train {len(np.unique(groups[train_idx]))}, test {len(np.unique(groups[test_idx]))}")

    # Prepare training and test data
    X_train = X[train_idx]
    y_train = y[train_idx]
    groups_train = groups[train_idx]

    X_test = X[test_idx]
    y_test = y[test_idx]
    groups_test = groups[test_idx]

    # Set up Cross-Validation Splitter
    cv_splitter, n_splits = get_participant_cv_splitter(
        groups_train,
        min_participants_for_kfold=min_participants_for_kfold,
        k=k_folds
    )

    # Define model class
    model_classes = {
        "cnn": Improved1DCNN_v2,
        "tcn": TCNClassifier,
        "transformer": TransformerECGClassifier,
    }
    model_class = model_classes[model_type.lower()]

    # Hyperparameters for CV
    hyperparams = {
        'lr': lr,
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'patience': patience,
        'scheduler_mode': scheduler_mode,
        'scheduler_factor': scheduler_factor,
        'scheduler_patience': scheduler_patience,
        'scheduler_min_lr': scheduler_min_lr,
    }

    # fingerprint
    fp = build_supervised_fingerprint({
        "model_name": model_type.lower(),
        "seed": seed,
        "lr": lr,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "patience": patience,
        "scheduler_mode": scheduler_mode,
        "scheduler_factor": scheduler_factor,
        "scheduler_patience": scheduler_patience,
        "scheduler_min_lr": scheduler_min_lr,
        "label_fraction": label_fraction,
        "k_folds": k_folds,
        "cv_enabled": True,
    })
    mlflow.log_params(fp)

    # Cross-validation and final training
    if force_retraining:
        print("Running sklearn GridSearchCV with skorch...")
        results = run_sklearn_grid_search_cv(
            model_class, X_train, y_train, groups_train, X_test, y_test,
            cv_splitter, device, model_type,
            scheduler_mode=scheduler_mode,
            scheduler_factor=scheduler_factor,
            scheduler_patience=scheduler_patience,
            scheduler_min_lr=scheduler_min_lr,
            verbose=verbose
        )

        # Extract results
        best_params = results['best_params']
        best_cv_f1 = results['best_cv_score']
        final_model = results['model']  # This is the PyTorch model
        best_threshold = results['best_threshold']
        test_metrics = results['test_metrics']

        acc = test_metrics['accuracy']
        auroc = test_metrics['auroc']
        f1 = test_metrics['f1']
        prauc = test_metrics['pr_auc']
        loss = 0.0  # Not returned by the function, set to 0

        mlflow.log_param("chosen_threshold", best_threshold)
        mlflow.log_param("best_cv_f1", best_cv_f1)

        # Save model
        saved_results = os.path.join(model_save_path, f"{model_type}_cv.pt")
        torch.save(
            {"model_parameters": final_model.state_dict(),
             "best_threshold": best_threshold,
             "best_cv_f1": best_cv_f1,
             "best_params": best_params},
            saved_results
        )

        # Save model to MLflow
        if np.isclose(label_fraction, 1.0):
            mlflow.pytorch.log_model(final_model, artifact_path="supervised_model_cv")

    else:
        # Load parameters from saved results
        saved_results = os.path.join(model_save_path, f"{model_type}.pt")

        if os.path.exists(saved_results):
            print(f"Loading saved model parameters from: {saved_results}")
            checkpoint = torch.load(saved_results, map_location=device, weights_only=False)

            # Initialize model and load state dict
            final_model = model_class().to(device)
            final_model.load_state_dict(checkpoint["model_parameters"])

            # Load best threshold and CV results
            best_threshold = checkpoint.get("best_threshold", 0.5)
            best_cv_f1 = checkpoint.get("best_cv_f1", -1.0)
            best_params = checkpoint.get("best_params", hyperparams)

            print(f"Loaded model with best_threshold: {best_threshold:.4f}, best_cv_f1: {best_cv_f1:.4f}")

            # Need to evaluate on test set if loading model
            print("Evaluating loaded model on test set...")
            te_ds = PhysiologicalDataset(X_test, y_test)
            test_loader = DataLoader(
                te_ds, batch_size=batch_size, shuffle=False,
                num_workers=min(8, os.cpu_count() or 2), pin_memory=pin_memory,
            )

            loss_fn = nn.BCEWithLogitsLoss()
            loss, acc, auroc, prauc, f1 = test(
                final_model, test_loader, device,
                threshold=best_threshold, loss_fn=loss_fn,
            )

            # Clean up test loader
            if hasattr(test_loader, '_iterator') and test_loader._iterator is not None:
                test_loader._iterator._shutdown_workers()
            del test_loader

        else:
            print(f"No saved model found at {saved_results}. Please run with --force_retraining")
            raise FileNotFoundError(f"Model file not found: {saved_results}")

    # Save results
    with open(os.path.join(results_save_path, "test_results.json"), "w") as f:
        classification_results = {
            "test_metrics": {
                "accuracy": acc,
                "auroc": auroc,
                "f1": f1,
                "pr_auc": prauc
            },
            "best_threshold": best_threshold,
            "best_cv_f1": best_cv_f1,
            "best_params": best_params,
        }
        json.dump(classification_results, f, indent=2, default=str)

    print(f"Test acc: {acc:.4f}, AUROC: {auroc:.4f}, F1: {f1:.4f}")

    # Log metrics to MLflow
    mlflow.log_metrics({
        "best_cv_f1": best_cv_f1,
        "test_loss": loss,
        "test_accuracy": acc,
        "test_auroc": auroc,
        "test_f1": f1,
        "test_pr_auc": prauc,
    })

    mlflow.log_params({
        "k_folds": k_folds,
        "n_cv_splits": n_splits,
    })

    # Log best parameters with sklearn GridSearchCV format
    if best_params:
        for param_name, param_value in best_params.items():
            mlflow.log_param(f"best_{param_name}", param_value)

    # cleanup
    for _ in range(3): gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    mlflow.end_run()
    print(f"=== Cross-Validation Training Complete! Results saved to {results_save_path} ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train ECG classifier with Cross-Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # ══════════════════════════════════════════════════════════════════════════════
    # General Setup
    # ══════════════════════════════════════════════════════════════════════════════
    general_group = parser.add_argument_group('General Setup')
    general_group.add_argument("--mlflow_tracking_uri",
                               default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"),
                               help="MLflow tracking URI for experiment logging")
    general_group.add_argument("--gpu", type=int, default=0,
                               help="GPU device ID to use")
    general_group.add_argument("--seed", type=int, default=42,
                               help="Random seed for reproducibility")
    general_group.add_argument("--verbose", action="store_true",
                               help="Show verbose training output")
    general_group.add_argument("--force_retraining", action="store_true",
                               help="Force retraining even if saved model exists")

    # ══════════════════════════════════════════════════════════════════════════════
    # Data Configuration
    # ══════════════════════════════════════════════════════════════════════════════
    data_group = parser.add_argument_group('Data Configuration')
    data_group.add_argument("--fs", default=1000, type=str,
                            help="Sampling frequency used for training")
    data_group.add_argument("--label_fraction", type=float, default=1.0,
                            help="Fraction of labeled participants to use (0.0-1.0)")

    # ══════════════════════════════════════════════════════════════════════════════
    # Model Configuration
    # ══════════════════════════════════════════════════════════════════════════════
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument("--model_type", choices=["cnn", "tcn", "transformer"],
                             default="cnn", help="Type of model architecture to use")

    # ══════════════════════════════════════════════════════════════════════════════
    # Training Configuration
    # ══════════════════════════════════════════════════════════════════════════════
    training_group = parser.add_argument_group('Training Configuration')
    training_group.add_argument("--lr", type=float, default=1e-5,
                                help="Learning rate for training")
    training_group.add_argument("--batch_size", type=int, default=16,
                                help="Batch size for training")
    training_group.add_argument("--num_epochs", type=int, default=2,
                                help="Maximum number of training epochs")
    training_group.add_argument("--patience", type=int, default=2,
                                help="Early stopping patience")

    # Scheduler Configuration
    scheduler_group = parser.add_argument_group('Learning Rate Scheduler')
    scheduler_group.add_argument("--scheduler_mode", default="min",
                                 help="Learning rate scheduler mode")
    scheduler_group.add_argument("--scheduler_factor", type=float, default=0.5,
                                 help="Learning rate reduction factor")
    scheduler_group.add_argument("--scheduler_patience", type=int, default=2,
                                 help="Scheduler patience")
    scheduler_group.add_argument("--scheduler_min_lr", type=float, default=1e-9,
                                 help="Minimum learning rate")

    # ══════════════════════════════════════════════════════════════════════════════
    # Cross-Validation Configuration
    # ══════════════════════════════════════════════════════════════════════════════
    cv_group = parser.add_argument_group('Cross-Validation')
    cv_group.add_argument("--k_folds", type=int, default=5,
                          help="Number of folds for cross-validation")
    cv_group.add_argument("--min_participants_for_kfold", type=int, default=5,
                          help="Minimum participants needed for k-fold (otherwise use LOGO-CV)")

    args = parser.parse_args()
    main(**vars(args))