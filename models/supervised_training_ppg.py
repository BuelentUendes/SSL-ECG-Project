############################################
# Simple script for supervised ppg training
############################################

####
# This is the script of supervised_training without metaflow
###

import os
import gc
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch

from torch.utils.data import DataLoader

from utils.helper_paths import SAVED_MODELS_PATH, DATA_PATH

from utils.torch_utilities import (
    load_processed_data,
    split_indices_by_participant,
    build_supervised_fingerprint,
    PhysiologicalDataset,
    set_seed,
    create_directory,
    train_one_epoch,
    test,
    EarlyStopping,
)

from models.supervised import (
    Improved1DCNN_v2,
    TCNClassifier,
    TransformerECGClassifier,
)


def main(
        window_data_path: str,
        mlflow_tracking_uri: str,
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
        "cnn": "Supervised_CNN",
        "tcn": "Supervised_TCN",
        "transformer": "Supervised_Transformer",
    }
    experiment_name = exp_map.get(model_type.lower())
    if experiment_name is None:
        raise ValueError(f"Unknown model_type '{model_type}'")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)
    run = mlflow.start_run(run_name=None)

    # Check if directory for saving model parameters exist, otherwise create it
    create_directory(SAVED_MODELS_PATH)

    # We save the model here via seeds, we create a separate folder for pretraining on all labels and on only task-related data
    model_save_path = os.path.join(SAVED_MODELS_PATH, f"{model_type}", f"{seed}", f"{label_fraction}")
    create_directory(model_save_path)

    # load data
    X, y, groups = load_processed_data(
        window_data_path,
        label_map={"baseline": 0, "mental_stress": 1},
    )
    y = y.astype(np.float32)
    n_features = X.shape[2]

    # train/val/test split
    tr_idx, val_idx, te_idx = split_indices_by_participant(groups, label_fraction=label_fraction, seed=seed)
    print(f"windows: train {len(tr_idx)}, val {len(val_idx)}, test {len(te_idx)}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train ECG classifier")
    parser.add_argument("--window_data_path",
                        default=f"{os.path.join(DATA_PATH, 'interim', 'windowed_data.h5')}")
    parser.add_argument("--mlflow_tracking_uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    parser.add_argument("--model_type", choices=["cnn", "tcn", "transformer"], default="cnn")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force_retraining", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-5) #lr 1e-4 was good
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--scheduler_mode", default="min")
    parser.add_argument("--scheduler_factor", type=float, default=0.5)
    parser.add_argument("--scheduler_patience", type=int, default=2)
    parser.add_argument("--scheduler_min_lr", type=float, default=1e-9)
    parser.add_argument("--label_fraction", type=float, default=0.1,
                        help="Percent of labeled participants in the training stage.")
    args = parser.parse_args()
    main(**vars(args))


