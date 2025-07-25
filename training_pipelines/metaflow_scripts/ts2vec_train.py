import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from metaflow import FlowSpec, step, Parameter, current, project, resources

from models.ts2vec import TS2Vec, build_fingerprint, search_encoder_fp, build_linear_loaders, \
    train_linear_classifier, evaluate_classifier

from utils.torch_utilities import load_processed_data, split_indices_by_participant, set_seed

from models.supervised import (
    LinearClassifier,
    MLPClassifier,
)

# Metaflow Pipeline for TS2Vec pretraining and classifier fine-tuning
@project(name="ecg_training_ts2vec")
class ECGTS2VecFlow(FlowSpec):
    # MLflow and data parameters
    mlflow_tracking_uri = Parameter("mlflow_tracking_uri",
                                    default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    window_data_path = Parameter("window_data_path",
                                 default="../data/interim/windowed_data.h5")
    seed = Parameter("seed", default=42)
    
    # TS2Vec pre-training parameters
    ts2vec_epochs = Parameter("ts2vec_epochs", help="Epochs for TS2Vec pretraining", default=50)
    ts2vec_lr = Parameter("ts2vec_lr", help="Learning rate for TS2Vec", default=0.001)
    ts2vec_batch_size = Parameter("ts2vec_batch_size", help="Batch size for TS2Vec", default=8)
    ts2vec_output_dims = Parameter("ts2vec_output_dims", help="Representation dimension (Co)", default=320)
    ts2vec_hidden_dims = Parameter("ts2vec_hidden_dims", help="Hidden dimension (Ch)", default=64)
    ts2vec_depth = Parameter("ts2vec_depth", help="Depth (# dilated conv blocks)", default=10)
    ts2vec_max_train_length = Parameter("ts2vec_max_train_length", help="Max training length", default=5000)
    ts2vec_temporal_unit = Parameter("ts2vec_temporal_unit", help="Temporal unit for hierarchical pooling", default=0)
    
    # Classifier training hyperparameters
    classifier_epochs = Parameter("classifier_epochs", default=25)
    classifier_lr = Parameter("classifier_lr", default=0.0001)
    classifier_batch_size = Parameter("classifier_batch_size", default=32)
    label_fraction = Parameter("label_fraction", default=1.0)

    @step
    def start(self):
        """Initialize MLflow and set seed."""
        set_seed(self.seed)
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        # Set a custom MLflow experiment name
        mlflow.set_experiment("TS2Vec")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            run = mlflow.start_run(run_name=current.run_id)
            self.mlflow_run_id = run.info.run_id
        except Exception as e:
            raise RuntimeError(f"MLflow connection failed: {str(e)}")

        logging.info(f"MLflow experiment 'SoftTS2Vec' (run: {self.mlflow_run_id})")
        print(f"Using device: {self.device}")
        self.next(self.preprocess_data)

    @step
    def preprocess_data(self):
        """
        Load the windowed processed data, create participant split indices,
        persist only those indices to the next step.
        """
        X, y, groups = load_processed_data(self.window_data_path,
                                           label_map={"baseline": 0, "mental_stress": 1})
        
        # split
        train_idx, val_idx, test_idx = split_indices_by_participant(groups, seed=42)
        
        # store artifacts 
        self.train_idx, self.val_idx, self.test_idx = train_idx, val_idx, test_idx
        self.y = y.astype(np.float32)                   
        self.n_features = X.shape[2]          

        print(f"windows: train {len(train_idx)}, val {len(val_idx)}, test {len(test_idx)}")
        self.next(self.train_ts2vec)

    @resources(memory=16000)
    @step
    def train_ts2vec(self):
        """Train TS2Vec or load from MLflow if matching config exists."""
        torch.cuda.empty_cache(), torch.cuda.ipc_collect()
        set_seed(self.seed)
        mlflow.set_experiment("TS2Vec")

        # Load only training data
        X, _, _ = load_processed_data(self.window_data_path)
        X_train = X[self.train_idx].astype(np.float32)
        del X

        # Build fingerprint and check for cache
        fp = {
            "model_name": "TS2Vec",
            "seed": self.seed,
            "ts2vec_epochs": self.ts2vec_epochs,
            "ts2vec_output_dims": self.ts2vec_output_dims,
            "ts2vec_hidden_dims": self.ts2vec_hidden_dims,
            "ts2vec_depth": self.ts2vec_depth,
            "ts2vec_max_train_length": self.ts2vec_max_train_length,
            "ts2vec_temporal_unit": self.ts2vec_temporal_unit,
        }
        fp = build_fingerprint(fp)

        run_id = search_encoder_fp(
            fp, experiment_name="TS2Vec",
            tracking_uri=self.mlflow_tracking_uri
        )

        if run_id:
            # Load from MLflow
            print(f"encoder found: re-using run {run_id}")
            uri = f"runs:/{run_id}/ts2vec_model"
            net = mlflow.pytorch.load_model(uri, map_location=self.device)

            self.ts2vec = TS2Vec(
                input_dims=self.n_features,
                output_dims=self.ts2vec_output_dims,
                hidden_dims=self.ts2vec_hidden_dims,
                depth=self.ts2vec_depth,
                device=self.device,
                lr=self.ts2vec_lr,
                batch_size=self.ts2vec_batch_size,
                max_train_length=self.ts2vec_max_train_length,
                temporal_unit=self.ts2vec_temporal_unit,
            )
            self.ts2vec.net = self.ts2vec._net = net

        else:
            # Train from scratch
            print("no cached encoder: training from scratch")
            self.ts2vec = TS2Vec(
                input_dims=self.n_features,
                output_dims=self.ts2vec_output_dims,
                hidden_dims=self.ts2vec_hidden_dims,
                depth=self.ts2vec_depth,
                device=self.device,
                lr=self.ts2vec_lr,
                batch_size=self.ts2vec_batch_size,
                max_train_length=self.ts2vec_max_train_length,
                temporal_unit=self.ts2vec_temporal_unit,
            )
            
            # Log and run pretraining
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            with mlflow.start_run(run_id=self.mlflow_run_id):
                mlflow.log_params(fp)

                loss_log = self.ts2vec.fit(
                    X_train,
                    n_epochs=self.ts2vec_epochs,
                    verbose=True
                )

                mlflow.pytorch.log_model(
                    pytorch_model=self.ts2vec.net,
                    artifact_path="ts2vec_model"
                )

        self.next(self.extract_representations)

    @step
    def extract_representations(self):
        """Extract feature representations using the trained TS2Vec encoder."""
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        X, _, _ = load_processed_data(self.window_data_path)

        # Get TS2Vec embeddings
        self.train_repr = self.ts2vec.encode(X[self.train_idx].astype(np.float32), encoding_window="full_series")
        self.val_repr = self.ts2vec.encode(X[self.val_idx  ].astype(np.float32), encoding_window="full_series")
        self.test_repr = self.ts2vec.encode(X[self.test_idx ].astype(np.float32), encoding_window="full_series")

        # Corresponding labels
        self.y_train = self.y[self.train_idx]
        self.y_val = self.y[self.val_idx]
        self.y_test = self.y[self.test_idx]

        print("\nRepresentations computed.")
        print(f"Extracted TS2Vec representations: train_repr shape={self.train_repr.shape}")
        self.next(self.train_classifier)

    @resources(memory=16000)
    @step
    def train_classifier(self):
        """Train a classifier with (reduced) labeled training data on the TS2Vec representations."""
        set_seed(self.seed)

        # subsample labeled training data
        labels = self.y_train
        if self.label_fraction < 1.0:
            tr_idx, _ = train_test_split(
                np.arange(len(labels)),
                train_size=self.label_fraction,
                stratify=labels,
                random_state=0
            )
        else:
            tr_idx = np.arange(len(labels))
    
        tr_loader = build_linear_loaders(self.train_repr[tr_idx], self.y_train[tr_idx],
                                         self.classifier_batch_size, self.device)
        val_loader= build_linear_loaders(self.val_repr, self.y_val,
                                         self.classifier_batch_size, self.device, shuffle=False)

        self.classifier = MLPClassifier(self.train_repr.shape[-1]).to(self.device)
        opt = torch.optim.Adam(self.classifier.parameters(), lr=self.classifier_lr)
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        with mlflow.start_run(run_id=self.mlflow_run_id):
            params = {
                "classifier_model": "MLPClassifier",
                "seed": self.seed,
                "classifier_lr": self.classifier_lr,
                "classifier_epochs": self.classifier_epochs,
                "classifier_batch_size": self.classifier_batch_size,
                "label_fraction": self.label_fraction
            }
            mlflow.log_params(params)
            model, self.best_threshold = train_linear_classifier(self.classifier, tr_loader, val_loader,
                                    opt, self.loss_fn, self.classifier_epochs, self.device)

        print("Classifier training complete.")
        self.next(self.evaluate)

    @step
    def evaluate(self):
        """Evaluate the classifier performance on the test data."""
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        test_loader = build_linear_loaders(self.test_repr, self.y_test,
                                           self.classifier_batch_size, self.device, shuffle=False)

        with mlflow.start_run(run_id=self.mlflow_run_id):
            self.test_accuracy, test_auroc, test_pr_auc, test_f1 = evaluate_classifier(
                model=self.classifier,
                test_loader=test_loader,
                device=self.device,
                threshold=self.best_threshold,
                loss_fn=self.loss_fn
            )
        self.next(self.end)

    @step
    def end(self):
        """Finalize the pipeline."""
        print("=== TS2Vec Training and Classifier Pipeline Complete ===")
        print(f"Final Test Accuracy: {self.test_accuracy:.4f}")
        mlflow.end_run() 
        print("Done!")

if __name__ == "__main__":
    ECGTS2VecFlow()
