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
import tempfile, os
from models.ts2vec_soft import (TS2Vec_soft, save_sim_mat, densify, train_linear_classifier, evaluate_classifier,
                        build_fingerprint, search_encoder_fp, compute_soft_labels, build_linear_loaders)

from torch_utilities import load_processed_data, set_seed, split_indices_by_participant

from models.supervised import (
    LinearClassifier,
    MLPClassifier,
)


# Metaflow Pipeline for Soft TS2Vec pretraining and classifier fine-tuning
@project(name="ecg_training_ts2vec_soft")
class ECGTS2VecFlow(FlowSpec):
    # MLflow and data parameters
    mlflow_tracking_uri = Parameter("mlflow_tracking_uri", 
                                    default=os.getenv("MLFLOW_TRACKING_URI", "https://127.0.0.1:5000"))
    window_data_path = Parameter("window_data_path",
                                 default="../data/interim/windowed_data.h5")
    seed = Parameter("seed", default=42)
    
    # Soft TS2Vec hyperparameters
    ts2vec_epochs = Parameter("ts2vec_epochs", help="Epochs for TS2Vec pretraining", default=50)
    ts2vec_lr = Parameter("ts2vec_lr", help="Learning rate for TS2Vec", default=0.001)
    ts2vec_batch_size = Parameter("ts2vec_batch_size", help="Batch size for TS2Vec", default=8)
    ts2vec_output_dims = Parameter("ts2vec_output_dims", help="Representation dimension (Co)", default=320)
    ts2vec_hidden_dims = Parameter("ts2vec_hidden_dims", help="Hidden dimension (Ch)", default=64)
    ts2vec_depth = Parameter("ts2vec_depth", help="Depth (# dilated conv blocks)", default=10)
    ts2vec_max_train_length = Parameter("ts2vec_max_train_length", help="Max training length", default=5000)
    ts2vec_temporal_unit = Parameter("ts2vec_temporal_unit", help="Temporal unit for hierarchical pooling", default=0)

    # Soft contrastive learning hyperparameters
    ts2vec_dist_type = Parameter("ts2vec_dist_type",
                                 help="Distance metric for soft labels (DTW, EUC, COS, TAM, GAK)",
                                 default="EUC")
    ts2vec_tau_inst = Parameter("ts2vec_tau_inst",
                                help="Temperature parameter tau_inst for soft instance CL", 
                                default=50.0)
    ts2vec_tau_temp = Parameter("ts2vec_tau_temp",
                                help="Temperature parameter tau_temp for soft temporal CL", 
                                default=2.5)
    ts2vec_alpha = Parameter("ts2vec_alpha",
                             help="Alpha for densification of soft labels", 
                             default=0.5)
    ts2vec_lambda = Parameter("ts2vec_lambda",
                              help="Weight lambda for instance vs temporal CL", 
                              default=0.5)

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
        mlflow.set_experiment("SoftTS2Vec")
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
        """
        If a TS2Vec encoder with identical hyper-parameters is already stored in
        MLflow, load it. Otherwise train and log it.
        """
        torch.cuda.empty_cache(), torch.cuda.ipc_collect()
        set_seed(self.seed)
        mlflow.set_experiment("SoftTS2Vec")
        
        X, _, _ = load_processed_data(self.window_data_path)
        X_train = X[self.train_idx].astype(np.float32)
        del X  
        
        # MLflow lookup
        fp = build_fingerprint({
            "model_name": "TS2Vec_soft",
            "seed": self.seed,
            "ts2vec_epochs": self.ts2vec_epochs,
            "ts2vec_output_dims": self.ts2vec_output_dims,
            "ts2vec_hidden_dims": self.ts2vec_hidden_dims,
            "ts2vec_depth": self.ts2vec_depth,
            "ts2vec_dist_type": self.ts2vec_dist_type,
            "ts2vec_tau_inst": self.ts2vec_tau_inst,
            "ts2vec_tau_temp": self.ts2vec_tau_temp,
            "ts2vec_alpha": self.ts2vec_alpha,
            "ts2vec_lambda": self.ts2vec_lambda,
            "ts2vec_max_train_length": self.ts2vec_max_train_length,
        })
        run_id = search_encoder_fp(fp, experiment_name="SoftTS2Vec",
                                   tracking_uri=self.mlflow_tracking_uri)

        if run_id:
            # Load from MLflow                                               
            print(f"encoder found: re-using run {run_id}")
            uri = f"runs:/{run_id}/ts2vec_soft_model"
            net = mlflow.pytorch.load_model(uri, map_location=self.device)

            self.ts2vec_soft = TS2Vec_soft(
                input_dims=self.n_features, output_dims=self.ts2vec_output_dims,
                hidden_dims=self.ts2vec_hidden_dims, depth=self.ts2vec_depth,
                device=self.device, lr=self.ts2vec_lr, batch_size=self.ts2vec_batch_size,
                lambda_=self.ts2vec_lambda, tau_temp=self.ts2vec_tau_temp,
                max_train_length=self.ts2vec_max_train_length,
                temporal_unit=self.ts2vec_temporal_unit,
                soft_instance=True, soft_temporal=True,
            )
            self.ts2vec_soft.net = self.ts2vec_soft._net = net

        # train
        else:                                                      
            print("no cached encoder: training …")

            soft_labels = compute_soft_labels(
                X_train, self.ts2vec_tau_inst, self.ts2vec_alpha,
                self.ts2vec_dist_type, self.ts2vec_max_train_length
            )

            self.ts2vec_soft = TS2Vec_soft(
                input_dims=self.n_features, output_dims=self.ts2vec_output_dims,
                hidden_dims=self.ts2vec_hidden_dims, depth=self.ts2vec_depth,
                device=self.device, lr=self.ts2vec_lr, batch_size=self.ts2vec_batch_size,
                lambda_=self.ts2vec_lambda, tau_temp=self.ts2vec_tau_temp,
                max_train_length=self.ts2vec_max_train_length,
                temporal_unit=self.ts2vec_temporal_unit,
                soft_instance=True, soft_temporal=True,
            )
            
            # Log and run pretraining
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            with mlflow.start_run(run_id=self.mlflow_run_id):
                mlflow.log_params(fp)

                run_dir = tempfile.mkdtemp(prefix="ts2vec_")
                self.ts2vec_soft.fit(
                    X_train, soft_labels,
                    run_dir=run_dir,
                    n_epochs=self.ts2vec_epochs,
                    verbose=True
                )
                # log pytorch model to MLflow
                mlflow.pytorch.log_model(
                    pytorch_model=self.ts2vec_soft.net,
                    artifact_path="ts2vec_soft_model"
                )

        self.next(self.extract_representations)  

    @step
    def extract_representations(self):
        """Extract feature representations using the trained TS2Vec encoder."""
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        X, _, _ = load_processed_data(self.window_data_path)
        self.train_repr = self.ts2vec_soft.encode(X[self.train_idx].astype(np.float32), encoding_window="full_series")
        self.val_repr = self.ts2vec_soft.encode(X[self.val_idx].astype(np.float32), encoding_window="full_series")
        self.test_repr = self.ts2vec_soft.encode(X[self.test_idx].astype(np.float32), encoding_window="full_series")

        # keep y arrays for the next step 
        self.y_train = self.y[self.train_idx]
        self.y_val = self.y[self.val_idx]
        self.y_test = self.y[self.test_idx]

        print("Representations computed.")
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

        self.classifier = LinearClassifier(self.train_repr.shape[-1]).to(self.device)
        opt = torch.optim.AdamW(self.classifier.parameters(), lr=self.classifier_lr)
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        with mlflow.start_run(run_id=self.mlflow_run_id):
            params = {
                "classifier_model": "LinearClassifier",
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
        print("=== Soft TS2Vec Training and Classifier Pipeline Complete ===")
        print(f"Final Test Accuracy: {self.test_accuracy:.4f}")
        mlflow.end_run() 
        print("Done!")

if __name__ == "__main__":
    ECGTS2VecFlow()
