import os
import sys
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import mlflow
import mlflow.pytorch
import tempfile
import argparse

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from utils.torch_utilities import load_processed_data, split_indices_by_participant, set_seed

from models.tstcc import data_generator_from_arrays, Trainer, base_Model, TC, Config as ECGConfig, \
    train_linear_classifier, evaluate_classifier, encode_representations, show_shape, build_tstcc_fingerprint, \
    search_encoder_fp, \
    build_linear_loaders

from models.supervised import (
    LinearClassifier,
    MLPClassifier,
)


class ECGTSTCCTrainer:
    def __init__(self, config):
        # Set all parameters from config
        self.mlflow_tracking_uri = config.mlflow_tracking_uri
        self.window_data_path = config.window_data_path
        self.seed = config.seed

        # TS-TCC pretraining
        self.tcc_epochs = config.tcc_epochs
        self.tcc_lr = config.tcc_lr
        self.tcc_batch_size = config.tcc_batch_size
        self.pretrain_all_conditions = config.pretrain_all_conditions

        # temporal contrasting
        self.tc_timesteps = config.tc_timesteps
        self.tc_hidden_dim = config.tc_hidden_dim

        # contextual contrasting
        self.cc_temperature = config.cc_temperature
        self.cc_use_cosine = config.cc_use_cosine

        # classifier fine-tuning
        self.classifier_epochs = config.classifier_epochs
        self.classifier_lr = config.classifier_lr
        self.classifier_batch_size = config.classifier_batch_size
        self.label_fraction = config.label_fraction

        # Initialize
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mlflow_run_id = None

    def initialize(self):
        """Initialize MLflow and set seed."""
        set_seed(self.seed)
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_experiment("TSTCC")

        try:
            run = mlflow.start_run(run_name=f"tstcc_training_{self.seed}")
            self.mlflow_run_id = run.info.run_id
        except Exception as e:
            raise RuntimeError(f"MLflow connection failed: {str(e)}")

        logging.info(f"MLflow experiment 'TSTCC' (run: {self.mlflow_run_id})")
        print(f"Using device: {self.device}")

    def preprocess_data(self):
        """
        Load the windowed processed data, create participant split indices,
        persist only those indices to the next step.
        """
        if self.pretrain_all_conditions:
            print("Pretraining with all conditions.")
            # Use full label map
            label_map = {
                "baseline": 0,
                "mental_stress": 1,
                "low_physical_activity": 2,
                "moderate_physical_activity": 3,
                "high_physical_activity": 4
            }
        else:
            print("Pretraining with baseline and mental stress conditions only.")
            # Use task-specific map only
            label_map = {
                "baseline": 0,
                "mental_stress": 1
            }

        self.label_map = label_map
        self.X, y, groups = load_processed_data(self.window_data_path, label_map=self.label_map)

        # split
        train_idx, val_idx, test_idx = split_indices_by_participant(groups, seed=42)

        # store artifacts
        self.train_idx, self.val_idx, self.test_idx = train_idx, val_idx, test_idx
        self.y = y.astype(np.float32)
        self.n_features = self.X.shape[2]

        # Keep track of which samples belong to the binary downstream task
        # 0 = baseline, 1 = mental_stress
        self.downstream_label_mask = {
            "train": np.isin(self.y[self.train_idx], [0, 1]),
            "val": np.isin(self.y[self.val_idx], [0, 1]),
            "test": np.isin(self.y[self.test_idx], [0, 1])
        }

        print(f"windows: train {len(train_idx)}, val {len(val_idx)}, test {len(test_idx)}")

    def train_tstcc(self):
        """
        Pre-train TS-TCC.
        If an identical encoder is already logged in MLflow, re-use it;
        otherwise train from scratch and log the checkpoint.
        """
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()
        set_seed(self.seed)
        mlflow.set_experiment("TSTCC")
        #
        # X, _, _ = load_processed_data(self.window_data_path, label_map=self.label_map)
        X_train = self.X[self.train_idx].astype(np.float32)
        X_val = self.X[self.val_idx].astype(np.float32)
        X_test = self.X[self.test_idx].astype(np.float32)
        # del X

        # Build fingerprint
        fp = build_tstcc_fingerprint({
            "model_name": "TSTCC",
            "seed": self.seed,
            "pretrain_all_conditions": self.pretrain_all_conditions,
            "tcc_epochs": self.tcc_epochs,
            "tcc_lr": self.tcc_lr,
            "tcc_batch_size": self.tcc_batch_size,
            "tc_timesteps": self.tc_timesteps,
            "tc_hidden_dim": self.tc_hidden_dim,
            "cc_temperature": self.cc_temperature,
            "cc_use_cosine": self.cc_use_cosine,
        })

        run_id = search_encoder_fp(fp,
                                   experiment_name="TSTCC",
                                   tracking_uri=self.mlflow_tracking_uri)

        if run_id:
            # Re-use existing encoder
            print(f"encoder found: re-using run {run_id}")
            uri = f"runs:/{run_id}/tstcc_model"
            ckpt_dir = mlflow.artifacts.download_artifacts(uri)
            ckpt_path = os.path.join(ckpt_dir, "tstcc.pt")

            # build fresh model objects with identical configs
            self.configs = ECGConfig()
            self.configs.num_epoch = self.tcc_epochs
            self.configs.batch_size = self.tcc_batch_size
            self.configs.TC.timesteps = self.tc_timesteps
            self.configs.TC.hidden_dim = self.tc_hidden_dim
            self.configs.Context_Cont.temperature = self.cc_temperature
            self.configs.Context_Cont.use_cosine_similarity = self.cc_use_cosine

            self.model = base_Model(self.configs).to(self.device)
            self.temporal_contr_model = TC(self.configs, self.device).to(self.device)

            state = torch.load(ckpt_path, map_location=self.device)
            self.model.load_state_dict(state["encoder"])
            self.temporal_contr_model.load_state_dict(state["tc_head"])

        else:
            # train from scratch
            print("no cached encoder: training from scratch")

            # build Config object
            self.configs = ECGConfig()
            self.configs.num_epoch = self.tcc_epochs
            self.configs.batch_size = self.tcc_batch_size
            self.configs.TC.timesteps = self.tc_timesteps
            self.configs.TC.hidden_dim = self.tc_hidden_dim
            self.configs.Context_Cont.temperature = self.cc_temperature
            self.configs.Context_Cont.use_cosine_similarity = self.cc_use_cosine

            # data loaders
            train_dl, val_dl, test_dl = data_generator_from_arrays(
                X_train, self.y[self.train_idx],
                X_val, self.y[self.val_idx],
                X_test, self.y[self.test_idx],
                self.configs, training_mode="self_supervised"
            )

            # models + optimisers
            self.model = base_Model(self.configs).to(self.device)
            self.temporal_contr_model = TC(self.configs, self.device).to(self.device)
            model_opt = optim.AdamW(self.model.parameters(), lr=self.tcc_lr, weight_decay=3e-4)
            tc_opt = optim.AdamW(self.temporal_contr_model.parameters(), lr=self.tcc_lr, weight_decay=3e-4)

            # MLflow run scope
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            with mlflow.start_run(run_id=self.mlflow_run_id):
                mlflow.log_params(fp)

                run_dir = tempfile.mkdtemp(prefix="tstcc_")
                Trainer(
                    model=self.model,
                    temporal_contr_model=self.temporal_contr_model,
                    model_optimizer=model_opt,
                    temp_cont_optimizer=tc_opt,
                    train_dl=train_dl, valid_dl=val_dl, test_dl=test_dl,
                    device=self.device, config=self.configs,
                    experiment_log_dir=run_dir,
                    training_mode="self_supervised",
                )

                # store checkpoint
                ckpt = os.path.join(run_dir, "tstcc.pt")
                torch.save(
                    {
                        "encoder": self.model.state_dict(),
                        "tc_head": self.temporal_contr_model.state_dict(),
                    },
                    ckpt,
                )
                mlflow.log_artifact(ckpt, artifact_path="tstcc_model")

    def extract_representations(self):
        """Extract feature representations using the trained TS-TCC encoder."""
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        set_seed(self.seed)
        # X, _, _ = load_processed_data(self.window_data_path, label_map=self.label_map)
        self.model.eval()
        self.temporal_contr_model.eval()

        self.train_repr, _ = encode_representations(
            self.X[self.train_idx], self.y[self.train_idx],
            self.model, self.temporal_contr_model,
            self.tcc_batch_size, self.device
        )

        self.val_repr, _ = encode_representations(
            self.X[self.val_idx], self.y[self.val_idx],
            self.model, self.temporal_contr_model,
            self.tcc_batch_size, self.device
        )

        self.test_repr, _ = encode_representations(
            self.X[self.test_idx], self.y[self.test_idx],
            self.model, self.temporal_contr_model,
            self.tcc_batch_size, self.device
        )

        # Only keep baseline and mental_stress samples
        self.train_repr = self.train_repr[self.downstream_label_mask["train"]]
        self.y_train = self.y[self.train_idx][self.downstream_label_mask["train"]]

        self.val_repr = self.val_repr[self.downstream_label_mask["val"]]
        self.y_val = self.y[self.val_idx][self.downstream_label_mask["val"]]

        self.test_repr = self.test_repr[self.downstream_label_mask["test"]]
        self.y_test = self.y[self.test_idx][self.downstream_label_mask["test"]]

        print(f"train_repr shape = {self.train_repr.shape}")
        show_shape("val_repr / test_repr",
                   (self.val_repr, self.test_repr))

    def train_classifier(self):
        """Train a classifier with (reduced) labeled training data on the TS-TCC representations."""
        set_seed(self.seed)

        # subsample labeled training data
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
        val_loader = build_linear_loaders(self.val_repr, self.y_val,
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

    def evaluate(self):
        """Evaluate the classifier on the held-out test windows."""
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

    def run_full_pipeline(self):
        """Run the complete training pipeline."""
        print("=== Starting TS-TCC Training Pipeline ===")

        # Initialize
        self.initialize()

        # Run pipeline steps
        print("Step 1: Preprocessing data...")
        self.preprocess_data()

        print("Step 2: Training TS-TCC...")
        self.train_tstcc()

        print("Step 3: Extracting representations...")
        self.extract_representations()

        print("Step 4: Training classifier...")
        self.train_classifier()

        print("Step 5: Evaluating...")
        self.evaluate()

        # Finish
        print("=== TS-TCC pipeline complete ===")
        print(f"Test accuracy: {self.test_accuracy:.4f}")
        mlflow.end_run()
        print("Done!")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="TS-TCC Training Pipeline")

    # MLflow and data parameters
    parser.add_argument("--mlflow_tracking_uri",
                        default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"),
                        help="MLflow tracking URI")
    parser.add_argument("--window_data_path",
                        default="../data/interim/windowed_data.h5",
                        help="Path to windowed data file")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # TS-TCC pretraining
    parser.add_argument("--tcc_epochs", type=int, default=40,
                        help="Number of TCC epochs")
    parser.add_argument("--tcc_lr", type=float, default=3e-4,
                        help="TCC learning rate")
    parser.add_argument("--tcc_batch_size", type=int, default=128,
                        help="TCC batch size")
    parser.add_argument("--pretrain_all_conditions", action="store_true",
                        help="Use all conditions for SSL pretraining")

    # temporal contrasting
    parser.add_argument("--tc_timesteps", type=int, default=70,
                        help="TC timesteps")
    parser.add_argument("--tc_hidden_dim", type=int, default=128,
                        help="TC hidden dimension")

    # contextual contrasting
    parser.add_argument("--cc_temperature", type=float, default=0.07,
                        help="CC temperature")
    parser.add_argument("--cc_use_cosine", action="store_true", default=True,
                        help="Use cosine similarity for CC")

    # classifier fine-tuning
    parser.add_argument("--classifier_epochs", type=int, default=25,
                        help="Classifier epochs")
    parser.add_argument("--classifier_lr", type=float, default=1e-4,
                        help="Classifier learning rate")
    parser.add_argument("--classifier_batch_size", type=int, default=32,
                        help="Classifier batch size")
    parser.add_argument("--label_fraction", type=float, default=1.0,
                        help="Fraction of labels to use")

    return parser.parse_args()


if __name__ == "__main__":
    # Parse arguments
    config = parse_args()

    # Create trainer and run
    trainer = ECGTSTCCTrainer(config)
    trainer.run_full_pipeline()