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

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from metaflow import FlowSpec, step, Parameter, current, project, resources

from torch_utilities import load_processed_data, split_indices_by_participant, set_seed

from models.tstcc import data_generator_from_arrays, Trainer, base_Model, TC, Config as ECGConfig, \
    train_linear_classifier, evaluate_classifier, encode_representations, show_shape, build_tstcc_fingerprint, search_encoder_fp, \
        build_linear_loaders

from models.supervised import (
    LinearClassifier,
    MLPClassifier,
)
        
@project(name="ecg_training_tstcc")
class ECGTSTCCFlow(FlowSpec):

    # MLflow and data parameters
    mlflow_tracking_uri = Parameter("mlflow_tracking_uri",
                                    default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    window_data_path = Parameter("window_data_path",
                                 default="../data/interim/windowed_data.h5")
    seed = Parameter("seed", default=42)

    # TS-TCC pretraining
    tcc_epochs = Parameter("tcc_epochs", default=40)
    tcc_lr = Parameter("tcc_lr", default=3e-4)
    tcc_batch_size = Parameter("tcc_batch_size", default=128)

    pretrain_all_conditions = Parameter(
        "pretrain_all_conditions",
        default=False,
        help="If True, uses all available conditions for SSL pretraining. If False, uses only baseline + mental_stress."
    )

    # temporal contrasting
    tc_timesteps = Parameter("tc_timesteps", default=70)
    tc_hidden_dim = Parameter("tc_hidden_dim", default=128)

    # contextual contrasting
    cc_temperature = Parameter("cc_temperature", default=0.07)
    cc_use_cosine = Parameter("cc_use_cosine", default=True)

    # classifier fine-tuning
    classifier_epochs = Parameter("classifier_epochs", default=25)
    classifier_lr = Parameter("classifier_lr", default=1e-4)
    classifier_batch_size = Parameter("classifier_batch_size", default=32)
    label_fraction = Parameter("label_fraction", default=1.0)

    @step
    def start(self):
        """Initialize MLflow and set seed."""
        set_seed(self.seed)
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        # Set a custom MLflow experiment name
        mlflow.set_experiment("TSTCC")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            run = mlflow.start_run(run_name=current.run_id)
            self.mlflow_run_id = run.info.run_id
        except Exception as e:
            raise RuntimeError(f"MLflow connection failed: {str(e)}")

        logging.info(f"MLflow experiment 'SoftTSTCC' (run: {self.mlflow_run_id})")
        print(f"Using device: {self.device}")
        self.next(self.preprocess_data)

    @step
    def preprocess_data(self):
        """
        Load the windowed processed data, create participant split indices,
        persist only those indices to the next step.
        """
        # X, y, groups = load_processed_data(self.window_data_path,
        #                                    label_map={"baseline": 0, "mental_stress": 1})
        
        # # split
        # train_idx, val_idx, test_idx = split_indices_by_participant(groups, seed=42)
        
        # # store artifacts 
        # self.train_idx, self.val_idx, self.test_idx = train_idx, val_idx, test_idx
        # self.y = y.astype(np.float32)                   
        # self.n_features = X.shape[2]          

        # print(f"windows: train {len(train_idx)}, val {len(val_idx)}, test {len(test_idx)}")
        # self.next(self.train_tstcc)
        
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
        X, y, groups = load_processed_data(self.window_data_path, label_map=self.label_map)
        
        # split
        train_idx, val_idx, test_idx = split_indices_by_participant(groups, seed=42)
        
        # store artifacts 
        self.train_idx, self.val_idx, self.test_idx = train_idx, val_idx, test_idx
        self.y = y.astype(np.float32)                   
        self.n_features = X.shape[2]    
        
        # Keep track of which samples belong to the binary downstream task
        # 0 = baseline, 1 = mental_stress
        self.downstream_label_mask = {
            "train": np.isin(self.y[self.train_idx], [0, 1]),
            "val":   np.isin(self.y[self.val_idx], [0, 1]),
            "test":  np.isin(self.y[self.test_idx], [0, 1])
        }      

        print(f"windows: train {len(train_idx)}, val {len(val_idx)}, test {len(test_idx)}")
        self.next(self.train_tstcc)

    @resources(memory=16000)
    @step
    def train_tstcc(self):
        """
        Pre-train TS-TCC.  
        If an identical encoder is already logged in MLflow, re-use it;
        otherwise train from scratch and log the checkpoint.
        """
        torch.cuda.empty_cache(), torch.cuda.ipc_collect()
        set_seed(self.seed)
        mlflow.set_experiment("TSTCC")

        # X, _, _ = load_processed_data(self.window_data_path)
        X, _, _ = load_processed_data(self.window_data_path, label_map=self.label_map)
        X_train = X[self.train_idx].astype(np.float32)
        X_val = X[self.val_idx].astype(np.float32)
        X_test = X[self.test_idx].astype(np.float32)
        del X

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
            # Re-use existing encoder
            print(f"encoder found: re-using run {run_id}")
            uri = f"runs:/{run_id}/tstcc_model"
            ckpt_dir = mlflow.artifacts.download_artifacts(uri)
            ckpt_path = os.path.join(ckpt_dir, "tstcc.pt")

            # build fresh model objects with identical configs
            self.configs = ECGConfig()
            self.configs.num_epoch  = self.tcc_epochs
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
                X_val,  self.y[self.val_idx],
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
                
        self.next(self.extract_representations)

    @step
    def extract_representations(self):
        """Extract feature representations using the trained TS-TCC encoder."""
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        set_seed(self.seed)
        # X, _, _ = load_processed_data(self.window_data_path)
        X, _, _ = load_processed_data(self.window_data_path, label_map=self.label_map)
        self.model.eval()
        self.temporal_contr_model.eval()
        
        self.train_repr, _ = encode_representations(
            X[self.train_idx], self.y[self.train_idx],
            self.model, self.temporal_contr_model,
            self.tcc_batch_size, self.device
        )
        
        self.val_repr, _ = encode_representations(
            X[self.val_idx], self.y[self.val_idx],
            self.model, self.temporal_contr_model,
            self.tcc_batch_size, self.device
        )
        
        self.test_repr, _ = encode_representations(
            X[self.test_idx], self.y[self.test_idx],
            self.model, self.temporal_contr_model,
            self.tcc_batch_size, self.device
        )
        
        # Only keep baseline and mental_stress samples
        self.train_repr = self.train_repr[self.downstream_label_mask["train"]]
        self.y_train    = self.y[self.train_idx][self.downstream_label_mask["train"]]

        self.val_repr = self.val_repr[self.downstream_label_mask["val"]]
        self.y_val    = self.y[self.val_idx][self.downstream_label_mask["val"]]

        self.test_repr = self.test_repr[self.downstream_label_mask["test"]]
        self.y_test    = self.y[self.test_idx][self.downstream_label_mask["test"]]
        
        # # keep y arrays for the next step 
        # self.y_train = self.y[self.train_idx]
        # self.y_val = self.y[self.val_idx]
        # self.y_test = self.y[self.test_idx]

        print(f"train_repr shape = {self.train_repr.shape}")
        show_shape("val_repr / test_repr",
                   (self.val_repr, self.test_repr))
        self.next(self.train_classifier)

    @resources(memory=16000)
    @step
    def train_classifier(self):
        """Train a classifier with (reduced) labeled training data on the TS-TCC representations."""
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
        self.next(self.end)

    @step
    def end(self):
        print("=== TS-TCC pipeline complete ===")
        print(f"Test accuracy: {self.test_accuracy:.4f}")
        mlflow.end_run()
        print("Done!")

if __name__ == "__main__":
    ECGTSTCCFlow()
