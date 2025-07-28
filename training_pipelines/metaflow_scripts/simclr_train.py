import os, mlflow, torch, numpy as np, torch.nn as nn, torch.optim as optim
import tempfile
import mlflow.pytorch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from metaflow import FlowSpec, step, Parameter, current, project, resources
from utils.helper_paths import SAVED_MODELS_PATH, DATA_PATH
from utils.torch_utilities import (load_processed_data, split_indices_by_participant,
                             set_seed, create_directory)

from models.simclr import (get_simclr_model, NTXentLoss, simclr_data_loaders, pretrain_one_epoch,
                    encode_representations, train_linear_classifier,
                    evaluate_classifier, show_shape, build_simclr_fingerprint, search_encoder_fp)

from models.supervised import (
    LinearClassifier,
    MLPClassifier,
)

@project(name="ecg_training_simclr")
class ECGSimCLRFlow(FlowSpec):

    # MLflow and data parameters
    mlflow_tracking_uri = Parameter("mlflow_tracking_uri",
                                    default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    window_data_path = Parameter("window_data_path",
                                 default=f"{os.path.join(DATA_PATH, 'interim', 'windowed_data.h5')}")

    seed = Parameter("seed", default=42)

    # SimCLR pre‑training
    epochs = Parameter("epochs", default=100)
    lr = Parameter("lr",default=1e-3)
    batch_size = Parameter("batch_size", default=256)
    temperature = Parameter("temperature", default=0.2)

    # linear classifier
    clf_epochs = Parameter("clf_epochs", default=25)
    clf_lr = Parameter("clf_lr", default=1e-4)
    clf_batch_size = Parameter("clf_batch_size", default=32)
    label_fraction = Parameter("label_fraction", default=1.0)

    @step
    def start(self):
        """Initialize MLflow and set seed."""
        set_seed(self.seed)
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        # Set a custom MLflow experiment name
        mlflow.set_experiment("SimCLR")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            run = mlflow.start_run(run_name=current.run_id)
            self.mlflow_run_id = run.info.run_id
        except Exception as e:
            raise RuntimeError(f"MLflow connection failed: {str(e)}")

        print(f"Using device: {self.device}")
        self.next(self.preprocess_data)

    @step
    def preprocess_data(self):
        """
        Load windowed ECG once, keep only indices for later steps.
        """
        X, y, groups = load_processed_data(
            self.window_data_path,
            label_map={"baseline": 0, "mental_stress": 1},
        )

        tr, va, te = split_indices_by_participant(groups, seed=42)
        self.train_idx, self.val_idx, self.test_idx = tr, va, te
        self.y = y.astype(np.float32)
        self.win_len = X.shape[1]     

        print(f"windows: train {len(tr)}, val {len(va)}, test {len(te)}")
        self.next(self.pretrain_simclr)

    @resources(memory=16000)
    @step
    def pretrain_simclr(self):
        """
        Pre-train SimCLR. If an identical encoder already exists in MLflow,
        load and reuse it; otherwise train from scratch and log it.
        """
        torch.cuda.empty_cache(), torch.cuda.ipc_collect()
        set_seed(self.seed)
        mlflow.set_experiment("SimCLR")
        
        # Build fingerprint and MLflow lookup
        fp = build_simclr_fingerprint({
            "model_name": "SimCLR",
            "seed":        self.seed,
            "epochs":      self.epochs,
            "lr":          self.lr,
            "batch_size":  self.batch_size,
            "temperature": self.temperature,
            "window_len":  self.win_len,
        })
        run_id = search_encoder_fp(fp, "SimCLR", self.mlflow_tracking_uri)
        
        self.model = get_simclr_model(window=self.win_len, device=self.device)
        
        if run_id:
            # Re-use existing encoder
            print(f"encoder found: re-using run {run_id}")
            uri = f"runs:/{run_id}/ssl_model"
            ckpt_dir = mlflow.artifacts.download_artifacts(uri)
            ckpt_path = os.path.join(ckpt_dir, "simclr_encoder.pt")
            self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        
        else:
            # train from scratch
            print("no cached encoder: training …")
            X, _, _ = load_processed_data(self.window_data_path)
            X_tr = X[self.train_idx].astype(np.float32)
            X_va = X[self.val_idx  ].astype(np.float32)
            del X

            loss_fn = NTXentLoss(self.batch_size, self.temperature)
            opt = optim.AdamW(self.model.parameters(),
                                   lr=self.lr, weight_decay=1e-4)
            tr_dl, _ = simclr_data_loaders(X_tr, X_va, self.batch_size)

            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            with mlflow.start_run(run_id=self.mlflow_run_id):
                mlflow.log_params(fp)
                for ep in range(1, self.epochs + 1):
                    tr_loss = pretrain_one_epoch(
                        self.model, tr_dl, loss_fn, opt, self.device)
                    mlflow.log_metric("ssl_train_loss", tr_loss, step=ep)
                    if ep == 1 or ep % 25 == 0:
                        print(f"Epoch {ep}/{self.epochs}: loss={tr_loss:.4f}")

                # save encoder weights
                ckpt = os.path.join(tempfile.mkdtemp(), "simclr_encoder.pt")
                torch.save(self.model.state_dict(), ckpt)
                mlflow.log_artifact(ckpt, artifact_path="ssl_model")
                
        self.next(self.extract_representations)

    @step
    def extract_representations(self):
        """Extract feature representations using the trained SimCLR encoder."""
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        X, _, _ = load_processed_data(self.window_data_path)
        
        with torch.no_grad():
            self.train_repr = encode_representations(
                self.model, X[self.train_idx].astype(np.float32),
                self.batch_size, self.device)
            self.val_repr   = encode_representations(
                self.model, X[self.val_idx].astype(np.float32),
                self.batch_size, self.device)
            self.test_repr  = encode_representations(
                self.model, X[self.test_idx].astype(np.float32),
                self.batch_size, self.device)

        show_shape("repr", (self.train_repr, self.val_repr, self.test_repr))
        self.next(self.train_classifier)

    @resources(memory=16000)
    @step
    def train_classifier(self):
        """Train a classifier with (reduced) labeled training data on the SimCLR representations."""
        set_seed(self.seed)
        
        # subsample labelled training windows
        labels = self.y[self.train_idx]
        if self.label_fraction < 1.0:
            tr_idx, _ = train_test_split(
                np.arange(len(labels)),
                train_size=self.label_fraction,
                stratify=labels,
                random_state=0
            )
        else:
            tr_idx = np.arange(len(labels))
        
        Xtr, ytr = self.train_repr[tr_idx], labels[tr_idx]

        self.classifier = LinearClassifier(Xtr.shape[1]).to(self.device)
        self.loss_fn = nn.BCEWithLogitsLoss()
        opt = optim.AdamW(self.classifier.parameters(), lr=self.clf_lr)

        tr_dl = DataLoader(
            TensorDataset(torch.from_numpy(Xtr).float(),
                          torch.from_numpy(ytr).float()),
            batch_size=self.clf_batch_size, shuffle=True)
        va_dl = DataLoader(
            TensorDataset(torch.from_numpy(self.val_repr).float(),
                          torch.from_numpy(self.y[self.val_idx]).float()),
            batch_size=self.clf_batch_size, shuffle=False)

        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        with mlflow.start_run(run_id=self.mlflow_run_id):
            mlflow.log_params({
                "classifier_model": "LinearClassifier",
                "seed": self.seed,
                "classifier_lr": self.clf_lr,
                "classifier_epochs": self.clf_epochs,
                "classifier_batch_size": self.clf_batch_size,
                "label_fraction": self.label_fraction,
            })
    
            model, self.best_threshold = train_linear_classifier(self.classifier, tr_dl, va_dl,
                                    opt, self.loss_fn, self.clf_epochs, self.device)
        
        print("Classifier training complete.")
        self.next(self.evaluate)

    @step
    def evaluate(self):
        """Evaluate the classifier on the held-out test windows."""
        te_dl = DataLoader(
            TensorDataset(torch.from_numpy(self.test_repr).float(),
                          torch.from_numpy(self.y[self.test_idx]).float()),
            batch_size=self.clf_batch_size, shuffle=False)
        
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        with mlflow.start_run(run_id=self.mlflow_run_id):
            self.test_acc, test_auroc, test_pr_auc, test_f1 = evaluate_classifier(
                model=self.classifier,
                test_loader=te_dl,
                device=self.device,
                threshold=self.best_threshold,
                loss_fn=self.loss_fn
            )
        self.next(self.end)

    @step
    def end(self):
        print(f"=== SimCLR pipeline done – test accuracy {self.test_acc:.4f} ===")
        mlflow.end_run() 
        print("Done!")
        
if __name__ == "__main__":
    ECGSimCLRFlow()
