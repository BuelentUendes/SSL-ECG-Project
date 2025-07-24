import os, tempfile, logging, json, gc, atexit, signal, sys
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
import mlflow, mlflow.pytorch
from mlflow.tracking import MlflowClient
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from metaflow import FlowSpec, step, Parameter, current, project, resources

from torch_utilities import (
    load_processed_data,
    split_indices_by_participant,
    build_supervised_fingerprint,
    search_encoder_fp,
    ECGDataset,
    set_seed,
    train_one_epoch,
    test,
    EarlyStopping,
)

from models.supervised import (
    Improved1DCNN_v2,
    TCNClassifier,
    TransformerECGClassifier,
)


#  Metaflow pipeline
@project(name="ecg_training_supervised")
class ECGSupervisedFlow(FlowSpec):
    # generic parameters
    mlflow_tracking_uri = Parameter("mlflow_tracking_uri",
                                    default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")  # Changed to http
                                    )
    window_data_path = Parameter("window_data_path",
                                 default="../data/interim/windowed_data.h5"
                                 )
    seed = Parameter("seed", default=42)

    model_type = Parameter("model_type", help="cnn, tcn or transformer",
                           default="cnn"
                           )

    gpu_number = Parameter("gpu", help="Which specific (cuda) gpu number to use (if available)",
                           type=int, default=0)

    # training hyper-parameters
    lr = Parameter("lr", default=1e-4)
    batch_size = Parameter("batch_size", default=32)
    num_epochs = Parameter("num_epochs", default=25)
    patience = Parameter("patience", default=5)
    scheduler_mode = Parameter("scheduler_mode", default="min")
    scheduler_factor = Parameter("scheduler_factor", default=0.1)
    scheduler_patience = Parameter("scheduler_patience", default=2)
    scheduler_min_lr = Parameter("scheduler_min_lr", default=1e-11)
    label_fraction = Parameter("label_fraction", default=1.0,
                               help="Fraction of labelled training windows actually used (0.0-1.0).")

    def __init__(self):
        super().__init__()
        self._active_loaders = []
        # Register cleanup on exit
        atexit.register(self._emergency_cleanup)

    def _emergency_cleanup(self):
        """Emergency cleanup called on exit to ensure all workers are terminated"""
        print("Emergency cleanup: terminating any remaining DataLoader workers...")
        self._force_cleanup_all_workers()

    def _force_cleanup_all_workers(self):
        """Nuclear option: force cleanup of all DataLoader workers"""
        # Clean up tracked loaders
        for loader in self._active_loaders[:]:
            try:
                self._shutdown_loader(loader)
            except:
                pass
        self._active_loaders.clear()

        # Clean up instance loaders
        for attr_name in ['tr_loader', 'va_loader', 'test_loader']:
            if hasattr(self, attr_name):
                try:
                    loader = getattr(self, attr_name)
                    self._shutdown_loader(loader)
                    delattr(self, attr_name)
                except:
                    pass

        # Force garbage collection multiple times
        for _ in range(3):
            gc.collect()

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        print("Emergency cleanup completed")

    def _shutdown_loader(self, loader):
        """Safely shutdown a single DataLoader"""
        if loader is None:
            return

        try:
            # First, try to shutdown the iterator
            if hasattr(loader, '_iterator') and loader._iterator is not None:
                loader._iterator._shutdown_workers()
                # Set to None to prevent reuse
                loader._iterator = None

            # Also try alternative cleanup methods
            if hasattr(loader, '_DataLoader__initialized') and loader._DataLoader__initialized:
                if hasattr(loader, '_workers') and loader._workers:
                    for w in loader._workers:
                        if w.is_alive():
                            w.terminate()
                            w.join(timeout=1.0)
                            if w.is_alive():
                                w.kill()  # Force kill if still alive

        except Exception as e:
            print(f"Warning during loader shutdown: {e}")

    def _create_dataloader(self, dataset, batch_size, shuffle=False):
        """Create DataLoader with proper tracking for cleanup"""
        # Use fewer workers to reduce resource pressure
        num_workers = min(4, max(1, os.cpu_count() // 2))

        loader = DataLoader(
            dataset,
            batch_size,
            shuffle=shuffle,
            persistent_workers=True,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=2,  # Reduce prefetching
            drop_last=False
        )

        # Track this loader for cleanup
        self._active_loaders.append(loader)
        return loader

    def _cleanup_dataloaders(self):
        """Comprehensive DataLoader cleanup with forced worker termination"""
        print("Starting DataLoader cleanup...")

        # Clean up tracked loaders
        for loader in self._active_loaders[:]:
            self._shutdown_loader(loader)
        self._active_loaders.clear()

        # Clean up instance attribute loaders
        for attr_name in ['tr_loader', 'va_loader', 'test_loader']:
            if hasattr(self, attr_name):
                loader = getattr(self, attr_name)
                self._shutdown_loader(loader)
                delattr(self, attr_name)

        # Multiple rounds of garbage collection to ensure cleanup
        for i in range(5):
            collected = gc.collect()
            if collected == 0 and i > 2:
                break
            if collected > 0:
                print(f"GC round {i + 1}: collected {collected} objects")

        # Clear CUDA memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        print("DataLoader cleanup completed")

    @step
    def start(self):
        """Set seed, choose MLflow experiment name, open run."""
        # Clean up any leftover workers from previous runs
        self._force_cleanup_all_workers()

        set_seed(self.seed)

        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.gpu_number}")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        print(f"We are running our models on: {self.device}")
        exp_map = {
            "cnn": "Supervised_CNN",
            "tcn": "Supervised_TCN",
            "transformer": "Supervised_Transformer",
        }
        self.experiment_name = exp_map.get(self.model_type.lower())
        if self.experiment_name is None:
            raise ValueError(f"Unknown model_type '{self.model_type}'")

        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        try:
            run = mlflow.start_run(run_name=current.run_id)
            self.mlflow_run_id = run.info.run_id
        except Exception as e:
            raise RuntimeError(f"MLflow connection failed: {str(e)}")

        print(f"Starting simple training pipeline... MLflow experiment: {self.experiment_name}")
        self.next(self.load_and_split)

    @step
    def load_and_split(self):
        """Load windowed data and produce participant-level train/val/test idx."""
        self.X, y, groups = load_processed_data(
            self.window_data_path,
            label_map={"baseline": 0, "mental_stress": 1}
        )
        self.n_features = self.X.shape[2]
        self.y = y.astype(np.float32)

        tr_idx, val_idx, te_idx = split_indices_by_participant(groups, seed=42)
        self.train_idx, self.val_idx, self.test_idx = tr_idx, val_idx, te_idx
        print(f"windows: train {len(tr_idx)}, val {len(val_idx)}, test {len(te_idx)}")
        self.next(self.train_model)

    def _get_number_parameters(self):
        self.total_parameters = sum(param.numel() for param in self.model.parameters())
        self.total_parameters_trainable = sum(param.numel() for param in self.model.parameters() if param.requires_grad)

    @resources(memory=16000)
    @step
    def train_model(self):
        """
        • Sub-sample training windows according to 'label_fraction'.
        • If all labels are kept: try to re-use an existing run.
        • Otherwise (or if no cached run) train from scratch.
        • Trained model only saved when label_fraction == 1.0.
        """
        set_seed(self.seed)

        # Stratified label-fraction subsampling
        if not (0 < self.label_fraction <= 1):
            raise ValueError("label_fraction must be in (0,1].")

        if self.label_fraction < 1.0:
            tr_idx, _ = train_test_split(
                np.arange(len(self.train_idx)),
                train_size=self.label_fraction,
                stratify=self.y[self.train_idx],
                random_state=0
            )
            sub_train_idx = self.train_idx[tr_idx]
        else:
            sub_train_idx = self.train_idx

        # Build datasets
        tr_ds = ECGDataset(self.X[sub_train_idx], self.y[sub_train_idx])
        va_ds = ECGDataset(self.X[self.val_idx], self.y[self.val_idx])

        # Create loaders with proper tracking
        self.tr_loader = self._create_dataloader(tr_ds, self.batch_size, shuffle=True)
        self.va_loader = self._create_dataloader(va_ds, self.batch_size, shuffle=False)

        # model choice
        if self.model_type.lower() == "cnn":
            self.model = Improved1DCNN_v2().to(self.device)
        elif self.model_type.lower() == "tcn":
            self.model = TCNClassifier().to(self.device)
        else:
            self.model = TransformerECGClassifier().to(self.device)

        # Print out the total parameter count and the total trainable ones
        self._get_number_parameters()
        print(f"Total number of parameters: {self.total_parameters}")
        print(f"Total number of trainable parameters: {self.total_parameters_trainable}")

        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=self.scheduler_mode, factor=self.scheduler_factor,
            patience=self.scheduler_patience, min_lr=self.scheduler_min_lr)

        es = EarlyStopping(self.patience)

        # build fingerprint
        self.fp = build_supervised_fingerprint({
            "model_name": self.model_type.lower(),
            "seed": self.seed,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "patience": self.patience,
            "scheduler_mode": self.scheduler_mode,
            "scheduler_factor": self.scheduler_factor,
            "scheduler_patience": self.scheduler_patience,
            "scheduler_min_lr": self.scheduler_min_lr,
            "label_fraction": self.label_fraction,
        })

        save_artifact = np.isclose(self.label_fraction, 1.0)  # only log full-label runs
        reused = False
        if save_artifact:
            run_id = search_encoder_fp(self.fp, self.experiment_name,
                                       self.mlflow_tracking_uri)
            if run_id:
                # Re-use existing model
                print(f"Re-using cached encoder {run_id}")
                uri = f"runs:/{run_id}/supervised_model"
                self.model = mlflow.pytorch.load_model(uri, map_location=self.device)
                reused = True

        # train if no matching cached model found or using partial labels
        if not reused:
            print("Training model from scratch...")
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            with mlflow.start_run(run_id=self.mlflow_run_id):
                mlflow.log_params(self.fp | {"optimizer": "Adam",
                                             "loss_fn": "BCEWithLogitsLoss"})

                best_t = 0.5
                best_f1 = -1.0
                for ep in range(1, self.num_epochs + 1):
                    print(f"\nEpoch {ep}/{self.num_epochs}")
                    val_loss, best_t, best_f1 = train_one_epoch(
                        self.model, self.tr_loader, self.va_loader,
                        optimizer, loss_fn,
                        self.device, ep,
                        best_threshold_so_far=best_t,
                        best_f1_so_far=best_f1,
                        log_interval=100
                    )
                    scheduler.step(val_loss)
                    es(val_loss)
                    if es.early_stop:
                        print("Early stopping triggered")
                        break

                self.best_threshold = best_t
                mlflow.log_param("chosen_threshold", best_t)

                # save only when using 100 % labels
                if save_artifact:
                    mlflow.pytorch.log_model(self.model,
                                             artifact_path="supervised_model")

        # Clean up training dataloaders immediately after training
        self._cleanup_dataloaders()
        self.next(self.evaluate)

    @step
    def evaluate(self):
        """Test-set evaluation."""
        set_seed(self.seed)

        te_ds = ECGDataset(self.X[self.test_idx], self.y[self.test_idx])
        self.test_loader = self._create_dataloader(te_ds, self.batch_size, shuffle=False)

        loss_fn = nn.BCEWithLogitsLoss()
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        with mlflow.start_run(run_id=self.mlflow_run_id):
            mlflow.log_params(self.fp | {
                "optimizer": "Adam",
                "loss_fn": "BCEWithLogitsLoss",
            })
            loss, self.acc, auroc, prauc, f1 = test(
                self.model,
                self.test_loader,
                self.device,
                threshold=self.best_threshold,
                loss_fn=loss_fn,
            )

        print(f"Test accuracy: {self.acc:.3f}")
        # Clean up test dataloader immediately after evaluation
        self._cleanup_dataloaders()
        self.next(self.end)

    @step
    def end(self):
        print(f"Supervised training flow with {self.model_type} complete.")
        print(f"Test accuracy: {self.acc:.4f}")

        # Final comprehensive cleanup
        self._force_cleanup_all_workers()

        mlflow.end_run()
        print("Done!")


if __name__ == "__main__":
    flow = ECGSupervisedFlow()
    try:
        flow.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user - cleaning up...")
        flow._force_cleanup_all_workers()
        sys.exit(1)
    except Exception as e:
        print(f"\nError occurred: {e}")
        flow._force_cleanup_all_workers()
        raise
    finally:
        # Ensure cleanup happens no matter what
        if hasattr(flow, '_force_cleanup_all_workers'):
            flow._force_cleanup_all_workers()