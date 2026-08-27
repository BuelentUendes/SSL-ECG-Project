"""Quick sanity-check: train InfoTS (unsupervised) on a UCR dataset and evaluate
with a logistic regression on the learned representations."""

import argparse
import sys
import os
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sktime.datasets import load_UCR_UEA_dataset

from models.infots import InfoTS
from utils.torch_utilities import set_seed

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "infots_results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_dataset(name: str):
    print(f"Loading {name} ...")
    X_train, y_train = load_UCR_UEA_dataset(name, split="train", return_type="numpy3D")
    X_test, y_test = load_UCR_UEA_dataset(name, split="test", return_type="numpy3D")

    # InfoTS expects [n_samples, n_timepoints, n_channels]
    X_train = X_train.transpose(0, 2, 1).astype(np.float32)
    X_test = X_test.transpose(0, 2, 1).astype(np.float32)

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    print(f"  train {X_train.shape}  test {X_test.shape}  classes {le.classes_}")
    return X_train, y_train, X_test, y_test


def chunk_data(X: np.ndarray, y: np.ndarray, n_chunks: int):
    """Split each sample along the time axis into n_chunks equal pieces.

    Returns:
        X_chunked: [n_samples * n_chunks, chunk_len, n_features]
        y_chunked: [n_samples * n_chunks]  (label replicated per chunk)
        chunk_len: int
    """
    n_samples, seq_len, n_features = X.shape
    chunk_len = seq_len // n_chunks
    usable_len = chunk_len * n_chunks  # drop remainder

    X_cut = X[:, :usable_len, :]                                      # [N, usable, F]
    X_chunked = X_cut.reshape(n_samples, n_chunks, chunk_len, n_features)
    X_chunked = X_chunked.reshape(n_samples * n_chunks, chunk_len, n_features)
    y_chunked = np.repeat(y, n_chunks)

    print(f"  Chunking: {seq_len} → {n_chunks} x {chunk_len} timesteps "
          f"(dropped {seq_len - usable_len} remainder steps)")
    return X_chunked, y_chunked, chunk_len


def aggregate_chunk_repr(repr_chunked: np.ndarray, n_chunks: int) -> np.ndarray:
    """Mean-pool chunk representations back to one vector per original sample."""
    n_total, repr_dim = repr_chunked.shape
    n_samples = n_total // n_chunks
    return repr_chunked.reshape(n_samples, n_chunks, repr_dim).mean(axis=1)


def parse_args():
    parser = argparse.ArgumentParser(description="InfoTS unsupervised pretraining + linear eval on UCR datasets")
    parser.add_argument("--seed", type=int, default=3,
                              help="Random seed for reproducibility")
    parser.add_argument("--dataset", type=str, default="ECG5000",
                        help="UCR/UEA dataset name (default: ECG200)")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of pretraining epochs (default: 30)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size (default: 16)")
    parser.add_argument("--output_dims", type=int, default=64,
                        help="Encoder output dimensions (default: 64)")
    parser.add_argument("--hidden_dims", type=int, default=32,
                        help="Hidden dimensions (default: 32)")
    parser.add_argument("--depth", type=int, default=5,
                        help="Number of dilated-conv blocks (default: 5)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Encoder learning rate (default: 1e-3)")
    parser.add_argument("--meta_lr", type=float, default=1e-2,
                        help="Meta-head learning rate (default: 1e-2)")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate (default: 0.1)")
    parser.add_argument("--aug_p1", type=float, default=0.2,
                        help="Augmentation probability 1 (default: 0.2)")
    parser.add_argument("--aug_p2", type=float, default=0.0,
                        help="Augmentation probability 2 (default: 0.0)")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="InfoTS beta (default: 1.0)")
    parser.add_argument("--split_number", type=int, default=8,
                        help="Number of local segments (default: 4)")
    parser.add_argument("--meta_epoch", type=int, default=5,
                        help="Meta update frequency in epochs (default: 5)")
    parser.add_argument("--meta_beta", type=float, default=1.0,
                        help="Meta beta (default: 1.0)")
    parser.add_argument("--supervised_meta", action="store_true", default=False,
                        help="Use supervised meta-learning (requires labels during pretraining, default: False)")
    parser.add_argument("--chunk", action="store_true", default=False,
                        help="Split each sample into --n_chunks equal pieces before training")
    parser.add_argument("--n_chunks", type=int, default=4,
                        help="Number of equal chunks to split each sample into (default: 4)")
    parser.add_argument("--device", type=str, default=None,
                        help="Force device: 'cpu', 'cuda', or 'mps'. "
                             "Default: auto-detect. Use 'cpu' for bitwise reproducibility.")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed, deterministic=True, infots_algo=True)

    X_train, y_train, X_test, y_test = load_dataset(args.dataset)

    if args.chunk:
        print(f"\nChunking into {args.n_chunks} pieces ...")
        X_train_fit, y_train_fit, chunk_len = chunk_data(X_train, y_train, args.n_chunks)
        X_test_fit, _, _ = chunk_data(X_test, y_test, args.n_chunks)
    else:
        X_train_fit, y_train_fit = X_train, y_train
        X_test_fit = X_test
        chunk_len = None

    n_samples, seq_len, n_features = X_train_fit.shape

    if args.device is not None:
        device = args.device
    else:
        device = (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
    print(f"\nDevice: {device}  |  train={n_samples}  seq_len={seq_len}  feat={n_features}")

    model = InfoTS(
        input_dims=n_features,
        output_dims=args.output_dims,
        hidden_dims=args.hidden_dims,
        depth=args.depth,
        device=device,
        lr=args.lr,
        meta_lr=args.meta_lr,
        batch_size=args.batch_size,
        dropout=args.dropout,
        aug_p1=args.aug_p1,
        aug_p2=args.aug_p2,
    )

    mode = "supervised" if args.supervised_meta else "unsupervised"
    chunk_info = f"  chunk_len={chunk_len} ({args.n_chunks} chunks)" if args.chunk else ""
    print(f"\nPre-training InfoTS for {args.epochs} epochs ({mode}){chunk_info} ...")
    model.fit(
        train_data=X_train_fit,
        n_epochs=args.epochs,
        verbose=True,
        supervised_meta=args.supervised_meta,
        train_labels=y_train_fit if args.supervised_meta else None,
        beta=args.beta,
        split_number=args.split_number,
        meta_epoch=args.meta_epoch,
        meta_beta=args.meta_beta,
        results_save_path=RESULTS_DIR,
    )

    print("\nEncoding representations ...")
    train_repr = model.encode(X_train_fit, batch_size=32)
    test_repr = model.encode(X_test_fit, batch_size=32)

    if args.chunk:
        # mean-pool chunk representations → one vector per original sample
        train_repr = aggregate_chunk_repr(train_repr, args.n_chunks)
        test_repr = aggregate_chunk_repr(test_repr, args.n_chunks)

    print(f"  train_repr {train_repr.shape}  test_repr {test_repr.shape}")

    scaler = StandardScaler()
    train_repr_s = scaler.fit_transform(train_repr)
    test_repr_s = scaler.transform(test_repr)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(train_repr_s, y_train)

    y_pred = clf.predict(test_repr_s)
    y_prob = clf.predict_proba(test_repr_s)

    acc = accuracy_score(y_test, y_pred)

    n_classes = y_prob.shape[1]
    if n_classes == 2:
        auc = roc_auc_score(y_test, y_prob[:, 1])
    else:
        auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")

    print(f"\n{'=' * 45}")
    print(f"  Dataset       : {args.dataset}")
    if args.chunk:
        print(f"  Chunking      : {args.n_chunks} x {chunk_len} timesteps (orig {X_train.shape[1]})")
    print(f"  Test Accuracy : {acc:.4f}")
    print(f"  Test AUROC    : {auc:.4f}")
    print(f"{'=' * 45}")


if __name__ == "__main__":
    main()