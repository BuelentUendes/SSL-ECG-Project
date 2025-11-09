# Beyond Supervision: Evaluating Contrastive Self-Supervised Learning Techniques for Electrocardiogram-Based Mental Stress Detection

**Authors:** [Buelent Uendes](https://buelentuendes.github.io/), [Carlos Laborda Gisbert](https://github.com/Carlos-Laborda), and Mark Hoogendoorn

This project explores mental stress detection using an ECG Dataset compromising 127 participants across 26 different conditions using both supervised and self-supervised learning (SSL) methods. It benchmarks CNNs, TCN, Transformers, and contrastive SSL approaches such as SimCLR, TSTCC, TS2Vec, and InfoTS, with a special focus on label efficiency. It also considers the S3 augmented version of the contrastive SSL approaches.

In the self-supervised setting, once encoders are pre-trained, they are frozen and their learned representations are used to train lightweight linear classifiers.

For the cross-dataset generalization, two additional datasets, StressID and WESAD were considered and the best-performing SSL method (TS-TCC and TS-TCC+S3) were compared using zero-shot, linear-probing (LP), and LP+fine-tuning (LP+FT).

The following figure illustrates the overall experimental setup:

<p align="center">
  <img src="pics/graphical_abstract.png" width="450">
</p>

## Table of Contents

- [Installation and Environment Setup](#installation-and-environment-setup)
- [Project Structure](#project-structure)
- [Running the Preprocessing Pipeline](#running-the-preprocessing-pipeline)
- [Running the Training Pipelines (Locally)](#running-the-training-pipelines-locally)
  - [Running Locally](#running-the-training-pipelines-locally)
  - [Running on DAS6 Cluster](#running-the-training-pipelines-on-das6-cluster)
    - [1. MLflow Server Setup](#1-activate-environment-and-launch-mlflow-on-das6)
    - [2. Remote Access](#2-accessing-mlflow-remotely)
    - [3. SLURM Configuration](#3-slurm-job-configuration)
    - [4. Job Submission](#4-launching-training-jobs)

## Installation and Environment Setup

This project uses a Conda environment with Python 3.11 and additional dependencies managed via `pip`.

### Create the environment

```bash
conda env create -f environment.yml
conda activate ECG-Project
```

## Project Structure

```text
.
├── README.md                  # Project overview and usage guide
├── environment.yml            # Conda environment definition
├── requirements.txt           # Optional pip requirements

├── data/                      # Data directories
│   ├── raw/                   # Original datasets
│   ├── external/              # Third-party or reference datasets
│   ├── interim/               # Intermediate transformation outputs
│   └── processed/             # Final input data used for modeling

├── preprocessing_pipeline/    # Metaflow pipeline to preprocess raw ECG data
│   ├── preprocess_flow.py     # Main flow file
│   ├── config.py              # Data configuration
│   └── common.py              # Cleaning and Preprocessing functions

├── models/                    # Core model definitions
│   ├── supervised.py          # CNN, TCN, Transformer and Linear and MLP classifiers
│   ├── simclr.py              # SimCLR encoder + projection head
│   ├── ts2vec.py              # TS2Vec architecture
│   ├── ts2vec_soft.py         # Soft TS2Vec variant
│   ├── tstcc.py               # TSTCC architecture
│   ├── tstcc_soft.py          # TSTCC soft variant
│   └── __init__.py

├── training_pipelines/       # Metaflow training flows (supervised and SSL)
│   ├── supervised_training.py
│   ├── simclr_train.py
│   ├── ts2vec_train.py
│   ├── ts2vec_soft_train.py
│   ├── tstcc_train.py
│   ├── tstcc_soft_train.py
│   ├── sophisticated_baseline.py
│   ├── torch_utilities.py     # Helper functions (training loop, metrics, etc.)
│   ├── train.sh               # SLURM job script
│   └── models -> ../models    # Symlink for shared access

├── evaluation_results/        # Metric aggregation and statistical tests
│   ├── collect_metrics_mlflow.py
│   ├── confidence_intervals.py
│   ├── mann_whitney_tests.py
│   └── table_results.py

├── visualization/             # Analysis and result plotting
│   ├── plot_ci.py
│   └── plot_label_efficiency.py

├── results/                   # Generated results and figures
├── PPG_transfer_learning/     # Experimental PPG transfer scripts
│   ├── ppg_preprocessing.py
│   ├── ppg_ts2vecsoft_train.py
│   ├── ppg_inspect.py
│   └── models -> ../models    # Shared model access

└── LICENSE                    # License file
```

## Running the Preprocessing Pipeline

The preprocessing pipeline segments, cleans, normalizes, and windows the raw ECG data using a Metaflow pipeline.

### 1. Activate the environment
```bash
conda activate ECG-Project
```

### 2. Start the MLflow tracking server
```bash
mlflow server --host 127.0.0.1 --port 5000
```

### 3. Run the preprocessing pipeline
```bash
python preprocess_flow.py run
```

This will:
- Segment the raw ECG data
- Clean and denoise signals
- Normalize all recordings
- Segment signals into fixed-length windows

The final output is saved to:
```bash
data/interim/windowed_data.h5
```

## Running the Training Pipelines (Locally)

This section shows how to run supervised and self-supervised training pipelines locally using [Metaflow](https://docs.metaflow.org/).

### 1. Activate the environment

```bash
conda activate ECG-Project
```

### 2. Start the MLflow tracking server
```bash
mlflow server --host 127.0.0.1 --port 5000
```

Keep this process running in a separate terminal. The experiment runs will appear at:
```cpp
http://127.0.0.1:5000
```

### 3. Run a Supervised or Self-Supervised Training Pipeline

From the `training_pipelines/` directory, you can run any Metaflow training script by specifying its parameters through the CLI.

#### Example (Supervised)

```bash
python supervised_training.py run \
  --model_type "cnn" \
  --batch_size 16 \
  --lr 1e-5 \
  --num_epochs 25 \
  --patience 10 \
  --label_fraction 0.01
```

Available supervised models: cnn, tcn, transformer.

#### Example (Self-Supervised)
```bash
python ts2vec_train.py run \
  --ts2vec_epochs 50 \
  --ts2vec_lr 0.001 \
  --ts2vec_batch_size 8 \
  --classifier_epochs 25 \
  --classifier_lr 0.0001 \
  --label_fraction 0.01
```

The corresponding flow will:
- Pretrain a self-supervised encoder (e.g., TS2Vec, TSTCC, SimCLR) and save it to mlflow.
- Freeze the encoder.
- Extract latent representations.
- Train a downstream classifier (linear or MLP) with (limited) labeled data.
- Evaluate the classifier on test set.
- Save metrics to mlflow. 

Replace the script name (ts2vec_train.py, simclr_train.py, tstcc_train.py, etc.) to run different SSL methods. Each script exposes model-specific CLI parameters. All training runs are automatically tracked with MLflow.

## Running the Training Pipelines on DAS6 Cluster

This section explains how to run experiments remotely on the DAS6 cluster using SLURM.

### 1. Activate Environment and Launch MLflow on DAS6

On the cluster login node:

```bash
source activate /var/scratch/username/ECG_env
mlflow server --host 0.0.0.0 --port 5005
```

### 2. Accessing MLflow Remotely

Then, on your local machine, open a tunnel to access MLflow:

```bash
ssh -L 5005:127.0.0.1:5005 username@fs0.das6.cs.vu.nl
```

Now you can view MLflow from your browser at:
```cpp
http://localhost:5005
```

### 3. SLURM Job Configuration

Edit `train.sh` to uncomment the model script you want to run. This script includes setups for:

- **Supervised models:** CNN, TCN, Transformer
- **SSL models:** TS2Vec, TS2VecSoft, TSTCC, TSTCCSoft, SimCLR
- **Transfer learning:** Using TS2VecSoft encoder trained on PPG signals

Example snippet from `train.sh` (CNN):

```bash
python supervised_training.py run \
  --mlflow_tracking_uri "http://fs0.das6.cs.vu.nl:5005" \
  --window_data_path "../../../../var/scratch/cla224/ECG-Project/data/windowed_data.h5" \
  --model_type "cnn" \
  --seed $1 \
  --batch_size 16 \
  --scheduler_factor 0.5 \
  --scheduler_min_lr 1e-09 \
  --patience 20 \
  --lr 1e-5 \
  --label_fraction 0.01
```

> **Note:** Use only one model block per SLURM run to avoid conflicts.

### 4. Launching Training Jobs

Submit training jobs with different random seeds:

```bash
for SEED in 1 42 1234 1337 2025; do
    sbatch train.sh $SEED
done
```

Logs will be written to `ecg_train.out` and `ecg_train.err` in the current working directory and everything will be tracked in MLflow.

