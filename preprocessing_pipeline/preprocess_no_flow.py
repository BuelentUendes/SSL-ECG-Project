import os
import mlflow
import argparse

from common import (
    process_ecg_data,
    process_save_cleaned_data,
    normalize_cleaned_data,
    segment_data_into_windows,
)

from utils.helper_paths import DATA_PATH
from utils.torch_utilities import create_directory


def main():
    """
    ECG Preprocessing Pipeline:
    - Segments raw ECG data
    - Cleans and normalizes signals
    - Segments into time windows
    """

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="ECG Preprocessing Pipeline")
    parser.add_argument(
        "--mlflow_tracking_uri",
        help="MLflow tracking URI",
        default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    )
    parser.add_argument(
        "--fs",
        help="Sampling frequency (Hz) to use. If set to 64 for example, it will use the 64 Hz version. Original is 1,000",
        default=1000,
        type=int,
    )
    parser.add_argument(
        "--window_size",
        help="Size of each window (seconds)",
        default=30, #30s
        type=int
    )
    parser.add_argument(
        "--step_size",
        help="Stride between windows (seconds)",
        default=10, #10s shift
        type=int
    )

    args = parser.parse_args()

    # Initialize MLflow
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment("ECGPreprocessing")

    # Set up paths
    ROOT_PATH = os.path.join(DATA_PATH, "interim", "ECG", f"{args.fs}", f"{args.window_size}", f"{args.step_size}")
    print(f"The data path is {ROOT_PATH}")

    create_directory(ROOT_PATH)

    segmented_data_path = os.path.join(ROOT_PATH, "ecg_data_segmented.h5")
    cleaned_data_path = os.path.join(ROOT_PATH, "ecg_data_cleaned.h5")
    normalized_data_path = os.path.join(ROOT_PATH, "ecg_data_normalized.h5")
    window_data_path = os.path.join(ROOT_PATH, "windowed_data.h5")

    # Start MLflow run
    try:
        run = mlflow.start_run(run_name=f"ecg_preprocessing_{args.fs}hz")
        mlflow_run_id = run.info.run_id
        print(f"MLflow run started: {mlflow_run_id}")
    except Exception as e:
        raise RuntimeError(f"MLflow connection failed: {str(e)}")

    try:
        print("Starting ECG preprocessing...")

        # Step 1: Clean raw ECG Data and Segment raw data
        if not os.path.exists(segmented_data_path):
            print("Clean and segment raw ECG data...")
            process_ecg_data(segmented_data_path, fs=args.fs)
        else:
            print(f"Using existing clean and segmented data: {segmented_data_path}")

        # Step 2: Normalize cleaned signals
        if not os.path.exists(normalized_data_path):
            print("Normalizing ECG data...")
            normalize_cleaned_data(segmented_data_path, normalized_data_path)
        else:
            print(f"Using existing normalized data: {normalized_data_path}")

        # Step 3: Segment into windows
        print("Creating windowed ECG data...")
        segment_data_into_windows(
            normalized_data_path,
            window_data_path,
            fs=args.fs,
            window_size=args.window_size,
            step_size=args.step_size,
        )
        print(f"Windowed data saved to: {window_data_path}")

        # Log parameters to MLflow
        mlflow.log_params({
            "fs": args.fs,
            "window_size": args.window_size,
            "step_size": args.step_size,
        })

        print("ECG preprocessing complete.")

    except Exception as e:
        print(f"Error during preprocessing: {e}")
        raise
    finally:
        mlflow.end_run()


if __name__ == "__main__":
    main()