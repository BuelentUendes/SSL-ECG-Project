import os
import mlflow
from metaflow import FlowSpec, step, Parameter, current, project

from common import (
    process_ecg_data,
    process_save_cleaned_data,
    normalize_cleaned_data,
    segment_data_into_windows,
)

from utils.helper_paths import DATA_PATH
from utils.torch_utilities import create_directory


@project(name="ecg_preprocessing")
class ECGPreprocessFlow(FlowSpec):
    """
    Preprocessing Flow:
    - Segments raw ECG data
    - Cleans and normalizes signals
    - Segments into time windows
    """

    mlflow_tracking_uri = Parameter(
        "mlflow_tracking_uri",
        help="MLflow tracking URI",
        default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    )

    downsample_signal = Parameter(
        "downsample_signal",
        help="Boolean. Downsample ECG signal",
        default=False,
    )

    fs = Parameter(
        "fs",
        help="Sampling frequency (Hz)",
        default=1000,
        type=int,
    )

    target_fs= Parameter(
        "target_fs",
        help="What target frequency (Hz) (if downsample_signal is set to True).",
        default=64,
        type=int,
    )

    window_size = Parameter(
        "window_size",
        help="Size of each window (seconds)",
        default=10
    )

    step_size = Parameter(
        "step_size",
        help="Stride between windows (seconds)",
        default=5
    )

    @step
    def start(self):
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_experiment("ECGPreprocessing")

        if self.downsample_signal:
            self.ROOT_PATH = os.path.join(DATA_PATH, "interim", f"ECG_{self.target_fs}")
        else:
            self.ROOT_PATH = os.path.join(DATA_PATH, "interim")

        create_directory(self.ROOT_PATH)

        self.segmented_data_path = os.path.join(self.ROOT_PATH, "ecg_data_segmented.h5")
        self.cleaned_data_path = os.path.join(self.ROOT_PATH, "ecg_data_cleaned.h5")
        self.normalized_data_path = os.path.join(self.ROOT_PATH, "ecg_data_normalized.h5")
        self.window_data_path = os.path.join(self.ROOT_PATH, "windowed_data.h5")

        try:
            run = mlflow.start_run(run_name=current.run_id)
            self.mlflow_run_id = run.info.run_id

        except Exception as e:
            raise RuntimeError(f"MLflow connection failed: {str(e)}")

        print("Starting ECG preprocessing...")
        self.next(self.preprocess)

    @step
    def preprocess(self):
        """Run preprocessing pipeline from raw to windowed HDF5."""
        # Step 1: Segment raw data
        if not os.path.exists(self.segmented_data_path):
            print("Segmenting raw ECG data...")
            process_ecg_data(self.segmented_data_path)
        else:
            print(f"Using existing segmented data: {self.segmented_data_path}")

        # Step 2: Clean ECG signals (downsample if specified
        if not os.path.exists(self.cleaned_data_path):
            print("Cleaning ECG data...")
            process_save_cleaned_data(
                self.segmented_data_path,
                self.cleaned_data_path,
                fs=self.fs,
                downsample_signal=self.downsample_signal,
                target_fs=self.target_fs,
            )
            if self.downsample_signal:
                self.fs = self.target_fs
        else:
            print(f"Using existing cleaned data: {self.cleaned_data_path}")

        # Step 3: Normalize cleaned signals
        if not os.path.exists(self.normalized_data_path):
            print("Normalizing ECG data...")
            normalize_cleaned_data(self.cleaned_data_path, self.normalized_data_path)
        else:
            print(f"Using existing normalized data: {self.normalized_data_path}")

        # Step 4: Segment into windows
        print("Creating windowed ECG data...")
        segment_data_into_windows(
            self.normalized_data_path,
            self.window_data_path,
            fs=self.fs,
            window_size=self.window_size,
            step_size=self.step_size,
        )
        print(f"Windowed data saved to: {self.window_data_path}")
        self.next(self.end)

    @step
    def end(self):
        mlflow.end_run() 
        print("ECG preprocessing complete.")

if __name__ == "__main__":
    ECGPreprocessFlow()