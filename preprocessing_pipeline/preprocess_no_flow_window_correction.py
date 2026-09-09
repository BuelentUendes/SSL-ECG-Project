import os
import mlflow
import argparse

from common import (
    process_ecg_data_window_correction,
    normalize_cleaned_data,
    segment_data_into_windows_sqi_filtered,
)

from utils.helper_paths import DATA_PATH
from utils.torch_utilities import create_directory


def main():
    """
    ECG Preprocessing Pipeline — window-level SQI correction.

    Order of operations:
      1. Apply length-preserving Butterworth highpass + notch filters.
      2. Compute per-sample SQI on the full-length signal (no removal).
      3. Slice into conditions using original timestamp indices.
      4. Normalize (participant-level z-score).
      5. Window each condition; drop any window containing a sample with SQI <= threshold.
         Individual samples are never modified — always whole-window keep or drop.
    """

    parser = argparse.ArgumentParser(
        description="ECG Preprocessing Pipeline (window-level SQI correction)"
    )
    parser.add_argument(
        "--mlflow_tracking_uri",
        help="MLflow tracking URI",
        default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"),
    )
    parser.add_argument(
        "--fs",
        help="Sampling frequency in Hz (original is 1000)",
        default=1000,
        type=int,
    )
    parser.add_argument(
        "--window_size",
        help="Window size in seconds",
        default=10,
        type=int,
    )
    parser.add_argument(
        "--step_size",
        help="Stride between windows in seconds",
        default=5,
        type=int,
    )
    parser.add_argument(
        "--sqi_threshold",
        help="SQI threshold below which a window is dropped (default 0.25)",
        default=0.25,
        type=float,
    )

    args = parser.parse_args()

    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment("ECGPreprocessing")

    ROOT_PATH = os.path.join(
        DATA_PATH,
        "interim",
        "ECG",
        str(args.fs),
        str(args.window_size),
        str(args.step_size),
    )
    print(f"Output path: {ROOT_PATH}")
    create_directory(ROOT_PATH)

    segmented_data_path = os.path.join(ROOT_PATH, "ecg_data_segmented.h5")
    sqi_data_path = os.path.join(ROOT_PATH, "ecg_data_sqi.h5")
    normalized_data_path = os.path.join(ROOT_PATH, "ecg_data_normalized.h5")
    window_data_path = os.path.join(ROOT_PATH, "windowed_data.h5")

    try:
        run = mlflow.start_run(run_name=f"ecg_preprocessing_window_correction_{args.fs}hz")
        mlflow_run_id = run.info.run_id
        print(f"MLflow run started: {mlflow_run_id}")
    except Exception as e:
        raise RuntimeError(f"MLflow connection failed: {str(e)}")

    try:
        print("Starting ECG preprocessing (window-level SQI correction)...")

        # Step 1: filter + SQI computation + condition slicing
        sqi_csv_path = os.path.join(ROOT_PATH, "sqi_statistics.csv")
        if not os.path.exists(segmented_data_path):
            print("Filtering, computing SQI, and slicing conditions...")
            sqi_df = process_ecg_data_window_correction(
                segmented_data_path,
                sqi_data_path,
                fs=args.fs,
                sqi_threshold=args.sqi_threshold,
            )
            sqi_df.to_csv(sqi_csv_path, index=False)
            mlflow.log_artifact(sqi_csv_path)

            condition_summary = (
                sqi_df[sqi_df["condition"] != "_ALL_"]
                .groupby(["class", "condition"])["proportion_removed"]
                .agg(["mean", "min", "max", "count"])
                .round(4)
            )
            print("\n--- SQI summary (by class/condition, informational — no removal yet) ---")
            print(condition_summary.to_string())

            participant_summary = (
                sqi_df[sqi_df["condition"] == "_ALL_"]
                .set_index("participant")[
                    ["n_original_samples", "n_removed_samples", "proportion_removed"]
                ]
            )
            print("\n--- SQI summary (per participant) ---")
            print(participant_summary.to_string())
        else:
            print(f"Using existing segmented data: {segmented_data_path}")

        # Step 2: participant-level z-score normalisation
        if not os.path.exists(normalized_data_path):
            print("Normalizing ECG data...")
            normalize_cleaned_data(segmented_data_path, normalized_data_path)
        else:
            print(f"Using existing normalized data: {normalized_data_path}")

        # Step 3: sliding-window with whole-window SQI keep/drop
        print("Creating windowed ECG data with window-level SQI filtering...")
        window_csv_path = os.path.join(ROOT_PATH, "window_drop_statistics.csv")
        window_df = segment_data_into_windows_sqi_filtered(
            normalized_data_path,
            sqi_data_path,
            window_data_path,
            fs=args.fs,
            window_size=args.window_size,
            step_size=args.step_size,
            sqi_threshold=args.sqi_threshold,
        )
        print(f"Windowed data saved to: {window_data_path}")

        # ── Window-drop summary ────────────────────────────────────────────
        window_df.to_csv(window_csv_path, index=False)
        mlflow.log_artifact(window_csv_path)

        # Totals across the whole dataset
        total_windows = window_df["n_total_windows"].sum()
        total_dropped = window_df["n_dropped_windows"].sum()
        total_kept = window_df["n_kept_windows"].sum()
        print(
            f"\n--- Window drop summary (overall) ---\n"
            f"Total windows : {total_windows}\n"
            f"Kept          : {total_kept}  "
            f"({100 * total_kept / total_windows:.1f}%)\n"
            f"Dropped       : {total_dropped}  "
            f"({100 * total_dropped / total_windows:.1f}%)"
        )

        # By class label
        by_label = (
            window_df.groupby("label")[["n_total_windows", "n_kept_windows", "n_dropped_windows"]]
            .sum()
            .assign(proportion_dropped=lambda d: d["n_dropped_windows"] / d["n_total_windows"])
            .round({"proportion_dropped": 4})
        )
        print("\n--- Window drop summary (by class label) ---")
        print(by_label.to_string())

        # By class label × original condition name
        by_condition = (
            window_df.groupby(["label", "condition"])[
                ["n_total_windows", "n_kept_windows", "n_dropped_windows"]
            ]
            .sum()
            .assign(proportion_dropped=lambda d: d["n_dropped_windows"] / d["n_total_windows"])
            .round({"proportion_dropped": 4})
            .sort_values(["label", "proportion_dropped"], ascending=[True, False])
        )
        print("\n--- Window drop summary (by class label × condition) ---")
        print(by_condition.to_string())

        # Per-participant totals
        by_participant = (
            window_df.groupby("participant")[["n_total_windows", "n_kept_windows", "n_dropped_windows"]]
            .sum()
            .assign(proportion_dropped=lambda d: d["n_dropped_windows"] / d["n_total_windows"])
            .round({"proportion_dropped": 4})
            .sort_values("proportion_dropped", ascending=False)
        )
        print("\n--- Window drop summary (per participant) ---")
        print(by_participant.to_string())
        # ──────────────────────────────────────────────────────────────────

        mlflow.log_params({
            "fs": args.fs,
            "window_size": args.window_size,
            "step_size": args.step_size,
            "sqi_threshold": args.sqi_threshold,
            "correction_mode": "window_level",
        })
        mlflow.log_metrics({
            "total_windows": int(total_windows),
            "windows_kept": int(total_kept),
            "windows_dropped": int(total_dropped),
            "proportion_dropped": round(total_dropped / total_windows, 4) if total_windows else 0.0,
        })

        print("\nECG preprocessing (window-level SQI correction) complete.")

    except Exception as e:
        print(f"Error during preprocessing: {e}")
        raise
    finally:
        mlflow.end_run()


if __name__ == "__main__":
    main()