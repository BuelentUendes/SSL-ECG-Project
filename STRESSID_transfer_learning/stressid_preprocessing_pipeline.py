import os, glob, h5py
import argparse

import numpy as np
import pandas as pd
import neurokit2 as nk

from utils.helper_paths import DATA_PATH
from utils.torch_utilities import create_directory

# Import other functions from the main preprocessing pipeline
from common import (
    clean_and_filter_ecg,
    normalize_cleaned_data,
    segment_data_into_windows,
)

# ───────────────────────────────
# helper functions
# ───────────────────────────────
def participant_id_from_filename(fpath: str) -> str:
    """
    Extract participant ID from file path
    /data/raw/STRESSID/Physiological/2ea4/2ea4_Baseline.txt -> 2ea4
    """
    return os.path.basename(os.path.dirname(fpath))

def task_from_filename(fpath: str) -> str:
    """
    Extract task name from file path
    /data/raw/STRESSID/Physiological/2ea4/2ea4_Baseline.txt -> Baseline
    """
    filename = os.path.basename(fpath)
    # Remove participant ID and extension: 2ea4_Baseline.txt -> Baseline
    task = filename.split('_', 1)[1].replace('.txt', '')
    return task

def is_valid_segment(sig, fs):
    return (len(sig) > 0) and (fs is not None) and (fs > 0)

def capitalize_sensor(value):
    return str(value).upper()

# ────────────────────────────────────────────
# .txt to raw HDF5 conversion
# ────────────────────────────────────────────

def process_stressid_data(root_dir, labels_csv_path, out_h5, physiological_sensor="ECG", fs=500):
    """
    Process StressID data: Convert TXT files to HDF5 format with cleaning and filtering.
    This mimics the process_ecg_data function - loads all participant data first, 
    then cleans the entire signal, then segments by task/condition.

    Parameters:
    root_dir (str): Root directory containing participant folders (Physiological/)
    labels_csv_path (str): Path to labels.csv file
    out_h5 (str): Output HDF5 file path
    physiological_sensor (str): Name of the physiological sensor (default: "ECG")
    fs (int): Sampling frequency (default: 500)
    """
    print(f"\n[INFO] Processing StressID data with cleaning and filtering: {root_dir}")
    
    # Load labels
    labels_df = pd.read_csv(labels_csv_path)
    
    # Create a mapping from subject/task to stress label
    label_mapping = {}
    for _, row in labels_df.iterrows():
        subject_task = row['subject/task']
        binary_stress = row['binary-stress']
        label_mapping[subject_task] = 'mental_stress' if binary_stress == 1 else 'baseline'

    with h5py.File(out_h5, "w") as fout:
        # Find all txt files in participant directories
        pattern = os.path.join(root_dir, "*", "*.txt")
        txt_files = sorted(glob.glob(pattern))

        if not txt_files:
            print(f"[WARNING] No TXT files found with pattern: {pattern}")
            return

        print(f"[INFO] Found {len(txt_files)} TXT files")

        # Group files by participant
        participants = {}
        for fpath in txt_files:
            part_id = participant_id_from_filename(fpath)
            task = task_from_filename(fpath)
            
            if part_id not in participants:
                participants[part_id] = []
            participants[part_id].append((fpath, task))

        for idx, (part_id, files) in enumerate(participants.items(), 1):
            print(f"Processing Participant {part_id} ({idx}/{len(participants)})")

            # Step 1: Load all task data for this participant and create exact task mapping
            participant_data = {}
            all_signals = []
            task_mapping = []  # Track which task each sample belongs to
            
            for fpath, task in files:
                try:
                    # Read TXT file
                    df = pd.read_csv(fpath)
                    
                    # Check if physiological sensor column exists
                    if physiological_sensor not in df.columns:
                        print(f" SKIP {os.path.basename(fpath)} – no {physiological_sensor} column found")
                        continue

                    # Extract physiological signal
                    signal = df[physiological_sensor].values.astype(np.float32)
                    participant_data[task] = signal
                    all_signals.append(signal)
                    
                    # Create task mapping: track which task each sample belongs to
                    task_mapping.extend([task] * len(signal))
                    
                except Exception as e:
                    print(f" ERROR loading {os.path.basename(fpath)}: {str(e)}")
                    continue
            
            if not all_signals:
                print(f" SKIP participant {part_id} - no valid signals found")
                continue
                
            # Step 2: Concatenate all signals and clean once (like main pipeline)
            concatenated_signal = np.concatenate(all_signals)
            task_mapping = np.array(task_mapping)  # Convert to numpy array for easier indexing
            
            if physiological_sensor == "ECG":
                try:
                    print(f" Cleaning entire ECG signal for {part_id} ({len(concatenated_signal)} samples)")
                    cleaned_signal, good_indices = clean_and_filter_ecg(concatenated_signal, sampling_rate=fs, return_indices=True)
                    
                    # Use the exact good indices returned by the cleaning function
                    print(f" Signal length changed from {len(concatenated_signal)} to {len(cleaned_signal)} after cleaning")
                    print(f" Using exact {len(good_indices)} good indices for task mapping")
                    
                    # Use exact indices to map back to task assignments
                    cleaned_task_mapping = task_mapping[good_indices]
                        
                except Exception as e:
                    print(f" WARNING: Could not clean ECG signal for {part_id}: {e}")
                    print(f" Using raw concatenated signal instead.")
                    cleaned_signal = concatenated_signal
                    cleaned_task_mapping = task_mapping
            else:
                cleaned_signal = concatenated_signal
                cleaned_task_mapping = task_mapping

            # Step 3: Split cleaned signal back into individual tasks using exact task mapping
            cleaned_participant_data = {}
            
            for task in participant_data.keys():
                # Find all indices where the task matches
                task_indices = np.where(cleaned_task_mapping == task)[0]
                
                if len(task_indices) > 0:
                    cleaned_participant_data[task] = cleaned_signal[task_indices]
                else:
                    print(f" WARNING: No samples found for task {task} after cleaning")
                    continue
            
            # Step 4: Create participant group and save segments
            participant_group = fout.create_group(f"participant_{part_id}")

            for task, cleaned_signal_segment in cleaned_participant_data.items():
                try:
                    # Get stress label for this subject/task combination
                    subject_task = f"{part_id}_{task}"
                    if subject_task not in label_mapping:
                        print(f" SKIP {subject_task} – no label found in labels.csv")
                        continue
                    
                    label_category = label_mapping[subject_task]

                    # Create or get category group under participant
                    if label_category not in participant_group:
                        category_group = participant_group.create_group(label_category)
                    else:
                        category_group = participant_group[label_category]

                    # Save this segment under a unique dataset name
                    seg_name = f"segment_{len(category_group.keys())}"
                    category_group.create_dataset(
                        seg_name,
                        data=cleaned_signal_segment,
                        compression="gzip",
                        compression_opts=4,
                        dtype=np.float32,
                    )

                    print(
                        f"[{idx}] {part_id}: stored {len(cleaned_signal_segment)} samples @ {fs} Hz for task: {task} -> {label_category}"
                    )

                except Exception as e:
                    print(f" ERROR processing {part_id}/{task}: {str(e)}")
                    continue

    print(f"[OK] Processed StressID data saved to {out_h5}")

def main(args):
    """
    StressID Preprocessing Pipeline:
    - Process raw StressID data with cleaning and filtering
    - Normalizes signals
    - Segments into time windows
    """
    
    # Set up paths (mimicking preprocess_no_flow.py structure)
    ROOT_PATH = os.path.join(DATA_PATH, "interim", "STRESSID", args.physiological_sensor, f"{args.fs}", f"{args.window_size}", f"{args.step_size}")
    print(f"The data path is {ROOT_PATH}")
    
    create_directory(ROOT_PATH)

    ROOT_DIR = os.path.join(DATA_PATH, "raw", "STRESSID", "Physiological")
    LABELS_CSV = os.path.join(DATA_PATH, "raw", "STRESSID", "labels.csv")

    # File paths following the preprocess_no_flow.py naming convention
    segmented_data_path = os.path.join(ROOT_PATH, "stressid_data_segmented.h5")
    normalized_data_path = os.path.join(ROOT_PATH, "stressid_data_normalized.h5")
    window_data_path = os.path.join(ROOT_PATH, "windowed_data.h5")

    try:
        print("Starting StressID preprocessing...")

        # Step 1: Clean and segment raw StressID data (combines cleaning and segmentation)
        if not os.path.exists(segmented_data_path):
            print("Clean and segment raw StressID data...")
            process_stressid_data(ROOT_DIR, LABELS_CSV, segmented_data_path, args.physiological_sensor, args.fs)
        else:
            print(f"Using existing clean and segmented data: {segmented_data_path}")

        # Step 2: Normalize cleaned signals
        if not os.path.exists(normalized_data_path):
            print("Normalizing StressID data...")
            normalize_cleaned_data(segmented_data_path, normalized_data_path)
        else:
            print(f"Using existing normalized data: {normalized_data_path}")

        # Step 3: Segment into windows
        print("Creating windowed StressID data...")
        segment_data_into_windows(
            normalized_data_path,
            window_data_path,
            fs=args.fs,
            window_size=args.window_size,
            step_size=args.step_size,
        )
        print(f"Windowed data saved to: {window_data_path}")

        print("StressID preprocessing complete.")

    except Exception as e:
        print(f"Error during preprocessing: {e}")
        raise

# ────────────────────────────────────────────────────────────────
# main script for preprocessing of the StressID dataset
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="StressID Dataset Preprocessing Pipeline")

    parser.add_argument(
        "--fs",
        help="Sampling frequency (Hz) to use. Original is 500 for StressID",
        default=500,
        type=int,
    )
    parser.add_argument(
        "--physiological_sensor",
        help="What sensor to use for the StressID dataset",
        default="ECG",
        choices=("ECG", "EDA", "RR"),
        type=capitalize_sensor,
    )

    parser.add_argument(
        "--window_size",
        help="Size of each window (seconds)",
        default=10,
        type=int
    )
    parser.add_argument(
        "--step_size",
        help="Stride between windows (seconds)",
        default=5,
        type=int
    )

    args = parser.parse_args()

    main(args)