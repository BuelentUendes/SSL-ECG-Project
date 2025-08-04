import os, glob, h5py
import argparse

import numpy as np
import pandas as pd
import neurokit2 as nk

from utils.helper_paths import DATA_PATH
from utils.torch_utilities import create_directory

# Category Mapping for the ECG and PPG WESAD dataset

CATEGORY_MAPPING_PPG = {
    "baseline": 1,
    "mental_stress": 2,
    "amusement": 3,
    "meditation": 4,
    "other": [5, 6, 7],
}

from common import (
    process_save_cleaned_data,
    normalize_cleaned_data,
    segment_data_into_windows,
)

# ───────────────────────────────
# helper funct.
# ───────────────────────────────
def get_label(condition, category_mapping):
    for key, values in category_mapping.items():
        if condition in values:
            return key

    print(f"we could not find {condition} in the mapping. Return {None}")
    return None

def participant_id(fpath: str) -> str:
    """
    /raw/WESAD/S10/ECG_complete_data.csv  ->  S10
    """
    return os.path.basename(os.path.dirname(fpath))

def is_valid_segment(sig, fs):
    return (len(sig) > 0) and (fs is not None) and (fs > 0)

def capitalize_sensor(value):
    return str(value).upper()

def process_save_cleaned_data_ppg(
        segmented_data_path,
        output_hdf5_path,
        fs:int=64,
):
    """
    Loads ECG data from segmented_data_path, cleans each individual segment,
    and saves the cleaned segments to output_hdf5_path preserving the structure.
    """
    with h5py.File(segmented_data_path, "r") as f_in, h5py.File(output_hdf5_path, "w") as f_out:
        for participant in f_in.keys():
            print(f"Cleaning data for {participant}...")
            participant_in = f_in[participant]
            participant_out = f_out.create_group(participant)
            for category in participant_in.keys():
                category_in = participant_in[category]
                category_out = participant_out.create_group(category)
                for segment_name in category_in.keys():
                    signal = category_in[segment_name][...]
                    try:
                        cleaned_signal = nk.ppg_clean(signal, fs)
                    except Exception as e:
                        print(f"Error cleaning signal for {participant}/{category}/{segment_name}: {e}")
                    # Save cleaned segment with same segment name
                    category_out.create_dataset(
                        segment_name,
                        data=cleaned_signal.astype(np.float32),
                        compression="gzip",
                        compression_opts=4,
                        dtype='float32'
                    )
    print(f"Cleaned ECG data saved to {output_hdf5_path}")

# ------------------------------------------------------

# ────────────────────────────────────────────
# .csv to raw HDF5 (rename it to csv to HDF5)
# ────────────────────────────────────────────

def csv_to_hdf5(root_dir, out_h5, physiological_sensor="ECG", placement="chest"):
    """
    Convert CSV files to HDF5 format - matching the AVRO function structure.

    Parameters:
    root_dir (str): Root directory containing subject folders
    out_h5 (str): Output HDF5 file path
    physiological_sensor (str): Name of the physiological sensor (default: "ECG")
    placement (str): Sensor placement (chest or wrist)
    """
    print(f"\n[INFO] CSV → HDF5   |   root = {root_dir}")

    # Define sampling rates based on placement
    if placement == "wrist":
        frequency_dict = {'ACC': 32, 'BVP': 64, 'EDA': 4, 'TEMP': 4, 'label': 700}
    else:
        frequency_dict = {'ECG': 700, "EMG": 700, "EDA": 700, "TEMP": 700, "RESP": 700, "label": 700}

    with h5py.File(out_h5, "w") as fout:
        # Updated pattern to match your file structure
        pattern = os.path.join(root_dir, "S*", f"{physiological_sensor}_complete_data.csv")
        csv_files = sorted(glob.glob(pattern))

        label_mapping = {
            0: "transient",
            1: "baseline",
            2: "mental_stress", #Here not mental stress but other stress, physiological stress
            3: "amusement",
            4: "meditation",
            5: "other",
            6: "other",
            7: "other"
        }

        if not csv_files:
            print(f"[WARNING] No CSV files found with pattern: {pattern}")
            # Try alternative patterns
            alt_pattern = os.path.join(root_dir, "S*", f"*{physiological_sensor}*.csv")
            csv_files = sorted(glob.glob(alt_pattern))
            if csv_files:
                print(f"[INFO] Found files with alternative pattern: {alt_pattern}")

        for idx, fpath in enumerate(csv_files, 1):# all participants
            print(f"Processing Participant {idx} /{len(csv_files)}")
            part = participant_id(fpath)

            try:
                # Read CSV and extract data
                df = pd.read_csv(fpath)

                # Create participant group
                participant_group = fout.create_group(f"participant_{part}")

                # Loop through intervals and save each segment individually
                complete_data = pd.read_csv(fpath)

                if physiological_sensor not in df.columns:
                    print(f" SKIP {os.path.basename(fpath)} – no {physiological_sensor} column found")
                    continue

                # vals = df[physiological_sensor].values.astype(np.float32)

                # Get sampling frequency (equivalent to bvp["samplingFrequency"])
                fs = frequency_dict.get(physiological_sensor, 700)

                # Get the corresponding segments:
                # Get unique conditions
                experimental_conditions = list(complete_data["label"].unique())

                for label in experimental_conditions:
                    # Get the segments that are associated with the experimental condition
                    labeled_segment_df = complete_data[complete_data["label"] == label]
                    segment = labeled_segment_df[physiological_sensor].values.astype(np.float32)

                    # Encode the integer to a nice string descriptive
                    label_encoding = label_mapping.get(label, "other")

                    # Create or get category group under participant
                    if label_encoding not in participant_group:
                        category_group = participant_group.create_group(label_encoding)
                    else:
                        category_group = participant_group[label_encoding]

                    # Save this segment under a unique dataset name
                    seg_name = f"segment_{len(category_group.keys())}"
                    category_group.create_dataset(
                        seg_name,
                        data=segment,
                        compression="gzip",
                        compression_opts=4,
                        dtype=np.float32,
                        )

                    print(
                        f"[{idx}] {part}: stored {len(segment)} samples @ {fs} Hz for label: {label_encoding} and signal {physiological_sensor}"
                    )

            except Exception as e:
                print(f" ERROR processing {os.path.basename(fpath)}: {str(e)}")
                continue

    print(f"[OK] CSV HDF5 → {out_h5}")

def main(args):
    # Setup the pipeline
    WESAD_SAVE_PATH = os.path.join(DATA_PATH, "interim","WESAD", args.physiological_sensor)
    create_directory(WESAD_SAVE_PATH)

    ROOT_DIR        = os.path.join(DATA_PATH, "raw", "WESAD")

    RAW_H5          = os.path.join(WESAD_SAVE_PATH, "wesad_raw.h5")
    CLEAN_H5        = os.path.join(WESAD_SAVE_PATH, "wesad_clean.h5")
    NORM_H5         = os.path.join(WESAD_SAVE_PATH, "wesad_norm.h5")
    WIN_H5          = os.path.join(WESAD_SAVE_PATH, "windowed_data.h5")

    csv_to_hdf5(ROOT_DIR, RAW_H5, args.physiological_sensor, args.placement)
    if args.physiological_sensor == "ECG":
        process_save_cleaned_data(RAW_H5,CLEAN_H5,fs=args.fs,)
    else:
        process_save_cleaned_data_ppg(RAW_H5, CLEAN_H5, fs=args.fs, )

    normalize_cleaned_data(CLEAN_H5, NORM_H5)
    segment_data_into_windows(NORM_H5, WIN_H5, fs=args.fs, window_size=args.window_size, step_size=args.step_size)

# ────────────────────────────────────────────────────────────────
# main script for preprocessing of the WESAD dataset
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="WESAD Dataset Preprocessing Pipeline")

    parser.add_argument(
        "--fs",
        help="Sampling frequency (Hz) to use. If set to 64 for example, it will use the 64 Hz version. Original is 700 for WESAD",
        default=700, #Chest ECG WESAD has 700. IMPORTANT: BVP has 64Hz!
        type=int,
    )
    parser.add_argument(
        "--physiological_sensor",
        help="What sensor to use for the WESAD dataset",
        default="ECG",
        choices=("ECG", "BVP"),
        type=capitalize_sensor,
    )
    parser.add_argument(
        "--placement",
        help="Where is the sensor placed? chest or wrist",
        default="chest",
        choices=("chest", "wrist"),
        type=str,
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

    # #ToDo:
    # # Do this now with PPG option as well
    # # -> Then everything in place

