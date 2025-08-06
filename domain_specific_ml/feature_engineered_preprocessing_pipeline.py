import os, glob, h5py
import argparse
from pyexpat import features

import numpy as np
import pandas as pd

from utils.helper_paths import DATA_PATH
from utils.torch_utilities import create_directory


# ───────────────────────────────
# helper funct.
# ───────────────────────────────

def participant_id(fpath: str) -> str:
    """
    /1000/10/5/30100.csv -> 30100 (str)
    """
    id = fpath.split("/")[-1].split(".csv")[0]
    return id

# ------------------------------------------------------

# ────────────────────────────────────────────
# .csv to raw HDF5 (rename it to csv to HDF5)
# ────────────────────────────────────────────

def csv_to_hdf5(root_dir, out_h5):
    """
    Convert CSV files to HDF5 format - matching the AVRO function structure.

    Parameters:
    root_dir (str): Root directory containing subject folders
    out_h5 (str): Output HDF5 file path
    physiological_sensor (str): Name of the physiological sensor (default: "ECG")
    placement (str): Sensor placement (chest or wrist)
    """
    print(f"\n[INFO] CSV → HDF5   |   root = {root_dir}")

    with h5py.File(out_h5, "w") as fout:
        pattern = os.path.join(root_dir, "*.csv")
        csv_files = sorted(glob.glob(pattern))

        if not csv_files:
            print(f"[WARNING] No CSV files found with pattern: {pattern}")
            # Try alternative patterns
            alt_pattern = os.path.join(root_dir, f"*.csv")
            csv_files = sorted(glob.glob(alt_pattern))
            if csv_files:
                print(f"[INFO] Found files with alternative pattern: {alt_pattern}")

        for idx, fpath in enumerate(csv_files, 1):# all participants
            print(f"Processing Participant {idx} /{len(csv_files)}")
            part = participant_id(fpath)

            try:
                # Create participant group
                participant_group = fout.create_group(f"participant_{part}")

                # Loop through intervals and save each segment individually
                complete_data = pd.read_csv(fpath)

                # Get the corresponding segments:
                # Get unique conditions
                experimental_conditions = list(complete_data["category"].unique())

                for label in experimental_conditions:
                    # Get the segments that are associated with the experimental condition
                    labeled_segment_df = complete_data[complete_data["category"] == label]

                    # I i should drop category and label and store the segment as npflot (it is a df of 687, 19) then
                    # Drop category and label columns, then convert to numpy
                    segment_df = labeled_segment_df.drop(columns=['category', 'label'])
                    feature_names = list(segment_df.columns)

                    # Add feature names as attribute to participant group, for later as we then want to normalize the features
                    participant_group.attrs['feature_names'] = feature_names

                    segment = segment_df.to_numpy()

                    # Create or get category group under participant
                    if label not in participant_group:
                        category_group = participant_group.create_group(label)
                    else:
                        category_group = participant_group[label]

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
                        f"[{idx}] {part}: stored {segment.shape} samples for label: {label}"
                    )
            except Exception as e:
                print(f" ERROR processing {os.path.basename(fpath)}: {str(e)}")
                continue

    print(f"[OK] CSV HDF5 → {out_h5}")

def main(args):
    # Setup the pipeline
    ECG_FEATURES_SAVE_PATH = os.path.join(DATA_PATH, "interim","ECG_features", str(args.window_size))
    create_directory(ECG_FEATURES_SAVE_PATH)

    ROOT_DIR        = os.path.join(DATA_PATH, "features", f"{str(args.fs)}", str(args.window_size), str(args.step_size))
    WIN_H5          = os.path.join(ECG_FEATURES_SAVE_PATH, "windowed_data.h5")

    # Each row corresponds already to a segment and window
    csv_to_hdf5(ROOT_DIR, WIN_H5)

# ────────────────────────────────────────────────────────────────
# main script for preprocessing of the WESAD dataset
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="ECG Feature Engineered Dataset Preprocessing Pipeline")

    parser.add_argument(
        "--fs",
        help="Sampling frequency (Hz) to use. If set to 64 for example, it will use the 64 Hz version. Original is 700 for WESAD",
        default=1000, #Chest ECG WESAD has 700. IMPORTANT: BVP has 64Hz!
        type=int,
    )

    parser.add_argument(
        "--window_size",
        help="Size of each window (seconds)",
        default=30,
        type=int
    )
    parser.add_argument(
        "--step_size",
        help="Stride between windows (seconds)",
        default=10,
        type=int
    )

    args = parser.parse_args()

    main(args)

