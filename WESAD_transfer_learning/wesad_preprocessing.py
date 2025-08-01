import os
import argparse

import pandas as pd
import pickle
import numpy as np
from tqdm import tqdm

from utils.helper_paths import DATA_PATH

class Subject:

    def __init__(self, main_path, subject_number):
        self.name = subject_number
        self.subject_keys = ['signal', 'label', 'subject']
        self.signal_keys = ['chest', 'wrist']
        self.chest_keys = ['ACC', 'ECG', 'EMG', 'EDA', 'Temp', 'Resp']
        self.wrist_keys = ['ACC', 'BVP', 'EDA', 'TEMP']

        with open(os.path.join(main_path, self.name) + '/' + self.name + '.pkl', 'rb') as file:
            self.data = pickle.load(file, encoding='latin1')

        self.labels = self.data['label']

    def get_wrist_data(self):
        data = self.data['signal']['wrist']
        return data

    def get_chest_data(self):
        return self.data['signal']['chest']


def get_physiological_data(
        subject_id: str,
        physiological_sensor:str,
        placement: str,
):
    """
    Return the requested physiological data and associated label
    """
    root_path = os.path.join(DATA_PATH, "raw", "WESAD")
    subject = Subject(root_path, subject_id)
    label_df = pd.DataFrame(subject.labels, columns=['label'])

    if placement == "chest":
        subject_data = subject.get_chest_data()
    else:
        subject_data = subject.get_wrist_data()

    if physiological_sensor == 'ACC':
        columns = ['ACC_x', 'ACC_y', 'ACC_z']
        physiological_df = pd.DataFrame(subject_data[physiological_sensor], columns=columns)
    else:
        physiological_df = pd.DataFrame(subject_data[physiological_sensor], columns=[physiological_sensor])

    return physiological_df, label_df


def get_label_for_timestamp(timestamp, label_df):
    """
    Find the associated label for a given timestamp.

    Parameters:
    timestamp (float): The timestamp to find the label for
    label_df (pd.DataFrame): DataFrame with 'timestamp' and 'label' columns

    Returns:
    int: The label value closest to the given timestamp
    """
    # Find the index of the closest timestamp
    closest_idx = np.argmin(np.abs(label_df['timestamp'] - timestamp))

    # Return the corresponding label
    return label_df.iloc[closest_idx]['label']


def get_label_for_timestamp_interpolated(timestamp, label_df):
    """
    Find the associated label for a given timestamp using interpolation logic.
    This version finds which label interval the timestamp falls into.

    Parameters:
    timestamp (float): The timestamp to find the label for
    label_df (pd.DataFrame): DataFrame with 'timestamp' and 'label' columns

    Returns:
    int: The label value for the interval containing the timestamp
    """
    # If timestamp is before the first label, return the first label
    if timestamp <= label_df['timestamp'].iloc[0]:
        return label_df['label'].iloc[0]

    # If timestamp is after the last label, return the last label
    if timestamp >= label_df['timestamp'].iloc[-1]:
        return label_df['label'].iloc[-1]

    # Find the label interval that contains this timestamp
    # This finds the last label timestamp that is <= our target timestamp
    valid_indices = label_df['timestamp'] <= timestamp
    if valid_indices.any():
        last_valid_idx = valid_indices[::-1].idxmax()  # Get the last True index
        return label_df.iloc[last_valid_idx]['label']
    else:
        # Fallback to first label if no valid indices found
        return label_df['label'].iloc[0]


# For multiple timestamps (vectorized version)
def get_labels_for_timestamps(timestamps, label_df):
    """
    Vectorized version to get labels for multiple timestamps at once.

    Parameters:
    timestamps (array-like): Array of timestamps
    label_df (pd.DataFrame): DataFrame with 'timestamp' and 'label' columns

    Returns:
    np.array: Array of corresponding labels
    """
    timestamps = np.array(timestamps)
    labels = []

    for ts in timestamps:
        labels.append(get_label_for_timestamp(ts, label_df))

    return np.array(labels)


# Alternative vectorized approach using searchsorted (more efficient for large datasets)
def get_labels_for_timestamps_fast(timestamps, label_df):
    """
    Fast vectorized version using numpy's searchsorted.

    Parameters:
    timestamps (array-like): Array of timestamps
    label_df (pd.DataFrame): DataFrame with 'timestamp' and 'label' columns

    Returns:
    np.array: Array of corresponding labels
    """
    timestamps = np.array(timestamps)

    # Find insertion points
    indices = np.searchsorted(label_df['timestamp'].values, timestamps, side='right') - 1

    # Clip indices to valid range
    indices = np.clip(indices, 0, len(label_df) - 1)

    # Return corresponding labels
    return label_df['label'].iloc[indices].values

def get_participants():
    # Get all items in the directory
    root_path = os.path.join(DATA_PATH, "raw", "WESAD")
    items = os.listdir(root_path)
    participant_files = [subject for subject in items if subject[0] == "S"]
    return participant_files

def get_timestamps(df, signal_name: str, placement: str):
    if placement == "wrist":
        frequency_dict = {'ACC': 32, 'BVP': 64, 'EDA': 4, 'TEMP': 4, 'label': 700}
    else:
        frequency_dict = {'ECG': 700, "EMG": 700, "EDA": 700, "TEMP": 700, "RESP": 700, "label": 700}

    df["timestamp"] = [(1 / frequency_dict[signal_name]) * i for i in range(len(df))]
    return df

def capitalize_sensor(value):
    return str(value).upper()

def main(physiological_sensor:str, placement: str,):

    participant_lists = get_participants()
    for subject in tqdm(participant_lists, desc=f"Processing WESAD Files. Sensor {physiological_sensor} @ {placement}"):
        # ECG is chest, 700Hz samples, PPG is BVP and wrist.
        physiological_data, label_data = get_physiological_data(
            subject,
            physiological_sensor=physiological_sensor,
            placement=placement
        )

        physiological_data = get_timestamps(physiological_data, signal_name=physiological_sensor, placement=placement)
        label_data = get_timestamps(label_data, signal_name="label", placement=placement)

        physiological_labels = get_labels_for_timestamps_fast(physiological_data['timestamp'], label_data)
        physiological_data["label"] = physiological_labels

        # Save the data:
        ROOT_DIR = os.path.join(DATA_PATH, "raw", "WESAD")
        physiological_data.to_csv(os.path.join(ROOT_DIR, f"{subject}", f"{physiological_sensor}_complete_data.csv"))
        # # Get the stress data
        # bvp_df_data = bvp_df[bvp_df['label'].isin([1, 2])]
        # label_df_data = label_df[label_df["label"].isin([1,2])]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WESAD preprocessing pipeline")
    parser.add_argument("--physiological_sensor", type=capitalize_sensor, default="bvp")
    parser.add_argument("--placement", default="wrist", choices=("chest", "wrist"))
    args = parser.parse_args()

    # Helper Dictionary for overview what sensors are placed where:
    chest_sensors = ["ACC", "ECG", "EMG", "EDA", "Temp", "Resp"]
    wrist_sensors = ["ACC", "BVP", "EDA", "TEMP"]

    #Overview labels:
    # 0 = not defined / transient, 1 = baseline, 2 = stress, 3 = amusement, 4 = meditation, 5/6/7 = should be ignored in this dataset
    main(**vars(args))





