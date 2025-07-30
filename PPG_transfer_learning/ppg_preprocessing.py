import os, glob, h5py

import numpy as np
import fastavro
import neurokit2 as nk
from scipy.signal import butter, filtfilt, iirnotch
import pandas as pd
import pytz
from datetime import datetime

from utils.helper_paths import DATA_PATH
from utils.torch_utilities import create_directory

# Category Mapping for the PPG Data

CATEGORY_MAPPING_PPG = {
    "baseline": ["sitting", "recovery", "breathing"],
    "mental_stress": ["arithmetic", "neurotask"],
    "high_physical_activity": ["biking"],
    "moderate_physical_activity": ["walking"],
    "low_physical_activity": ["standing"],
}

# ───────────────────────────────
# helper funct.
# ───────────────────────────────

def get_label(condition, category_mapping):
    for key, values in category_mapping.items():
        if condition in values:
            return key

    print(f"we could not find {condition} in the mapping. Return {None}")
    return None

def combine_date_time(date, time_obj):
    """Combine date and time objects into datetime"""
    return datetime.combine(date, time_obj)

def participant_id(avro_path: str) -> str:
    """
    .../P07_empatica/raw_data/v6/file.avro  ->  P07_empatica
    """
    return os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(avro_path))))

def is_valid_segment(sig, fs):
    return (len(sig) > 0) and (fs is not None) and (fs > 0)

def butter_bandpass_safe(sig, fs, low=0.5, high=4.0, order=4):
    """Band-pass 0.5–4 Hz got at low fs."""
    if fs <= 0:
        print(f"[WARN] fs={fs} Hz not valid → skip band-pass")
        return sig
    nyq = 0.5 * fs
    high = min(high, 0.45 * fs)       
    if low >= high:                   
        print(f"[WARN] fs={fs} Hz too low for 0.5–4 Hz → skip band-pass")
        return sig
    low_norm, high_norm = low/nyq, high/nyq
    try:
        b, a = butter(order, [low_norm, high_norm], btype="band")
        return filtfilt(b, a, sig)
    except ValueError as e:
        print(f"[WARN] butter() failed ({e}) → skip band-pass")
        return sig

def notch_50_safe(sig, fs, q=30):
    """Notch 50 Hz only if allowed."""
    if fs < 100:                     
        return sig
    nyq = 0.5 * fs
    w0 = 50 / nyq
    b, a = iirnotch(w0, q)
    return filtfilt(b, a, sig)

def clean_ppg(sig, fs):
    sig = butter_bandpass_safe(sig, fs)
    sig = notch_50_safe(sig, fs)
    return sig.astype(np.float32)

# ───────────────────────────────
# .avro to raw HDF5
# ───────────────────────────────
def avro_to_hdf5(root_dir, out_h5):
    print(f"\n[INFO] AVRO → HDF5   |   root = {root_dir}")
    with h5py.File(out_h5, "w") as fout:
        for idx, fpath in enumerate(sorted(glob.glob(os.path.join(
                        root_dir, "P*_empatica", "raw_data", "v6", "*.avro"))), 1):
            part = participant_id(fpath)
            with open(fpath, "rb") as f:
                rec = next(fastavro.reader(f))
                bvp = rec["rawData"]["bvp"]
                fs, vals = bvp["samplingFrequency"], np.asarray(bvp["values"], dtype=np.float32)
                if len(vals) >0:
                    timestamps = [bvp["timestampStart"]+ i * (1e6 / fs) for i in range(len(vals))]
            if not is_valid_segment(vals, fs):
                print(f" SKIP {os.path.basename(fpath)} – empty or fs=0")
                continue

            grp  = fout.require_group(part)
            print(os.path.splitext(os.path.basename(fpath))[0])
            dset = grp.create_dataset(
                os.path.splitext(os.path.basename(fpath))[0],
                data=vals, compression="gzip", compression_opts=4, dtype='float32'
            )
            dset.attrs.update({"fs": fs, "t0_us": bvp["timestampStart"]})
            print(f"[{idx}] {part}: stored {len(vals)} samples @ {fs} Hz")
    print(f"[OK] Raw HDF5 → {out_h5}")

# ───────────────────────────────
# cleaning
# ───────────────────────────────
def clean_hdf5(in_h5, out_h5, method="custom"):
    with h5py.File(in_h5, "r") as fin, h5py.File(out_h5, "w") as fout:
        for part in fin:
            grp_out = fout.create_group(part)
            kept = 0
            for seg in fin[part]:
                sig, fs = fin[part][seg][...], fin[part][seg].attrs["fs"]
                if not is_valid_segment(sig, fs): 
                    continue
                if method == "custom":
                    grp_out.create_dataset(seg, data=clean_ppg(sig, fs),
                                           compression="gzip", compression_opts=4)
                elif method == "neurokit":
                    grp_out.create_dataset(seg, data=nk.ppg_clean(sig, sampling_rate=int(fs)),
                                           compression="gzip", compression_opts=4)
                else:
                    raise AttributeError(f"Please specify a proper preprocessing pipeline. Options: 'custom' or 'neurokit'.")

                grp_out[seg].attrs.update(fin[part][seg].attrs)
                kept += 1
            if kept == 0:
                del fout[part]                            
                print(f"  Removed {part} – no valid segments")
    print(f"[OK] Clean HDF5 → {out_h5} with method {method}")

# ───────────────────────────────
# user-specific z-score
# ───────────────────────────────
def normalize_hdf5(in_h5, out_h5):
    with h5py.File(in_h5, "r") as fin, h5py.File(out_h5, "w") as fout:
        for part in fin:
            segs = list(fin[part].keys())
            if len(segs) == 0:
                continue
            concat = np.concatenate([fin[part][s][...] for s in segs])
            mu, sigma = concat.mean(), concat.std() or 1.0
            grp_out = fout.create_group(part)
            for seg in segs:
                norm = (fin[part][seg][...] - mu) / sigma
                ds = grp_out.create_dataset(seg, data=norm.astype(np.float32),
                                            compression="gzip", compression_opts=4)
                ds.attrs.update(fin[part][seg].attrs)
                ds.attrs["mu"], ds.attrs["sigma"] = float(mu), float(sigma)
    print(f"[OK] Normalised HDF5 → {out_h5}")

# ───────────────────────────────
# windowing (10 s, 5 s stride)
# ───────────────────────────────

def window_hdf5(in_h5, out_h5, root_dir, win_sec=10, step_sec=5):
    with h5py.File(in_h5, "r") as fin, h5py.File(out_h5, "w") as fout:
        for part in fin:  # part is participant
            subfiles = list(fin[part].keys())

            fs = next((fin[part][seg].attrs["fs"] for seg in fin[part]
                       if fin[part][seg].attrs["fs"] > 0), None)

            if fs is None:
                print(f" Skip {part} – fs=0 in all segments")
                continue
            w, s = int(win_sec * fs), int(step_sec * fs)
            if w == 0 or s == 0:
                print(f" Skip {part} – w or s == 0")
                continue

            grp_out = fout.create_group(part)

            # Read the events file
            try:
                events_file_pd = pd.read_csv(
                    os.path.join(root_dir, str(part), "events", f"{part.replace('empatica', 'events')}.csv"))
                events_file_pd["timestamp"] = pd.to_datetime(events_file_pd["timestamp"], format='%H:%M:%S').dt.time
            except FileNotFoundError:
                print(f"Warning: Events file not found for {part}, skipping...")
                continue

            experimental_conditions = list(events_file_pd["conditions"].unique())

            # Create groups for each category found
            categories_present = set()
            for condition in experimental_conditions:
                category = get_label(condition, CATEGORY_MAPPING_PPG)
                if category:
                    categories_present.add(category)

            for category in categories_present:
                grp_out.create_group(category)

            # Initialize segment counters for each category
            segment_counters = {category: 0 for category in categories_present}

            for seg in fin[part]:  # seg is the specific subfile under v6
                sig = fin[part][seg][...]
                t0 = fin[part][seg].attrs["t0_us"]

                if len(sig) < w:
                    continue

                # Create timestamps for the signal
                timestamps_us = [t0 + i * (1e6 / fs) for i in range(len(sig))]
                timestamps_dt = [datetime.fromtimestamp(ts / 1e6) for ts in timestamps_us]

                # Set timezone
                amsterdam_tz = pytz.timezone('Europe/Amsterdam')
                timestamps_dt = [amsterdam_tz.localize(dt) for dt in timestamps_dt]

                # Create a temporary dataframe for this signal segment
                sig_df = pd.DataFrame({
                    'timestamp': timestamps_dt,
                    'signal': sig,
                    'label': None
                })

                # Segment the signal data based on conditions
                for condition in experimental_conditions:
                    chop = events_file_pd[events_file_pd["conditions"] == condition]
                    chop_start = chop[chop["datapoint"] == "start"]
                    chop_end = chop[chop["datapoint"] == "end"]

                    if chop_start.empty or chop_end.empty:
                        print(f"Warning: Missing start or end event for condition {condition} in participant {part}.")
                        continue

                    # Combine date with event time
                    condition_start = combine_date_time(sig_df['timestamp'].dt.date.iloc[0],
                                                        chop_start["timestamp"].iloc[0])
                    condition_end = combine_date_time(sig_df['timestamp'].dt.date.iloc[0],
                                                      chop_end["timestamp"].iloc[0])

                    # Ensure timezone is consistent
                    condition_start = amsterdam_tz.localize(condition_start)
                    condition_end = amsterdam_tz.localize(condition_end)

                    # Label the signal data for this condition
                    mask = (sig_df["timestamp"] >= condition_start) & (sig_df["timestamp"] <= condition_end)
                    sig_df.loc[mask, "label"] = condition

                # Now create windows for each condition found in this segment
                for condition in sig_df["label"].dropna().unique():
                    # Get signal data for this condition
                    condition_mask = sig_df["label"] == condition
                    condition_signal = sig_df.loc[condition_mask, "signal"].values

                    if len(condition_signal) < w:
                        continue

                    # Get the category for this condition
                    category = get_label(condition, CATEGORY_MAPPING_PPG)
                    if category is None:
                        continue

                    # Create windows for this condition
                    idx = np.arange(0, len(condition_signal) - w + 1, s)
                    windows = np.stack([condition_signal[i:i + w] for i in idx])

                    # Create dataset name using segment counter (like ECG format)
                    dataset_name = f"segment_{segment_counters[category]}"

                    # Store in the appropriate category group
                    grp_out[category].create_dataset(dataset_name, data=windows,
                                                     compression="gzip", compression_opts=4)

                    print(f"Stored {dataset_name} in {part}/{category} with shape {windows.shape}")

                    # Increment the segment counter for this category
                    segment_counters[category] += 1

    print(f"[OK] Windowed HDF5 → {out_h5}")

# ───────────────────────────────
# main
# ───────────────────────────────
if __name__ == "__main__":

    PPG_SAVE_PATH = os.path.join(DATA_PATH, "interim","PPG")
    create_directory(PPG_SAVE_PATH)

    ROOT_DIR       = "../data/raw/Empatica Avro Files"

    RAW_H5  = os.path.join(PPG_SAVE_PATH, "ppg_raw.h5")
    CLEAN_H5       = os.path.join(PPG_SAVE_PATH, "ppg_clean.h5")
    NORM_H5        = os.path.join(PPG_SAVE_PATH, "ppg_norm.h5")
    WIN_H5         = os.path.join(PPG_SAVE_PATH, "windowed_data.h5")

    METHOD = "neurokit" # option: "custom" or "neurokit" to preprocess

    WINDOW_SIZE = 10
    STEP_SIZE = 5

    avro_to_hdf5(ROOT_DIR, RAW_H5)
    clean_hdf5(RAW_H5, CLEAN_H5, METHOD)
    normalize_hdf5(CLEAN_H5, NORM_H5)
    window_hdf5(NORM_H5, WIN_H5, root_dir=ROOT_DIR, win_sec=WINDOW_SIZE, step_sec=STEP_SIZE)
