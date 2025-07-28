import os, glob, h5py
from datetime import datetime, timezone

import numpy as np
import fastavro
import neurokit2 as nk
from scipy.signal import butter, filtfilt, iirnotch
from torch.fx.experimental.sym_node import METHOD_TO_OPERATOR


# ───────────────────────────────
# helper funct.
# ───────────────────────────────
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
            if not is_valid_segment(vals, fs):
                print(f" SKIP {os.path.basename(fpath)} – empty or fs=0")
                continue

            grp  = fout.require_group(part)
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
                    print(f"We use our custom preprocessing pipeline")
                    grp_out.create_dataset(seg, data=clean_ppg(sig, fs),
                                           compression="gzip", compression_opts=4)
                elif method == "neurokit":
                    print(f"We use the neurokit2 preprocessing pipeline")
                    grp_out.create_dataset(seg, data=nk.ppg_clean(sig, sampling_rate=int(fs)),
                                           compression="gzip", compression_opts=4)
                else:
                    raise AttributeError(f"Please specify a proper preprocessing pipeline. Options: 'custom' or 'neurokit'.")

                grp_out[seg].attrs.update(fin[part][seg].attrs)
                kept += 1
            if kept == 0:
                del fout[part]                            
                print(f"  Removed {part} – no valid segments")
    print(f"[OK] Clean HDF5 → {out_h5}")

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
def window_hdf5(in_h5, out_h5, win_sec=10, step_sec=5):
    with h5py.File(in_h5, "r") as fin, h5py.File(out_h5, "w") as fout:
        for part in fin:
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
            for seg in fin[part]:
                sig = fin[part][seg][...]
                if len(sig) < w:
                    continue
                idx = np.arange(0, len(sig) - w + 1, s)
                windows = np.stack([sig[i:i+w] for i in idx])
                grp_out.create_dataset(seg, data=windows, compression="gzip", compression_opts=4)
    print(f"[OK] Windowed HDF5 → {out_h5}")

# ───────────────────────────────
# main
# ───────────────────────────────
if __name__ == "__main__":
    ROOT_DIR       = "../data/raw/Empatica Avro Files"
    RAW_H5         = "../data/interim/ppg_raw.h5"
    CLEAN_H5       = "../data/interim/ppg_clean.h5"
    NORM_H5        = "../data/interim/ppg_norm.h5"
    WIN_H5         = "../data/interim/ppg_windows.h5"

    METHOD = "neurokit" # option: "custom" or "neurokit" to preprocess

    avro_to_hdf5(ROOT_DIR, RAW_H5)
    clean_hdf5(RAW_H5, CLEAN_H5, METHOD)
    normalize_hdf5(CLEAN_H5, NORM_H5)
    window_hdf5(NORM_H5, WIN_H5, win_sec=10, step_sec=5)
