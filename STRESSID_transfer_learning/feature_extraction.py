import os
import h5py
import numpy as np
import argparse
import warnings
from typing import Dict, Any

import neurokit2 as nk
from neurokit2.hrv.hrv_utils import _hrv_format_input

from neurokit2.complexity import (
    entropy_approximate,
    entropy_fuzzy,
    entropy_sample,
    entropy_shannon,
)

from neurokit2.hrv.hrv_nonlinear import (_hrv_nonlinear_fragmentation, _hrv_nonlinear_poincare,
                                         _hrv_nonlinear_poincare_hra, _hrv_dfa)
from neurokit2.hrv.hrv_rqa import hrv_rqa

from sia.features import extract_peaks, extract_hr_from_peaks
from sia.features.time_domain import  Feature as TimeFeature
from sia.features.frequency_domain import frequency_domain
from sia.features.morphology_domain import calculate_twa
from sia.features.morphology_domain import morphology_domain, Feature as MorphologyFeature

from utils.helper_paths import DATA_PATH

def detect_r_peaks(ecg_signal: np.ndarray, sampling_rate: int) -> np.ndarray:
    """
    Detect R-peaks from raw ECG signal using NeuroKit2.
    """
    try:
        _, rpeaks = nk.ecg_peaks(ecg_signal, sampling_rate=sampling_rate)
        r_peaks_binary = np.zeros_like(ecg_signal)
        if 'ECG_R_Peaks' in rpeaks:
            peak_indices = rpeaks['ECG_R_Peaks']
            r_peaks_binary[peak_indices] = 1
        return r_peaks_binary
    except Exception as e:
        print(f"Error in R-peak detection: {e}")
        return np.zeros_like(ecg_signal)


def get_non_linear_features(rpeaks: np.ndarray, sampling_rate: int, window_size: int=30):
    hrv_nonlinear_features = {}
    rri, rri_time, rri_missing = _hrv_format_input(rpeaks, sampling_rate=sampling_rate)
    # Complexity
    tolerance = 0.2 * np.std(rri, ddof=1)
    hrv_nonlinear_features["ApEn"], _ = entropy_approximate(rri, delay=1, dimension=2, tolerance=tolerance)
    hrv_nonlinear_features["SampEn"], _ = entropy_sample(rri, delay=1, dimension=2, tolerance=tolerance)
    hrv_nonlinear_features["ShanEn"], _ = entropy_shannon(rri)
    hrv_nonlinear_features["FuzzyEn"], _ = entropy_fuzzy(rri, delay=1, dimension=2, tolerance=tolerance)

    # Get Fragmentation features
    hrv_nonlinear_features = _hrv_nonlinear_fragmentation(rri, rri_time,rri_missing=rri_missing, out=hrv_nonlinear_features)

    # Get Poincare features
    hrv_nonlinear_features = _hrv_nonlinear_poincare(rri, rri_time,rri_missing=rri_missing, out=hrv_nonlinear_features)

    # Area index
    hrv_nonlinear_features = _hrv_nonlinear_poincare_hra(rri, rri_time, rri_missing, out=hrv_nonlinear_features)

    # RQA features
    rqa_features = hrv_rqa(rpeaks, sampling_rate=sampling_rate)
    hrv_nonlinear_features.update({
        f"w": rqa_features['W'].item(),
        f"wmax": rqa_features['WMax'].item(),
        f"wen": rqa_features['WEn'].item()
    })

    # DFA alpha1 feature (suppress DFA_alpha2 warning for short windows)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="DFA_alpha2 related indices will not be calculated")
        hrv_nonlinear_features = _hrv_dfa(rri, out=hrv_nonlinear_features)

    return hrv_nonlinear_features

def clean_ecg_signal(ecg_signal: np.ndarray, sampling_rate: int) -> np.ndarray:
    """
    Clean ECG signal using NeuroKit2 preprocessing.
    """
    try:
        cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)
        return cleaned
    except Exception as e:
        print(f"Error in ECG cleaning: {e}")
        return ecg_signal


def extract_features_from_window(ecg_window: np.ndarray, sampling_rate: int) -> Dict[str, Any]:
    """
    Extract features from a single ECG window.
    
    Args:
        ecg_window: ECG signal window of shape (window_length,)
        sampling_rate: Sampling frequency
    
    Returns:
        Dictionary of extracted features
    """
    features = {}
    
    try:
        # Clean the ECG signal
        ecg_clean = clean_ecg_signal(ecg_window, sampling_rate)
        
        # Detect R-peaks
        r_peaks_binary = detect_r_peaks(ecg_clean, sampling_rate)
        
        # Extract peak indices
        r_peaks = extract_peaks(r_peaks_binary)

        rri, rri_time, rri_missing = _hrv_format_input(r_peaks, sampling_rate=sampling_rate)

        # Quality checks similar to original pipeline
        if len(r_peaks) < 12:
            return None  # insufficient_peaks_skip
            
        hr_values = extract_hr_from_peaks(r_peaks_binary, sample_frequency=sampling_rate)
        if min(hr_values) < 40 or max(hr_values) > 220:
            return None  # low_hr_skip or high_hr_skip
        
        # Extract HR statistics
        hr_stats = {
            'hr_min': min(hr_values),
            'hr_max': max(hr_values), 
            'hr_mean': np.mean(hr_values),
            'hr_std': np.std(hr_values)
        }
        features.update(hr_stats)
        
        # Extract time domain features
        time_features = [
            TimeFeature.NK_RMSSD, TimeFeature.NK_MeanNN, TimeFeature.NK_SDNN, 
            TimeFeature.NK_MAD_NN, TimeFeature.NK_SD_RMSSD, TimeFeature.NK_IQR_NN, 
            TimeFeature.NN20, TimeFeature.NK_PNN20, TimeFeature.NN50, 
            TimeFeature.NK_PNN50, TimeFeature.NK_CVNN, TimeFeature.NK_CVSD
        ]

        time_domain_features = nk.hrv_time(r_peaks, sampling_rate=sampling_rate)

        # Calculate time domain features
        for feature in time_features:
            try:
                if feature == TimeFeature.NK_RMSSD:
                    features["nk_rmssd"] = time_domain_features["HRV_RMSSD"].iloc[0]
                elif feature == TimeFeature.NK_MeanNN:
                    features['nk_mean_nn'] = time_domain_features["HRV_MeanNN"].iloc[0]
                elif feature == TimeFeature.NK_SDNN:
                    features['nk_sd_nn'] = time_domain_features["HRV_SDNN"].iloc[0]
                elif feature == TimeFeature.NK_MAD_NN:
                    features['nk_mad_nn'] = time_domain_features["HRV_MadNN"].iloc[0]
                elif feature == TimeFeature.NK_SD_RMSSD:
                    features["nk_sd_rmssd"] = time_domain_features["HRV_SDRMSSD"].iloc[0]
                elif feature == TimeFeature.NK_IQR_NN:
                    features['nk_iqr_nn'] = time_domain_features["HRV_IQRNN"].iloc[0]
                elif feature == TimeFeature.NN20:
                    diff_rri = np.diff(rri)
                    features["nn20"] = np.sum(np.abs(diff_rri) > 20)
                elif feature == TimeFeature.NK_PNN20:
                    features['nk_pnn20'] = time_domain_features["HRV_pNN20"].iloc[0]
                elif feature == TimeFeature.NN50:
                    diff_rri = np.diff(rri)
                    features["nn50"] = np.sum(np.abs(diff_rri) > 50)
                elif feature == TimeFeature.NK_PNN50:
                    features['nk_pnn50'] = time_domain_features["HRV_pNN50"].iloc[0]
                elif feature == TimeFeature.NK_CVNN:
                    features['nk_cvnn'] = time_domain_features["HRV_CVNN"].iloc[0]
                elif feature == TimeFeature.NK_CVSD:
                    features['nk_cvsd'] = time_domain_features["HRV_CVSD"].iloc[0]
            except Exception as e:
                print(f"Error extracting {feature}: {e}")
                features[str(feature).split('.')[-1]] = np.nan

        non_linear_features = get_non_linear_features(r_peaks, sampling_rate)
        features["apen"] = non_linear_features["ApEn"]
        features["fuzzyen"] = non_linear_features["FuzzyEn"]
        features["pip"] = non_linear_features["PIP"]
        features["ials"] = non_linear_features["IALS"]
        features["pss"] = non_linear_features["PSS"]
        features["pas"] = non_linear_features["PAS"]
        features["sd1"] = non_linear_features["SD1"]
        features["sd2"] = non_linear_features["SD2"]
        features["sd1_sd2"] = non_linear_features["SD1SD2"]
        features["area_index"] = non_linear_features["AI"]
        features["w"] = non_linear_features["w"]
        features["wmax"] = non_linear_features["wmax"]
        features["wen"] = non_linear_features["wen"]
        features["dfa_alpha1"] = non_linear_features["DFA_alpha1"]

        # Extract frequency domain features using sia package
        freq_extractor = frequency_domain(sampling_rate=sampling_rate)
        frequency_features = freq_extractor(r_peaks)
        
        # Add frequency domain features to the feature dictionary
        for key, value in frequency_features.items():
            features[f"{key}"] = value

        features["twa"] = calculate_twa(ecg_clean, sampling_rate)

        return features
        
    except Exception as e:
        print(f"Error processing window: {e}")
        return None


def process_stressid_data(h5_file_path: str, output_path: str, sampling_rate: int = 500):
    """
    Process STRESSID windowed data and extract features, maintaining windowed structure.
    
    Args:
        h5_file_path: Path to the windowed STRESSID H5 file
        output_path: Directory to save extracted features
        sampling_rate: Sampling rate (default 500 Hz for STRESSID)
    """
    os.makedirs(output_path, exist_ok=True)
    
    output_file = os.path.join(output_path, 'windowed_data.h5')
    
    # Get feature names from a sample extraction to determine output shape
    sample_features = None
    feature_names = []
    
    with h5py.File(h5_file_path, 'r') as f_in, h5py.File(output_file, 'w') as f_out:
        for participant_id in f_in.keys():
            print(f"Processing {participant_id}")
            participant_group = f_in[participant_id]
            
            # Create participant group in output file
            out_participant_group = f_out.create_group(participant_id)
            
            for condition in participant_group.keys():  # baseline, mental_stress, etc.
                condition_group = participant_group[condition]
                
                # Create condition group in output file
                out_condition_group = out_participant_group.create_group(condition)
                
                for segment_name in condition_group.keys():  # segment_0, segment_1, etc.
                    segment_data = condition_group[segment_name][:]  # Shape: (n_windows, window_length)
                    n_windows = segment_data.shape[0]
                    
                    # Extract features for all windows in this segment
                    segment_features_list = []
                    valid_windows = 0
                    
                    for window_idx in range(n_windows):
                        ecg_window = segment_data[window_idx]
                        
                        # Extract features from this window
                        window_features = extract_features_from_window(ecg_window, sampling_rate)
                        
                        if window_features is not None:
                            # Remove metadata keys, keep only feature values
                            feature_dict = {k: v for k, v in window_features.items() 
                                          if k not in ['participant_id', 'condition', 'segment', 'window_idx', 'label']}
                            
                            # Get feature names from first valid window
                            if sample_features is None:
                                sample_features = feature_dict
                                feature_names = list(feature_dict.keys())
                                print(f"Feature names: {feature_names}")
                                print(f"Number of features: {len(feature_names)}")
                                
                                # Add feature names as attribute to all participant groups
                                for pid in f_out.keys():
                                    f_out[pid].attrs['feature_names'] = feature_names
                            
                            # Convert to array in consistent order
                            feature_values = [feature_dict.get(name, np.nan) for name in feature_names]
                            segment_features_list.append(feature_values)
                            valid_windows += 1
                        else:
                            # Add NaN values for failed feature extraction
                            if sample_features is not None:
                                feature_values = [np.nan] * len(feature_names)
                                segment_features_list.append(feature_values)
                    
                    # Convert to numpy array and save to H5
                    if segment_features_list:
                        segment_features_array = np.array(segment_features_list, dtype=np.float32)
                        out_condition_group.create_dataset(segment_name, data=segment_features_array)
                        print(f"  {segment_name}: {segment_features_array.shape} ({valid_windows}/{n_windows} valid windows)")
                    else:
                        print(f"  {segment_name}: No valid features extracted")
    
    if feature_names:
        print(f"\nFeatures saved to {output_file}")
        print(f"Feature names saved as H5 attributes")
        print(f"Total features per window: {len(feature_names)}")
    else:
        print("No features extracted!")


def main(args):
    """
    Main function to extract features from STRESSID dataset.
    """
    input_path = os.path.join(DATA_PATH, 'interim/STRESSID/ECG', str(args.sample_frequency),
                             str(args.window_size), str(args.window_shift))
    h5_file_path = os.path.join(input_path, 'windowed_data.h5')
    
    output_path = os.path.join(DATA_PATH, 'interim/STRESSID_features/ECG', str(args.sample_frequency),
                              str(args.window_size), str(args.window_shift))
    
    if not os.path.exists(h5_file_path):
        print(f"H5 file not found: {h5_file_path}")
        return
    
    print(f"Processing STRESSID data from: {h5_file_path}")
    print(f"Output directory: {output_path}")
    
    process_stressid_data(h5_file_path, output_path, args.sample_frequency)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from STRESSID windowed ECG data")
    parser.add_argument("--sample_frequency", type=int, default=500,
                        help="Sampling rate used for the dataset")
    parser.add_argument("--window_size", type=int, default=30,
                        help="Window size in seconds")
    parser.add_argument("--window_shift", type=int, default=10,
                        help="Window shift in seconds")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    main(args)