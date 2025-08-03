import warnings
from warnings import warn
import neurokit2 as nk

try:
    import cupy as cp
    np = cp
except ImportError:
    import numpy as np
    
from enum import Enum

class Feature(str, Enum):
    TWA = "twa"
    """T-wave alternans (TWA) feature."""

def morphology_domain(features: tuple[Feature], sampling_rate):
    """Compute morphology domain features.

    Parameters
    ----------
    features : tuple[Feature]
        A tuple with the features to be computed.

    Returns
    -------
    function
        A function that computes the features in the morphology domain.    
    """
    def inner(ECG_Clean: list[float], sampling_rate=sampling_rate):
        result = {}
        warnings.filterwarnings("ignore")
        for feature in features:
            if feature == Feature.TWA:
                twa = calculate_twa(ECG_Clean, sampling_rate=sampling_rate)
                result.update({"twa": float(twa)})
            else:
                raise ValueError(f"Feature {feature} is not valid.")
        warnings.filterwarnings("default")
        return result
    return inner


def calculate_delta_modified_moving_average_non_vectorized(delta, max_delta=32):

    if delta <= -max_delta:
        return - max_delta
    elif -max_delta < delta <= -1:
        return delta
    elif -1 < delta < 0:
        return -1
    elif delta == 0:
        return 0
    elif 0 < delta <= 1:
        return 1
    elif 1 < delta < max_delta:
        return delta
    elif delta >= max_delta:
        return max_delta

def ones_to_intervals(binary_annotation: list[int]) -> list[tuple[int,int]]:
    """
    Given a list of 0/1 flags where 1 marks S-peak, then T-offset, then S-peak, …
    return a list of (start, end) index pairs.
    """
    # 1) collect all the positions of 1’s
    idx = [i for i, flag in enumerate(binary_annotation) if flag == 1]

    # # 2) sanity check: must be an even number of markers (if odd, then it is a S, so we drop it)
    if len(idx) % 2 != 0:
        idx = idx[:-1]

    # 3) zip them into (S, T) pairs
    intervals = list(zip(idx[0::2], idx[1::2]))
    return intervals

# Old code legacy code!
# def calculate_twa(signal: list[float], tpeaks: list[int]):
#     """Compute the T-wave alternans (TWA) feature.
#
#     Parameters
#     ----------
#     signal : list[float]
#         The ECG signal.
#     tpeaks : list[int]
#         The T-wave peaks.
#
#     Returns
#     -------
#     dict
#         A dictionary containing the TWA feature.
#     """
#     # Check if signal is empty
#     if signal is None or len(signal) == 0:
#         return np.nan
#
#     # Convert to numpy array and check if conversion was successful
#     try:
#         signal = np.array(signal)
#         if signal.size == 0:
#             warn("Empty signal array", RuntimeWarning)
#             return np.nan
#
#     except Exception as e:
#         warn(f"Failed to convert signal to numpy array: {str(e)}", RuntimeWarning)
#         return np.nan
#
#     if len(tpeaks) < 2:
#         warn("Insufficient T-peaks to calculate TWA (minimum 2 required)", RuntimeWarning)
#         return np.nan
#
#     # Ensure T-peaks are within signal bounds
#     valid_tpeaks = [t for t in tpeaks if 0 <= t < len(signal)]
#     if len(valid_tpeaks) < 2:
#         warn("No valid T-peaks within signal bounds", RuntimeWarning)
#         return np.nan
#
#     # Divide the T-peaks into two buckets, even and odd.
#     even_bucket = tpeaks[1::2]
#     odd_bucket = tpeaks[::2]
#
#     # Calculate the average of the even and odd buckets.
#     average_t_even = np.mean(np.take(signal, even_bucket))
#     average_t_odd = np.mean(np.take(signal, odd_bucket))
#
#     if average_t_even is None or average_t_odd is None:
#         return np.nan
#     else:
#         # Calculate the difference in amplitude between the even and odd buckets.
#         twa = abs(average_t_even - average_t_odd)
#         return twa


def calculate_delta_modified_moving_average_vectorized(delta: np.ndarray, max_delta: float = 32.0) -> np.ndarray:
    delta_clipped = np.clip(delta, -max_delta, max_delta)
    result = np.zeros_like(delta_clipped)

    result[delta_clipped <= -1] = delta_clipped[delta_clipped <= -1]
    result[(delta_clipped > -1) & (delta_clipped < 0)] = -1
    result[(delta_clipped > 0) & (delta_clipped <= 1)] = 1
    result[(delta_clipped > 1) & (delta_clipped < max_delta)] = delta_clipped[(delta_clipped > 1) & (delta_clipped < max_delta)]
    result[delta_clipped >= max_delta] = max_delta
    result[delta_clipped <= -max_delta] = -max_delta

    return result

def calculate_twa(signal: list[float], sampling_rate: int, max_delta=32) -> dict:
    """
    Compute TWA using scalar Modified Moving Average on ST segments
    (S-peak to T-offset) instead of j-to-T, per your adaptation.
    Returns the peak absolute alternans between the even- and odd-beat MMAs.
    """
    try:
        # 1) Delineate S and T-offsets
        _, rpeaks = nk.ecg_peaks(signal, sampling_rate=sampling_rate)
        delineate_dict, _ = nk.ecg_delineate(
            signal, rpeaks, method="dwt", sampling_rate=sampling_rate
        )
        # We want to measure the T-wave alternans
        # Ideally, we measure beginning from the J point. However, NeuroKit2 does not offer this
        # So we will proceed and measure the twave and get the alternans within this segment of the ECG beat
        t_onsets = delineate_dict["ECG_T_Onsets"]
        t_offsets = delineate_dict["ECG_T_Offsets"]

        # Now put them together and then
        t_wave_interval = t_onsets + t_offsets
        intervals = ones_to_intervals(t_wave_interval)

        # Calculate the minimum difference, as we need to make sure each beat segment is of the same size for the comparison later
        min_len_st = np.min([(end-start) for start,end in intervals])

        # 2) Slice out each raw beat segment:
        # We need to make sure each beat is of the same length!
        raw_beats = []
        for (s, t) in intervals:
            if 0 <= s < t <= len(signal):
                raw_beats.append(np.asarray(signal[s:min(t, s+min_len_st)], dtype=float))

        # 3) Split into even (A) and odd (B) by index in the beat list:
        beats_A = raw_beats[0::2]  # B-series uses the 1st, 3rd, … raw beats
        beats_B = raw_beats[1::2]  # A-series uses the 2nd, 4th, … raw beats

        # Minimal length checks:
        assert np.min([a.size for a in beats_A]) == min_len_st, "Something with the minimal st length went wrong!"
        assert np.min([b.size for b in beats_B]) == min_len_st, "Something with the minimal st length went wrong!"

        mma_A = beats_A[0].copy()
        mma_B = beats_B[0].copy()

        # Update MMA for A beats
        for beat_series in beats_A[1:]:
            diffs = (beat_series - mma_A) / 8
            delta = calculate_delta_modified_moving_average_vectorized(diffs, max_delta)
            mma_A += delta

        # Update MMA for B beats
        for beat_series in beats_B[1:]:
            diffs = (beat_series - mma_B) / 8
            delta = calculate_delta_modified_moving_average_vectorized(diffs, max_delta)
            mma_B += delta

        # 6) TWA magnitude = max |MMA_A – MMA_B| (as shown in equation (4) Paper:
        # Modified moving average analysis of T-wave alternans to predict ventricular fibrillation with high accuracy
        #    (if series lengths differ by one, align on shorter)
        length = min(mma_A.size, mma_B.size)
        twa_value = np.max(np.abs(mma_A[:length] - mma_B[:length]))

        # Sanity check on the result
        if np.isnan(twa_value) or np.isinf(twa_value):
            print("Warning: Invalid TWA value computed")
            return np.nan

        else:
            return twa_value

    except Exception as e:
        print(f"Error {e} in TWA calculation return np.nan")
        return np.nan

