import warnings
import neurokit2 as nk
from neurokit2 import ecg_peaks, signal_rate, ecg_quality, ecg_delineate, ecg_phase
import pandas as pd

from typing import Callable


def preprocessing_pipeline(ecg_cleaned, sampling_rate: int = 1000, method: str = "neurokit"):
    """
    Runs through the preprocessing pipeline for an already cleaned signal
    :param signal: Cleaned ECG signal
    :return: pd dataframe
    """

    # Detect R-peaks
    instant_peaks, info = ecg_peaks(
        ecg_cleaned=ecg_cleaned,
        sampling_rate=sampling_rate,
        method=method,
        correct_artifacts=True,
    )

    # Calculate heart rate
    rate = signal_rate(
        info, sampling_rate=sampling_rate, desired_length=len(ecg_cleaned)
    )

    # Assess signal quality
    quality = ecg_quality(
        ecg_cleaned, rpeaks=info["ECG_R_Peaks"], sampling_rate=sampling_rate
    )

    # Merge signals in a DataFrame
    signals = pd.DataFrame(
        {
            "ECG_Clean": ecg_cleaned,
            "ECG_Rate": rate,
            "ECG_Quality": quality,
        }
    )

    # Delineate QRS complex
    delineate_signal, delineate_info = ecg_delineate(
        ecg_cleaned=ecg_cleaned, rpeaks=info["ECG_R_Peaks"], sampling_rate=sampling_rate
    )
    info.update(delineate_info)  # Merge waves indices dict with info dict

    # Determine cardiac phases
    cardiac_phase = ecg_phase(
        ecg_cleaned=ecg_cleaned,
        rpeaks=info["ECG_R_Peaks"],
        delineate_info=delineate_info,
    )

    # Add additional information to signals DataFrame
    signals = pd.concat(
        [signals, instant_peaks, delineate_signal, cardiac_phase], axis=1
    )

    # return signals DataFrame and R-peak locations
    return signals, info


def neurokit(sampling_rate: int = 1000,
             method: str = "neurokit") -> Callable[[list], dict]:
    """Compute ECG features using the NeuroKit algorithm.

    Parameters
    ----------
    sampling_rate : int, optional
        The sampling rate of the ECG signal, by default 1000.
    method: str, optional
        Which method to use for cleaning and preprocessing the ECG signal.
        Choices: 'neurokit', 'engzeemod2012', 'elgendi2010', 'hamilton2002', 'pantompkins1985'

    Returns
    -------
    function
        A function that computes the ECG features using the NeuroKit algorithm.
    """
    def inner(signal: list[float]): 
        warnings.filterwarnings("ignore")
        df, _ = nk.ecg_process(signal, sampling_rate=sampling_rate, method=method)
        warnings.filterwarnings("default")
        return df.to_dict('list')
    return inner