import warnings

import pandas as pd 
import neurokit2 as nk

try:
    import cupy as cp
    np = cp
except ImportError:
    import numpy as np
    
from enum import Enum

class Feature(str, Enum):
    ENTROPY = "entropy"
    """Entropy features."""
    POINCARE = "poincare"
    """Poincaré features."""
    FRAGMENTATION = "fragmentation"
    """Indices of Heart Rate Fragmentation (Costa, 2017)"""
    RQA = "rqa"
    """Recurrence Quantification Analysis (RQA) features."""
    DFA = "dfa"
    """Detrended Fluctuation Analysis (DFA) features."""
    HEART_ASYMMETRY = "heart_asymmetry"
    """Heart Asymmetry measure. Here we take the area index."""


def nonlinear_domain(features: tuple[Feature], sampling_rate: int = 1000):
    """Compute nonlinear domain features.

    Parameters
    ----------
    features : tuple[Feature]
        A tuple with the features to be computed.
    sampling_rate : int
        The sampling rate of the ECG signal.

    Returns
    -------
    function
        A function that computes the features in the nonlinear domain
    """
    def inner(rpeaks: list[int]):
        result = {}
        warnings.filterwarnings("ignore")
        for feature in features:
            if feature == Feature.ENTROPY:
                hrv_nonlinear = nk.hrv_nonlinear(rpeaks, sampling_rate=sampling_rate)
                # hrv_nonlinear = hrv_nonlinear.fillna(0)
                hrv_nonlinear = hrv_nonlinear.iloc[0].to_dict()

                # We do not take 'sampen' as this feature is problematic in short segments and therefore results
                # in a lot of infinity values 15% which is unacceptable
                result.update({
                    f'apen': hrv_nonlinear.get("HRV_ApEn", np.nan), #0
                    f'fuzzyen': hrv_nonlinear.get("HRV_FuzzyEn", np.nan),
                })

            elif feature == Feature.POINCARE:
                hrv_nonlinear = nk.hrv_nonlinear(rpeaks, sampling_rate=sampling_rate)
                # hrv_nonlinear = hrv_nonlinear.fillna(0)
                hrv_nonlinear = hrv_nonlinear.iloc[0].to_dict()

                result.update({
                    f'sd1': hrv_nonlinear.get("HRV_SD1", np.nan),
                    f'sd2': hrv_nonlinear.get("HRV_SD2", np.nan),
                    f'sd1_sd2': hrv_nonlinear.get("HRV_SD1SD2", np.nan),
                })
            elif feature == Feature.FRAGMENTATION:
                hrv_nonlinear = nk.hrv_nonlinear(rpeaks, sampling_rate=sampling_rate)
                # hrv_nonlinear = hrv_nonlinear.fillna(0)
                hrv_nonlinear = hrv_nonlinear.iloc[0].to_dict()

                result.update({
                    f'pss': hrv_nonlinear.get("HRV_PSS", np.nan),
                    # We do not take the other ones,as pip and ials are highly correlated with pss,
                    # PAS is somewhat different, but measures a specific alternation pattern.
                    # f'pip': hrv_nonlinear.get("HRV_PIP", np.nan),
                    # f"ials": hrv_nonlinear.get("HRV_IALS", np.nan),
                    # f"pas": hrv_nonlinear.get("HRV_PAS", np.nan),
                })
            elif feature == Feature.RQA:
                rqa = nk.hrv_rqa(rpeaks, sampling_rate=sampling_rate)
                # rqa = rqa.fillna(0)
                result.update({
                    f"w": rqa['W'].item(),
                    f"wmax": rqa['WMax'].item(),
                    f"wen": rqa['WEn'].item()
                })

            elif feature == Feature.DFA:
                hrv_nonlinear = nk.hrv_nonlinear(rpeaks, sampling_rate=sampling_rate)
                hrv_nonlinear = hrv_nonlinear.iloc[0].to_dict()
                result.update({
                    f"dfa_alpha1": hrv_nonlinear.get("HRV_DFA_alpha1", np.nan)
                })
            elif feature == Feature.HEART_ASYMMETRY:
                hrv_nonlinear = nk.hrv_nonlinear(rpeaks, sampling_rate)
                hrv_nonlinear = hrv_nonlinear.iloc[0].to_dict()
                result.update({
                    f"hrv_area_index": hrv_nonlinear.get("HRV_AI", np.nan)
                })
            else:
                raise ValueError(f"Feature {feature} is not valid.")
        warnings.filterwarnings("default")
        return result
    return inner
    