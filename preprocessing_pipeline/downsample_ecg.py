import os
import warnings

from typing import List
import argparse

import neurokit2 as nk
import numpy as np
from pyedflib import highlevel
from tqdm import tqdm

from utils.helper_paths import RAW_DATA_PATH


def find_all_edf_files(directory: str) -> List[str]:
    """
    Find all EDF files in the given directory.

    Args:
        directory: Path to the directory to search for EDF files

    Returns:
        List of full paths to EDF files
    """
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.edf')]


def create_directory(path: str) -> None:
    """
    Create a directory if it doesn't exist.

    Args:
        path: Path to create
    """
    if not os.path.exists(path):
        os.makedirs(path)


def downsample_ecg_file(
        input_path: str,
        output_path: str,
        desired_sampling_rate: int,
        method: str = "interpolation",
) -> None:
    """
    Downsample an ECG signal from an EDF file and save it.

    Args:
        input_path: Path to input EDF file
        output_path: Path where downsampled EDF file should be saved
        desired_sampling_rate: Target sampling rate in Hz
        method: downsampling method, default: FFT, could also be 'interpolated'.
        Important, downsampling method with FFT does not really then get 64Hz, but effectively samples it to 62.5



    Notes:
        - Assumes ECG signal is the first channel in the EDF file
        - Original sampling rate is assumed to be 1000 Hz
        - NaN values are replaced with zeros
    """
    signals, signal_headers, header = highlevel.read_edf(input_path)

    # Clean and downsample the ECG signal
    # Now I need to design a lowpass filter to cut all frequencies above,
    # so they do not creep into my downsampled signal (anti-aliasing!)

    # The nyquist frequency is important
    nyquist_frequency = float(desired_sampling_rate / 2)
    cleaned_signal = nk.signal_filter(signals[0], sampling_rate=1000, highcut=nyquist_frequency, order=2)

    downsampled_ecg = nk.signal_resample(
        cleaned_signal,
        sampling_rate=1000,
        desired_sampling_rate=desired_sampling_rate,
        method=method
    )
    downsampled_ecg = np.nan_to_num(downsampled_ecg).reshape(1, -1)

    # Update header for the new sampling rate
    new_header = signal_headers[0].copy()
    # new_header['sample_rate'] = desired_sampling_rate
    new_header["sample_frequency"] = desired_sampling_rate

    # Write the downsampled EDF file
    highlevel.write_edf(output_path, downsampled_ecg, [new_header], header)


def main(args: argparse.Namespace) -> None:
    """
    Main function to process all EDF files in the input directory.

    Args:
        args: Command line arguments containing desired_sampling_rate
    """

    # Important: As FFT uses a factor of 2, we need to issue a warning!"
    # we assume sampling rate of 1000
    if args.downsampling_method == "FFT":
        if args.original_sampling_rate % args.desired_sampling_rate != 0:
            ratio = np.round(args.original_sampling_rate / args.desired_sampling_rate)
            true_sampling_rate = args.original_sampling_rate / ratio

            warnings.warn("The downsampled frequency will have a different rate than the desired.\n"
                          f"The downsampled frequency will have {true_sampling_rate}"
                          "\nUse either 'interpolation' or choose a different desired sampling rate."
                          "Hence, the desired sampling rate be will be adjusted!")

            args.desired_sampling_rate = true_sampling_rate

    if args.data_chunk == -1:
        edf_files = find_all_edf_files(os.path.join(RAW_DATA_PATH, "1000"))
    else:
        edf_files = [
            file
            for chunk in range(1, 6)
            for file in find_all_edf_files(os.path.join(RAW_DATA_PATH, "1000", f"part_{str(chunk)}"))
        ]
    output_path = os.path.join(RAW_DATA_PATH, str(args.desired_sampling_rate))
    create_directory(output_path)

    for edf_file in tqdm(edf_files, desc=f"Processing EDF files. Target frequency {args.desired_sampling_rate}"):
        save_path = os.path.join(output_path, os.path.basename(edf_file))
        downsample_ecg_file(edf_file, save_path, args.desired_sampling_rate, args.downsampling_method)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Simple script to downsample ECG signals and save them in EDF files "
            "for which we can then use the preprocessing pipeline."
        )
    )
    parser.add_argument(
        "--desired_sampling_rate",
        type=int,
        help="Desired sampling rate for downsampling in Hz.",
        default=64
    )

    parser.add_argument(
        "--data_chunk",
        type=int,
        default=-1,
        help="Which data chunk to process. 1 for part 1, 2 for part 2, and -1 for all."
             "Important: -1 will most likely lead to memory issues."
    )

    parser.add_argument(
        "--downsampling_method",
        type=str,
        default="interpolate",  # Original was FFT
        help="Downsampling method to use in neurokit. Important FFT use can result in slightly deviating results due to padding."
             "Interpolation seemed to have worked well and results in the precise downsample result."
             "However FFT is the most accurate (if sample is periodic)."
             "For instance, if I do 128, it rounds to 125 (as it is a factor of 2)."
             "Interpolate is exact to 64"
    )

    parser.add_argument(
        "--original_sampling_rate",
        type=int,
        default=1000,
        help="Original sampling rate for the original signal. Assumed to be 1_000 Hz."
    )

    args = parser.parse_args()
    main(args)

