# Simple script to extract features from the cleaned data

import os
import glob
import pandas as pd
import numpy as np
import argparse

from collections import Counter
from sia import Segmenter
from sia.io import read_csv, write_csv
from sia.segmenters import SlidingWindow

from sia.features import extract_peaks, extract_hr_from_peaks, delineate, Waves
from sia.features.time_domain import hr, hrv, time_domain, Statistic, Feature as TimeFeature
from sia.features.frequency_domain import frequency_domain, Feature as FrequencyFeature
from sia.features.nonlinear_domain import nonlinear_domain, Feature as NonlinearFeature
from sia.features.morphology_domain import morphology_domain, Feature as MorphologyFeature

from utils.helper_paths import CLEANED_DATA_PATH, FEATURE_DATA_PATH
from utils.torch_utilities import create_directory

#Feature Extraction Pipeline
#The pipeline defined below segments using a Sliding Window technique and calculates the features of the data.

def main(args):

    WINDOW_SIZE = args.window_size * args.sample_frequency  # how many time points we have effectively
    STEP_SIZE = int(args.window_shift * args.sample_frequency)  # time points (units) which we shift

    input_path = os.path.join(CLEANED_DATA_PATH, str(args.sample_frequency))
    output_path = os.path.join(FEATURE_DATA_PATH, str(args.sample_frequency), str(args.window_size),
                               f"{str(args.window_shift)}")
    create_directory(output_path)

    input_file = str(args.participant_number) + ".parquet" if args.participant_number != -1 else "*.parquet"
    print(f"we are processing {input_file}" if args.participant_number != -1 else f"we are processing all files")

    Segmenter() \
        .data(read_csv(os.path.join(input_path, f'{input_file}'),
                       columns=['timestamp', 'ECG_Clean', 'ECG_R_Peaks', 'category', 'label'])) \
        .segment(SlidingWindow(WINDOW_SIZE, STEP_SIZE)) \
        .set_log_file(os.path.join(output_path, 'skip_statistics.json')) \
        .skip(lambda category: len(set(category)) > 1, "mixed_category_skip") \
        .skip(lambda label: len(set(label)) > 1, "mixed_label_skip") \
        .skip(lambda ECG_R_Peaks: len(extract_peaks(ECG_R_Peaks)) < 12, "insufficient_peaks_skip") \
        .skip(lambda ECG_R_Peaks: min(extract_hr_from_peaks(ECG_R_Peaks, sample_frequency=args.sample_frequency)) < 40,
              "low_hr_skip") \
        .skip(lambda ECG_R_Peaks: max(extract_hr_from_peaks(ECG_R_Peaks, sample_frequency=args.sample_frequency)) > 220,
              "high_hr_skip") \
        .extract('category', lambda category: Counter(category).most_common(1)[0][0]) \
            .extract('label', lambda label: Counter(label).most_common(1)[0][0]) \
            .use('rpeaks', lambda ECG_R_Peaks: extract_peaks(ECG_R_Peaks)) \
            .extract(hr([Statistic.MIN, Statistic.MAX, Statistic.MEAN, Statistic.STD], sampling_rate=args.sample_frequency)) \
            .extract(time_domain(
        [
        TimeFeature.NK_RMSSD, TimeFeature.NK_MeanNN, TimeFeature.NK_SDNN, TimeFeature.NK_MAD_NN, TimeFeature.NK_SD_RMSSD,
            TimeFeature.NK_IQR_NN, TimeFeature.NN20, TimeFeature.NK_PNN20, TimeFeature.NN50, TimeFeature.NK_PNN50,
            TimeFeature.NK_CVNN, TimeFeature.NK_CVSD
        ], sampling_rate=args.sample_frequency)) \
            .extract(frequency_domain(sampling_rate=args.sample_frequency)) \
            .extract(nonlinear_domain([NonlinearFeature.ENTROPY], sampling_rate=args.sample_frequency)) \
        .to(write_csv(os.path.join(output_path, '[0-9]{5}.csv'), use_parquet=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline for extracting features of the cleaned ECG data")
    parser.add_argument("--sample_frequency", type=int, default=1_000,
                        help="Sampling rate used for the dataset")
    parser.add_argument("--window_size", type=int, default=10, help="How many seconds we consider")
    parser.add_argument("--window_shift", type=float, default=5,
                        help="How much shift in seconds between consecutive windows.")
    parser.add_argument("--participant_number", type=int, help="which specific number to run. Set -1 for all",
                        default=30100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(args)


