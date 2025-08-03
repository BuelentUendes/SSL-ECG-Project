"""Segmenters module for Stress-in-Action."""
import sys

from abc import ABC
from collections.abc import Iterable

from datasets import Dataset

from typing import Any
if sys.version_info >= (3,11):
    from typing import Self 
else:
    Self = Any

class BaseSegmenter(ABC, Iterable):
    """Base class for segmenters."""
    def set_dataset(self, dataset: Dataset) -> Self:
        """Set the dataset to segment.
        
        Parameters
        ----------
        dataset : Dataset
            The dataset to segment.
        
        Returns
        -------
        Self
            The instance of the class for method
        """
        raise NotImplementedError

class SlidingWindow(BaseSegmenter):
    """A sliding window segmenter."""
    def __init__(self, window_size: int, step_size: int = None, reset_on_label: bool = True):
        """
        Parameters
        ----------
        window_size : int
            The size of the window.
        step_size : int, optional
            The step size, by default, it is the same as the window size.
        reset_on_label : bool, optional
            If True, the sliding window will reset at each new label.
        """
        self.dataset = None
        self.window_size = window_size
        self.step_size = step_size or window_size
        self.reset_on_label = reset_on_label

    def set_dataset(self, dataset: Dataset) -> Self:
        self.dataset = dataset
        return self

    def __len__(self):
        if self.dataset is None:
            return 0
        else:
            return len(self.dataset) // self.step_size

    def __iter__(self):
        if self.dataset is None:
            return

        if self.reset_on_label:
            # Precompute label boundaries to avoid per-row iterations.
            labels = self.dataset["label"]
            boundaries = [0]
            for i in range(1, len(labels)):
                if labels[i] != labels[i - 1]:
                    boundaries.append(i)
            boundaries.append(len(labels))
            # Iterate over each label group.
            for idx in range(len(boundaries) - 1):
                group_start = boundaries[idx]
                group_end = boundaries[idx + 1]
                group_length = group_end - group_start
                for start_idx in range(0, group_length, self.step_size):
                    if start_idx + self.window_size > group_length:
                        break
                    yield self.dataset.select(
                        range(group_start + start_idx, group_start + start_idx + self.window_size))
        else:
            for start_idx in range(0, len(self.dataset), self.step_size):
                if start_idx + self.window_size > len(self.dataset):
                    break
                yield self.dataset[start_idx:start_idx + self.window_size]
