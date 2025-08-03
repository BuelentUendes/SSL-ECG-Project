import warnings

import re
from pathlib import Path

from datasets import Dataset, IterableDataset

from typing import Callable, Union

def write_csv(path: str, use_parquet: bool = True) -> Callable[[str, Union[Dataset, IterableDataset]], None]:
    """Save data to a CSV file or parquet file.
    
    Parameters
    ----------
    path : str
        The path to the CSV file.
    use_parquet : bool, optional
        If True, saves data in parquet format. If False, saves as CSV. Default is False.

    Returns
    -------
    Callable[[str, Union[Dataset, IterableDataset]], None]
        A function that saves the data to a CSV or parquet file
    """
    def inner(filename: str, ds: Union[Dataset, IterableDataset]) -> None:
        warnings.filterwarnings("ignore")
        location = Path(path)

        stem = location.stem
        if stem == '*':
            filename = filename
        else:
            # Extract the number part from the original filename
            number_match = re.search(r'(\d{5})', filename)
            if not number_match:
                raise ValueError(f'No 5-digit number found in filename: {filename}')
            filename = number_match.group(1)

        location.parent.mkdir(parents=True, exist_ok=True)
        
        if use_parquet:
            output_path = location.parent / f"{filename}.parquet"
            ds.to_parquet(output_path)
        else:
            output_path = location.parent / f"{filename}.csv"
            ds.to_csv(output_path)
            
        warnings.filterwarnings("default")
    return inner
