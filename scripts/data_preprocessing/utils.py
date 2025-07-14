"""Utility functions that are used by multiple data generation scripts."""

import json
import logging
from pathlib import Path

import numpy as np
import requests


def download_file(url: str, local_filename: str | Path) -> None:
    """Downloads a file from a URL and saves it locally.

    Args:
        url: The URL to download the file from.
        local_filename: The path to save the downloaded file.
    """
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()
    with open(str(local_filename), "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    logging.info("File downloaded successfully and saved as %s", local_filename)


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy data types."""

    def default(self, o):
        """Convert NumPy types to Python native types for JSON serialization.

        Returns:
            Python native type (list, int, float, dict, or bool) for JSON serialization.
        """
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(
            o,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(o)
        if isinstance(o, (np.float_, np.float16, np.float32, np.float64)):
            return float(o)
        if isinstance(o, (np.complex_, np.complex64, np.complex128)):
            return {"real": o.real, "imag": o.imag}
        if isinstance(o, np.bool_):
            return bool(o)
        return super(NumpyEncoder, self).default(o)
