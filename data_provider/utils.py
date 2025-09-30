# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Utility functions for Aria Gen2 Pilot Dataset data providers."""

import bisect
import os
from typing import Any, List, Tuple

import numpy as np
from PIL import Image
from projectaria_tools.core.sensor_data import TimeQueryOptions


def find_timestamp_index_by_time_query_option(
    timestamps_ns: List[int],
    target_timestamp_ns: int,
    time_query_option: TimeQueryOptions,
) -> int:
    """Find best matching timestamp index using binary search."""
    if not timestamps_ns:
        return -1

    if time_query_option == TimeQueryOptions.BEFORE:
        idx = bisect.bisect_right(timestamps_ns, target_timestamp_ns)
        return idx - 1 if idx > 0 else -1
    elif time_query_option == TimeQueryOptions.AFTER:
        idx = bisect.bisect_left(timestamps_ns, target_timestamp_ns)
        return idx if idx < len(timestamps_ns) else -1
    else:  # TimeQueryOptions.CLOSEST
        idx = bisect.bisect_left(timestamps_ns, target_timestamp_ns)
        if idx == 0:
            return 0
        if idx == len(timestamps_ns):
            return len(timestamps_ns) - 1
        return (
            idx - 1
            if target_timestamp_ns - timestamps_ns[idx - 1]
            <= timestamps_ns[idx] - target_timestamp_ns
            else idx
        )


def find_data_by_timestamp_ns(
    timestamps_data: List[Tuple[int, Any]],
    target_timestamp_ns: int,
    time_query_option: TimeQueryOptions,
) -> int:
    """Find best matching data index using binary search.

    Args:
        timestamps_data: List[Tuple[int, Any]] where first element in Tuple is timestamp_ns, second is data
        target_timestamp_ns: Target timestamp to search for
        time_query_option: Search strategy (BEFORE, AFTER, CLOSEST)

    Returns:
        Index of best match, or -1 if no valid match found
    """
    # Extract timestamps from tuples for bisect operations
    timestamps_ns = [item[0] for item in timestamps_data]

    target_timestamp_index = find_timestamp_index_by_time_query_option(
        timestamps_ns, target_timestamp_ns, time_query_option
    )
    return (
        timestamps_data[target_timestamp_index][1].copy()
        if target_timestamp_index >= 0
        else None
    )


def check_valid_file(file_path: str) -> None:
    """Check if file path is valid and file has content.

    Args:
        file_path: Path to the file to validate

    Raises:
        RuntimeError: If file doesn't exist, path is invalid, or file is empty
    """
    if not file_path or not isinstance(file_path, str):
        raise RuntimeError("Invalid file path: path must be a non-empty string")

    if not os.path.exists(file_path):
        raise RuntimeError(f"File does not exist: {file_path}")

    if not os.path.isfile(file_path):
        raise RuntimeError(f"Path is not a file: {file_path}")

    try:
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            raise RuntimeError(f"File is empty: {file_path}")
    except OSError as e:
        raise RuntimeError(f"Error checking file size: {file_path} - {e}")


def check_valid_csv(file_path: str, expected_headers: str) -> None:
    """Check if CSV file is valid and has expected headers.

    Args:
        file_path: Path to the CSV file to validate
        expected_headers: Comma-separated string of expected column headers in the CSV

    Raises:
        RuntimeError: If file is invalid or headers don't match
    """
    # First check if file is valid
    check_valid_file(file_path)

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()

            if first_line != expected_headers:
                raise RuntimeError(
                    f"CSV headers don't match. Expected: '{expected_headers}', Found: '{first_line}'"
                )

    except UnicodeDecodeError as e:
        raise RuntimeError(f"File encoding error: {file_path} - {e}")
    except Exception as e:
        if isinstance(e, RuntimeError):
            raise
        raise RuntimeError(f"Error reading CSV file: {file_path} - {e}")


def load_image(
    folder_path: str, filename_pattern: str, index: int, dtype: type
) -> np.ndarray:
    """Load image from file using specified pattern and data type.

    Supports various image formats including PNG, JPEG, TIFF, etc. through PIL.

    Args:
        folder_path: Directory containing the image files
        filename_pattern: Format string for filename (e.g., "depth_{:08d}.png", "image_{:08d}.jpg")
        index: Index to format into the filename pattern
        dtype: NumPy data type for the output array (e.g., np.uint16, np.uint8)

    Returns:
        Numpy array with specified dtype

    Raises:
        RuntimeError: If file doesn't exist or loading fails
    """
    file_path = os.path.join(folder_path, filename_pattern.format(index))

    if not os.path.exists(file_path):
        raise RuntimeError(f"{file_path} not exist")
    try:
        with Image.open(file_path) as img:
            image_array = np.array(img, dtype=dtype)
        return image_array

    except Exception as e:
        raise RuntimeError(f"Failed to load image from {file_path}: {e}")
