# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""
RLE (Run-Length Encoding) utilities for COCO mask format

RLE (Run-Length Encoding) is a compression technique used by COCO dataset to efficiently
store binary segmentation masks. Instead of storing each pixel value, RLE stores the
lengths of consecutive runs of 0s and 1s.

COCO RLE format has two main components:
1. Compressed string encoding: A special encoding similar to LEB128 but using 6 bits per
   character with ASCII chars 48-111. This packs run lengths into a compact string format.
2. Run-length counts: Alternating counts of background (0) and foreground (1) pixels.
   The first count is always for background pixels (0), then foreground (1), and so on.

Key features of COCO RLE:
- Uses delta encoding: values beyond the first two are stored as differences from
  the value 2 positions back to improve compression
- Masks are stored in column-major (Fortran) order: pixels are read column by column
  from top to bottom, left to right
- Variable-length integer encoding: each integer can span multiple characters with
  continuation bits to handle large run lengths

Python translation of COCO maskApi.c functions
Original implementation: https://github.com/cocodataset/cocoapi/blob/8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9/common/maskApi.c#L4
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from .aria_gen2_pilot_dataset_data_types import (
    HandObjectInteractionData,
    HandObjectInteractionDataRaw,
)


def rle_from_string(encoded_string: str, height: int, width: int) -> List[int]:
    """
    Convert COCO compressed string to RLE counts array.
    Direct translation of rleFrString() from maskApi.c

    The string uses a custom encoding where each character encodes 5 bits of data
    plus a continuation bit. Delta encoding is used to compress the counts further.

    Args:
        encoded_string: RLE encoded string in COCO format
        height: Height of the image
        width: Width of the image

    Returns:
        List of RLE counts (alternating background and foreground pixel counts)
    """
    # Pre-convert string to bytes for faster access
    encoded_bytes = encoded_string.encode("latin-1")
    encoded_length = len(encoded_bytes)

    counts_array = [0] * encoded_length  # Allocate maximum possible counts
    num_counts = 0
    string_position = 0

    while string_position < encoded_length:
        decoded_value = 0
        bit_position = 0
        has_more_bits = True

        # Decode variable-length integer from consecutive characters
        while has_more_bits and string_position < encoded_length:
            char_value = encoded_bytes[string_position] - 48  # Convert from ASCII
            decoded_value |= (char_value & 0x1F) << (
                5 * bit_position
            )  # Extract 5 bits and shift
            has_more_bits = (char_value & 0x20) != 0  # Check continuation bit
            string_position += 1
            bit_position += 1

            # Sign extension for negative numbers (two's complement)
            if not has_more_bits and (char_value & 0x10):
                decoded_value |= -1 << (5 * bit_position)

        # Reverse delta encoding: add count from 2 positions back
        if num_counts > 2:
            decoded_value += counts_array[num_counts - 2]

        counts_array[num_counts] = decoded_value
        num_counts += 1

    return counts_array[:num_counts]  # Return only used portion


def rle_decode(
    run_length_counts: List[int], height: int, width: int
) -> Optional[np.ndarray]:
    """
    Convert RLE counts to binary mask.
    Direct translation of rleDecode() from maskApi.c lines 45-63

    Args:
        run_length_counts: List of RLE counts (alternating background/foreground)
        height: Height of the image
        width: Width of the image

    Returns:
        1D numpy array of binary mask, or None if invalid RLE
    """
    total_pixels = height * width
    mask = np.zeros(total_pixels, dtype=np.uint8)

    current_position = 0  # Current position in mask
    current_value = 0  # Current pixel value (0 or 1)

    # Process each run-length count using vectorized operations
    for count in run_length_counts:
        if count <= 0:  # Skip invalid counts
            continue

        end_position = current_position + count
        if end_position > total_pixels:
            # Memory boundary check - invalid RLE
            return None

        # Fill range using NumPy slicing (much faster than Python loop)
        if current_value == 1:
            mask[current_position:end_position] = 1
        # No need to explicitly set 0s since array is already zero-initialized

        current_position = end_position
        # Flip value for next run (0 -> 1, 1 -> 0)
        current_value = 1 - current_value

    return mask


def decode_coco_rle_to_mask(rle_obj: Dict[str, any]) -> np.ndarray:
    """
    Implementation of COCO RLE decoding using direct C translation

    Args:
        rle_obj: Dictionary containing 'size' and 'counts' keys

    Returns:
        2D numpy array representing the binary mask
    """
    height, width = rle_obj["size"]
    rle_string = rle_obj["counts"]

    # Convert bytes to string if needed - optimize common case
    if isinstance(rle_string, bytes):
        string_data = rle_string.decode("latin-1")
    elif isinstance(rle_string, str):
        string_data = rle_string  # No conversion needed
    else:
        string_data = str(rle_string)

    # Step 1: Convert compressed string to counts (rleFrString equivalent)
    run_length_counts = rle_from_string(string_data, height, width)

    # Step 2: Convert counts to binary mask (rleDecode equivalent)
    mask_flat = rle_decode(run_length_counts, height, width)

    if mask_flat is None:
        raise ValueError("Invalid RLE data")

    # Step 3: Reshape to 2D in Fortran order (column-major)
    mask_2d = mask_flat.reshape((height, width), order="F")

    return mask_2d


def calculate_rle_area(rle_dict: Dict) -> float:
    """
    Calculate area from RLE counts in a COCO RLE dictionary.
    Area equals the sum of all foreground pixel counts.

    Args:
        rle_dict: Dictionary containing 'size' and 'counts' keys

    Returns:
        Area as float (number of foreground pixels)
    """
    height, width = rle_dict["size"]
    rle_string = rle_dict["counts"]

    # Convert bytes to string if needed - optimize common case
    if isinstance(rle_string, bytes):
        string_data = rle_string.decode("latin-1")
    elif isinstance(rle_string, str):
        string_data = rle_string  # No conversion needed
    else:
        string_data = str(rle_string)

    run_length_counts = rle_from_string(string_data, height, width)
    # Use slice notation for better performance
    area = sum(run_length_counts[1::2])  # Sum odd-indexed counts (foreground pixels)
    return float(area)


def rle_to_bbox(rle_dict: Dict) -> Optional[Tuple[float, float, float, float]]:
    """
    Compute bounding box from RLE without full decoding.

    Args:
        rle_dict: Dictionary with 'size' and 'counts' keys

    Returns:
        Tuple of (x, y, width, height) or None if empty mask
    """
    if not isinstance(rle_dict, dict):
        return None

    size = rle_dict.get("size", [0, 0])
    if len(size) != 2:
        return None

    height, width = size
    if height <= 0 or width <= 0:
        return None

    counts = rle_dict.get("counts", [])

    if isinstance(counts, (str, bytes)):
        # Convert bytes to string if needed - optimize common case
        if isinstance(counts, bytes):
            string_data = counts.decode("latin-1")
        else:
            string_data = counts  # Already a string
        counts = rle_from_string(string_data, height, width)
    elif isinstance(counts, list):
        pass
    else:
        return None

    if not counts:
        return None

    # Find bounding box by tracking min/max coordinates of foreground pixels
    min_x, min_y = width, height
    max_x, max_y = 0, 0

    pixel_position = 0
    found_foreground = False

    for run_index, count in enumerate(counts):
        if count <= 0:  # Skip invalid counts
            continue

        if run_index % 2 == 1:  # Foreground pixels (odd indices)
            found_foreground = True
            for pixel_offset in range(count):
                position = pixel_position + pixel_offset
                if position >= height * width:  # Bounds check
                    break

                # Convert from column-major linear index to (x, y) coordinates
                y = position % height
                x = position // height

                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)

        pixel_position += count

    if not found_foreground or min_x >= max_x or min_y >= max_y:
        return None  # Empty mask

    return (
        float(min_x),
        float(min_y),
        float(max_x - min_x + 1),
        float(max_y - min_y + 1),
    )


# Convenience functions matching pycocotools API


def decode(rle_objects) -> np.ndarray:
    """
    Decode RLE to binary mask(s) - matches pycocotools.mask.decode API.

    Args:
        rle_objects: Single RLE dict or list of RLE dicts

    Returns:
        Binary mask array. If single RLE: shape (h, w). If list: shape (h, w, n)
    """
    if isinstance(rle_objects, dict):
        # Single RLE
        return decode_coco_rle_to_mask(rle_objects)
    elif isinstance(rle_objects, list):
        if not rle_objects:
            return np.array([])

        # Multiple RLEs - ensure all have same size for stacking
        masks = []
        for rle in rle_objects:
            mask = decode_coco_rle_to_mask(rle)
            masks.append(mask)

        # Check if all masks have the same shape
        if len(masks) > 1:
            first_shape = masks[0].shape
            if not all(mask.shape == first_shape for mask in masks):
                # Handle different shapes by finding common size
                # For now, just return the masks as-is without stacking
                # This matches pycocotools behavior when shapes differ
                return masks[0] if len(masks) == 1 else masks

        # Stack masks along third dimension
        return np.stack(masks, axis=2)
    else:
        raise ValueError("rle_objects must be dict or list of dicts")


def area(rle_objects) -> np.ndarray:
    """
    Calculate area(s) from RLE(s) - matches pycocotools.mask.area API.

    Args:
        rle_objects: Single RLE dict or list of RLE dicts

    Returns:
        Area(s) as numpy array
    """
    if isinstance(rle_objects, dict):
        # Single RLE
        return np.array([calculate_rle_area(rle_objects)])
    elif isinstance(rle_objects, list):
        # Multiple RLEs
        areas = [calculate_rle_area(rle) for rle in rle_objects]
        return np.array(areas)
    else:
        raise ValueError("rle_objects must be dict or list of dicts")


def toBbox(rle_objects) -> np.ndarray:
    """
    Get bounding box(es) from RLE(s) - matches pycocotools.mask.toBbox API.

    Args:
        rle_objects: Single RLE dict or list of RLE dicts

    Returns:
        Bounding box(es) as numpy array. Format: [x, y, width, height]
    """
    if isinstance(rle_objects, dict):
        # Single RLE
        bbox = rle_to_bbox(rle_objects)
        if bbox is None:
            return np.array([0.0, 0.0, 0.0, 0.0])
        return np.array(bbox)
    elif isinstance(rle_objects, list):
        # Multiple RLEs
        bboxes = []
        for rle in rle_objects:
            bbox = rle_to_bbox(rle)
            if bbox is None:
                bbox = (0.0, 0.0, 0.0, 0.0)
            bboxes.append(bbox)
        return np.array(bboxes)
    else:
        raise ValueError("rle_objects must be dict or list of dicts")


def convert_to_decoded_format(
    undecoded_data_list: List[HandObjectInteractionDataRaw],
) -> List[HandObjectInteractionData]:
    """
    Convert undecoded RLE data to decoded format by decoding RLE masks on-demand.

    Args:
        undecoded_data_list: List of HandObjectInteractionDataRaw objects

    Returns:
        List of HandObjectInteractionData objects with decoded masks
    """
    # Import here to avoid circular imports
    from .aria_gen2_pilot_dataset_data_types import HandObjectInteractionData

    if not undecoded_data_list:
        return []

    # Group by category_id and decode masks, bboxes, scores for each category
    category_groups: Dict[int, Dict[str, List]] = {}
    timestamp_ns = undecoded_data_list[0].timestamp_ns  # Get timestamp once

    for undecoded_data in undecoded_data_list:
        category_id = undecoded_data.category_id

        # Pre-allocate category group if needed
        if category_id not in category_groups:
            category_groups[category_id] = {"masks": [], "bboxes": [], "scores": []}

        category_group = category_groups[category_id]  # Cache reference

        # Decode RLE to binary mask - reuse objects
        rle = {
            "size": undecoded_data.segmentation_size,
            "counts": undecoded_data.segmentation_counts,
        }
        decoded_mask = decode_coco_rle_to_mask(rle)

        category_group["masks"].append(decoded_mask)
        category_group["bboxes"].append(undecoded_data.bbox)
        category_group["scores"].append(undecoded_data.score)

    # Create HandObjectInteractionData objects - use list comprehension for speed
    decoded_data_list = [
        HandObjectInteractionData(
            timestamp_ns=timestamp_ns,
            category_id=category_id,
            masks=data["masks"],
            bboxes=data["bboxes"],
            scores=data["scores"],
        )
        for category_id, data in category_groups.items()
    ]

    return decoded_data_list
