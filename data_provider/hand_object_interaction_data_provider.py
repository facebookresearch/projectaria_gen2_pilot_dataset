# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import json
import os
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from projectaria_tools.core.sensor_data import TimeQueryOptions
from tqdm import tqdm

# Use self-contained RLE utilities instead of external pycocotools
from . import rle_utils

from .aria_gen2_pilot_dataset_data_types import (
    HandObjectInteractionData,
    HandObjectInteractionDataRaw,
)
from .utils import find_timestamp_index_by_time_query_option


class HandObjectInteractionDataProvider:
    """Hand-object interaction data provider for Aria Gen2 Pilot Dataset."""

    def __init__(self, data_path: str, rgb_width: int, rgb_height: int):
        """
        Initialize with path to hand_object_interaction_results.json and rgb size.

        hand_object_interaction_results.json contains COCO-format detection results:
        - List of annotations with segmentation, bbox, score, image_id, category_id
        - category_id: 1=left_hand, 2=right_hand, 3=interacting_object
        - timestamp_ns = image_id * 1e6

        rgb image size is needed to resize the segmentation mask to the same size as rgb image
        """
        # Store data and timestamps in separate lists following the standard pattern
        # Now storing decoded data directly instead of raw RLE data
        self.hoi_data_list: List[List[HandObjectInteractionData]] = []
        self.timestamps_ns: List[int] = []
        self.rgb_width = rgb_width
        self.rgb_height = rgb_height

        self._load_data(data_path)

    def _load_data(self, data_path: str) -> None:
        """Load hand-object interaction JSON and build timestamp-indexed data."""
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"File not found: {data_path}")

        try:
            with open(data_path, "r") as f:
                annotations = json.load(f)

            if not isinstance(annotations, list):
                raise ValueError(
                    "hand-object interaction results not an array of objects"
                )
            temp_data: Dict[int, List[HandObjectInteractionDataRaw]] = {}

            # Process annotations with progress bar
            print("Loading hand-object interaction annotations...")
            for annotation in tqdm(annotations, desc="Processing annotations"):
                original_image_id = int(annotation["image_id"])
                timestamp_ns = int(original_image_id * 1e6)

                segmentation = annotation["segmentation"]
                if (
                    not isinstance(segmentation, dict)
                    or "size" not in segmentation
                    or "counts" not in segmentation
                ):
                    raise ValueError("Invalid segmentation format in annotation")

                hoi_data = HandObjectInteractionDataRaw(
                    timestamp_ns=timestamp_ns,
                    original_image_id=original_image_id,
                    category_id=int(annotation["category_id"]),
                    bbox=annotation["bbox"],
                    segmentation_size=segmentation["size"],
                    segmentation_counts=segmentation["counts"],
                    score=float(annotation["score"]),
                )

                temp_data.setdefault(timestamp_ns, []).append(hoi_data)

            # Convert raw data to decoded format during loading
            sorted_temp_data = sorted(temp_data.items())
            self.hoi_data_list = []
            self.timestamps_ns = []

            # Decode RLE data with progress bar
            print("Decoding RLE masks...")
            for timestamp_ns, raw_interactions in tqdm(
                sorted_temp_data, desc="Decoding masks"
            ):
                # Decode RLE data immediately during loading
                decoded_interactions = rle_utils.convert_to_decoded_format(
                    raw_interactions
                )
                self.hoi_data_list.append(decoded_interactions)
                self.timestamps_ns.append(timestamp_ns)

        except Exception as e:
            raise RuntimeError(
                f"Failed to load hand-object interaction data from {data_path}: {e}"
            )
        if len(self.timestamps_ns) == 0:
            raise RuntimeError(
                f"No hand-object interaction data found in {data_path}, can not initialize HandObjectInteractionDataProvider."
            )

    def get_hoi_data_by_timestamp_ns(
        self,
        timestamp_ns: int,
        time_query_options: TimeQueryOptions = TimeQueryOptions.CLOSEST,
        resize_masks: bool = True,
    ) -> Optional[List[HandObjectInteractionData]]:
        """Get all interactions at timestamp (hands + objects)."""
        index = find_timestamp_index_by_time_query_option(
            self.timestamps_ns, timestamp_ns, time_query_options
        )
        return self.get_hoi_data_by_index(index, resize_masks)

    def get_hoi_data_by_index(
        self, index: int, resize_masks: bool = True
    ) -> Optional[List[HandObjectInteractionData]]:
        """Get interactions by index."""
        if 0 <= index < len(self.hoi_data_list):
            # Data is already decoded, just retrieve it
            decoded_data_list = self.hoi_data_list[index]

            if resize_masks:
                # Resize masks directly in-place by modifying the existing data
                for hoi_data in decoded_data_list:
                    # Resize each mask directly
                    for i, mask in enumerate(hoi_data.masks):
                        # Convert numpy array to PIL Image and resize
                        mask_image = Image.fromarray(mask.astype(np.uint8))
                        resized_mask = mask_image.resize(
                            (self.rgb_width, self.rgb_height), resample=Image.NEAREST
                        )
                        hoi_data.masks[i] = np.array(resized_mask)

            return decoded_data_list
        return None

    def get_hoi_total_number(self) -> int:
        """Get total number of interaction timestamps."""
        return len(self.hoi_data_list)
