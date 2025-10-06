# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import json
import os
from typing import Dict, List, Optional

from projectaria_tools.core.sensor_data import TimeQueryOptions

# Use self-contained RLE utilities instead of external pycocotools
from . import rle_utils

from .aria_gen2_pilot_dataset_data_types import (
    HandObjectInteractionData,
    HandObjectInteractionDataRaw,
)
from .utils import find_timestamp_index_by_time_query_option


class HandObjectInteractionDataProvider:
    """Hand-object interaction data provider for Aria Gen2 Pilot Dataset."""

    def __init__(self, data_path: str):
        """
        Initialize with path to hand_object_interaction_results.json.

        The file contains COCO-format detection results:
        - List of annotations with segmentation, bbox, score, image_id, category_id
        - category_id: 1=left_hand, 2=right_hand, 3=interacting_object
        - timestamp_ns = image_id * 1e6
        """
        # Store data and timestamps in separate lists following the standard pattern
        self.hoi_data_list: List[List[HandObjectInteractionDataRaw]] = []
        self.timestamps_ns: List[int] = []

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

            for annotation in annotations:
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

            sorted_temp_data = sorted(temp_data.items())
            self.hoi_data_list = [interactions for _, interactions in sorted_temp_data]
            self.timestamps_ns = [ts for ts, _ in sorted_temp_data]

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
    ) -> Optional[List[HandObjectInteractionData]]:
        """Get all interactions at timestamp (hands + objects)."""
        index = find_timestamp_index_by_time_query_option(
            self.timestamps_ns, timestamp_ns, time_query_options
        )
        return self.get_hoi_data_by_index(index)

    def get_hoi_data_by_index(
        self, index: int
    ) -> Optional[List[HandObjectInteractionData]]:
        """Get interactions by index."""
        if 0 <= index < len(self.hoi_data_list):
            undecoded_data = self.hoi_data_list[index]
            return rle_utils.convert_to_decoded_format(undecoded_data)
        return None

    def get_hoi_total_number(self) -> int:
        """Get total number of interaction timestamps."""
        return len(self.hoi_data_list)
