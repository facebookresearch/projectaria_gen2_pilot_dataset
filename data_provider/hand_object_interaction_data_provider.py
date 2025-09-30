# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import json
import os
from typing import Dict, List, Optional, Tuple

from projectaria_tools.core.sensor_data import TimeQueryOptions

# Use self-contained RLE utilities instead of external pycocotools
from . import rle_utils

from .aria_gen2_pilot_dataset_data_types import (
    HandObjectInteractionData,
    HandObjectInteractionDataRaw,
)
from .utils import find_data_by_timestamp_ns


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
        # Aria-style timestamp-indexed data for efficient queries
        self.interaction_data_list: List[
            Tuple[int, List[HandObjectInteractionDataRaw]]
        ] = []

        self._load_data(data_path)

    def _load_data(self, data_path: str) -> None:
        """Load hand-object interaction JSON and build timestamp-indexed data."""
        if not os.path.exists(data_path):
            return

        try:
            with open(data_path, "r") as f:
                annotations = json.load(f)

            if not isinstance(annotations, list):
                raise ValueError(
                    "hand-object interaction results not an array of objects"
                )
            if len(annotations) == 0:
                return

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

            self.interaction_data_list = [
                (ts, interactions) for ts, interactions in sorted(temp_data.items())
            ]

        except (
            FileNotFoundError,
            json.JSONDecodeError,
            KeyError,
            ValueError,
            AssertionError,
        ) as e:
            raise RuntimeError(
                f"Failed to load hand-object interaction data from {data_path}: {e}"
            )

    # ================================================
    # Core Query APIs
    # ================================================

    def get_hoi_data_by_timestamp_ns(
        self,
        timestamp_ns: int,
        time_query_options: TimeQueryOptions = TimeQueryOptions.CLOSEST,
    ) -> Optional[List[HandObjectInteractionData]]:
        """Get all interactions at timestamp (hands + objects)."""
        undecoded_data = find_data_by_timestamp_ns(
            self.interaction_data_list, timestamp_ns, time_query_options
        )
        if undecoded_data is None:
            return None
        return rle_utils.convert_to_decoded_format(undecoded_data)

    def get_hoi_data_by_index(
        self, index: int
    ) -> Optional[List[HandObjectInteractionData]]:
        """Get interactions by index."""
        if 0 <= index < len(self.interaction_data_list):
            undecoded_data = self.interaction_data_list[index][1]
            return rle_utils.convert_to_decoded_format(undecoded_data)
        return None

    # ================================================
    # Metadata APIs
    # =========================================

    def get_hoi_timestamps_ns(self) -> List[int]:
        """Get all interaction timestamps."""
        return [timestamp for timestamp, _ in self.interaction_data_list]

    def get_hoi_total_number(self) -> int:
        """Get total number of interaction timestamps."""
        return len(self.interaction_data_list)

    def has_hoi_data(self) -> bool:
        """Check if interaction data exists."""
        return len(self.interaction_data_list) > 0
