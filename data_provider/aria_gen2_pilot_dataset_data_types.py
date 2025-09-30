# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""
AriaGen2PilotDataset Data Types

This module defines core data structures and types used in the Aria Gen2 Pilot Dataset.
"""

from dataclasses import dataclass
from typing import List

import numpy as np

from projectaria_tools.core.calibration import CameraProjection
from projectaria_tools.core.sophus import SE3

from projectaria_tools.projects.adt import BoundingBox2dData, BoundingBox3dData


@dataclass
class HeartRateData:
    """Heart rate data with timestamp. Unit: beats per minute."""

    timestamp_ns: int
    heart_rate_bpm: int  # beats per minute

    def __init__(self, timestamp_ns: int, heart_rate_bpm: int):
        self.timestamp_ns = timestamp_ns
        self.heart_rate_bpm = heart_rate_bpm

    def copy(self):
        return HeartRateData(self.timestamp_ns, self.heart_rate_bpm)


@dataclass
class DiarizationData:
    """
    A class representing diarization data from diarization_results.csv.

    CSV Format:
    - start_timestamp_ns (int): Timestamp in nanoseconds in device time domain
    - end_timestamp_ns (int): Timestamp in nanoseconds in device time domain
    - speaker (string): Unique identifier of the speaker
    - content (string): The ASR results in text
    """

    start_timestamp_ns: int
    end_timestamp_ns: int
    speaker: str
    content: str

    def __init__(
        self, start_timestamp_ns: int, end_timestamp_ns: int, speaker: str, content: str
    ):
        self.start_timestamp_ns = start_timestamp_ns
        self.end_timestamp_ns = end_timestamp_ns
        self.speaker = speaker
        self.content = content


@dataclass
class BoundingBox3D:
    start_timestamp_ns: int
    bbox3d: BoundingBox3dData

    def copy(self):
        return BoundingBox3D(self.start_timestamp_ns, self.bbox3d.copy())


@dataclass
class BoundingBox2D:
    start_timestamp_ns: int
    bbox2d: BoundingBox2dData

    def copy(self):
        return BoundingBox2D(self.start_timestamp_ns, self.bbox2d.copy())


@dataclass
class HandObjectInteractionDataRaw:
    """
    Storage format with raw RLE segmentation data for efficient loading.

    This format is used internally by the hand object interaction provider for fast loading and low memory usage.
    RLE segmentation is kept in compressed format until decoding is requested.
    """

    timestamp_ns: int  # PRIMARY KEY: timestamp in nanoseconds
    original_image_id: int  # Original COCO image_id for reference
    category_id: int  # 1=left_hand, 2=right_hand, 3=interacting_object
    bbox: List[float]  # [x, y, width, height] in pixels
    segmentation_size: List[int]  # [height, width] from RLE
    segmentation_counts: str  # RLE compressed string
    score: float  # Confidence score [0.0, 1.0]


@dataclass
class HandObjectInteractionData:
    """
    User-facing format with decoded segmentation masks for convenient access.
    """

    timestamp_ns: int  # PRIMARY KEY: timestamp in nanoseconds
    category_id: int  # 1=left_hand, 2=right_hand, 3=interacting_object
    masks: List[np.ndarray]  # List of decoded binary masks (height, width) uint8 arrays
    bboxes: List[
        List[float]
    ]  # List of bounding boxes [x, y, width, height] for each mask
    scores: List[float]  # List of confidence scores [0.0, 1.0] for each mask


@dataclass
class CameraIntrinsicsAndPose:
    """
    Camera model info for depth data, including timestamp, camera model name, intrinsic parameters(camera_intrinsic_params), and extrinsic parameters(transform_world_camera).
    """

    timestamp_ns: int  # PRIMARY KEY: timestamp in nanoseconds
    camera_projection: CameraProjection  # Camera projection model, stores the intrinsic parameters and camera model name
    transform_world_camera: SE3
