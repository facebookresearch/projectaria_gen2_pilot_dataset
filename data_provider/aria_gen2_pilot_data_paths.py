# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import logging
import os
from dataclasses import dataclass


@dataclass
class AriaGen2PilotDataPaths:
    """
    A class that includes the core file paths for Aria Gen2 Pilot Dataset sequences.

    This focuses on VRS data access and excludes MPS and algorithm-specific paths.
    """

    def __init__(self, sequence_path: str) -> None:
        """
        Initialize AriaGen2PilotDataPaths for a given sequence.

        Args:
            sequence_path: Root path to the sequence directory
        """
        self.sequence_path: str = sequence_path
        self.sequence_name: str = os.path.basename(sequence_path)

        # Set up logger
        self.logger = logging.getLogger(self.__class__.__name__)

        # ====== Core VRS Files ======
        self.vrs_file_path: str = ""
        self.vrs_health_check_path: str = ""
        self.mps_folder_path: str = ""

        # ====== Algorithm Generated Data Paths ======
        self.heart_rate_results_file_path: str = ""
        self.diarization_results_file_path: str = ""

        # ====== Egocentric Voxel Lifting Algorithm Data Paths ======
        self.evl_instances_file_path: str = ""
        self.evl_bbox_3d_file_path: str = ""
        self.evl_scene_objects_file_path: str = ""
        self.evl_bbox_2d_file_path: str = ""

        # ====== Hand Object Interaction Algorithm Data Paths ======
        self.hand_object_interaction_results_file_path: str = ""

        # ====== Foundation Stereo Algorithm Data Paths ======
        self.stereo_depth_data_folder_path: str = ""

    def to_string(self) -> str:
        """
        Generate a string representation of all data paths.

        Returns:
            String representation of the data paths
        """
        lines = [f"AriaGen2PilotDataPaths for sequence: {self.sequence_name}"]
        lines.append(f"  Sequence path: {self.sequence_path}")
        lines.append(f"  VRS file: {self.vrs_file_path}")
        lines.append(f"  VRS health check: {self.vrs_health_check_path}")
        lines.append(f"  MPS folder: {self.mps_folder_path}")

        return "\n".join(lines)

    def is_vrs_data_path_valid(self) -> bool:
        """Check if VRS file exists."""
        return self.vrs_file_path != "" and os.path.isfile(self.vrs_file_path)

    def is_mps_data_path_valid(self) -> bool:
        """Check if MPS folder exists."""
        return self.mps_folder_path != "" and os.path.isdir(self.mps_folder_path)

    def is_heart_rate_data_path_valid(self) -> bool:
        """Check if heart rate results file exists."""
        return self.heart_rate_results_file_path != "" and os.path.isfile(
            self.heart_rate_results_file_path
        )

    def is_diarization_data_path_valid(self) -> bool:
        """Check if diarization results file exists."""
        return self.diarization_results_file_path != "" and os.path.isfile(
            self.diarization_results_file_path
        )

    def is_egocentric_voxel_lifting_data_path_valid(self) -> bool:
        """Check if any egocentric voxel lifting data files exist."""
        return (
            self.evl_instances_file_path != ""
            and os.path.isfile(self.evl_instances_file_path)
            and self.evl_bbox_3d_file_path != ""
            and os.path.isfile(self.evl_bbox_3d_file_path)
            and self.evl_scene_objects_file_path != ""
            and os.path.isfile(self.evl_scene_objects_file_path)
            and self.evl_bbox_2d_file_path != ""
            and os.path.isfile(self.evl_bbox_2d_file_path)
        )

    def is_hand_object_interaction_data_path_valid(self) -> bool:
        """Check if hand-object interaction results file exists."""
        return self.hand_object_interaction_results_file_path != "" and os.path.isfile(
            self.hand_object_interaction_results_file_path
        )

    def is_stereo_depth_data_path_valid(self) -> bool:
        """Check if foundation stereo data folder exists."""
        return self.stereo_depth_data_folder_path != "" and os.path.isdir(
            self.stereo_depth_data_folder_path
        )

    def is_valid(self) -> bool:
        """Check if this data paths object has valid data.
        VRS data is required. MPS and algorithm-specific data are optional.
        """
        # VRS data is required, raise error if missing
        if not self.is_vrs_data_path_valid():
            raise RuntimeError("VRS data is not valid or missing.")

        return True


class EgocentricVoxelLiftingDataPaths:
    """Data paths for egocentric voxel lifting dataset."""

    def __init__(
        self,
        instances_file_path: str = "",
        bbox_3d_file_path: str = "",
        scene_objects_file_path: str = "",
        bbox_2d_file_path: str = "",
    ):
        self.instances_file_path = instances_file_path
        self.bbox_3d_file_path = bbox_3d_file_path
        self.scene_objects_file_path = scene_objects_file_path
        self.bbox_2d_file_path = bbox_2d_file_path
