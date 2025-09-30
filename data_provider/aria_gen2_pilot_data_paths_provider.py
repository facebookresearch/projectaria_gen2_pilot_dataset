# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.


import os
from typing import Optional

from aria_gen2_pilot_dataset.data_provider.aria_gen2_pilot_data_file_keys import (
    BBOX_2D_FILE,
    BBOX_3D_FILE,
    DIARIZATION_FOLDER,
    DIARIZATION_RESULTS_FILE,
    EVL_FOLDER,
    EVL_INSTANCE_FILE,
    HAND_OBJECT_INTERACTION_FOLDER,
    HAND_OBJECT_INTERACTION_RESULTS_FILE,
    HEART_RATE_FOLDER,
    HEART_RATE_RESULTS_FILE,
    MPS_FOLDER,
    SCENE_OBJECTS_FILE,
    STEREO_DEPTH_FOLDER,
    VRS_FILE_NAME,
    VRS_HEALTH_CHECK_FILE,
)

from aria_gen2_pilot_dataset.data_provider.aria_gen2_pilot_data_paths import (
    AriaGen2PilotDataPaths,
)


class AriaGen2PilotDataPathsProvider:
    """Loads core data file paths from a Aria Gen2 Pilot Dataset sequence directory."""

    def __init__(self, sequence_path: str) -> None:
        """Initialize the data paths provider for a given sequence."""
        if not os.path.isdir(sequence_path):
            raise ValueError(f"Invalid sequence directory: {sequence_path}")

        self.sequence_path = sequence_path

    def get_data_paths(self) -> Optional[AriaGen2PilotDataPaths]:
        """Retrieve the DataPaths for this sequence."""

        data_paths = AriaGen2PilotDataPaths(self.sequence_path)

        # Check for VRS file
        vrs_file_path = os.path.join(self.sequence_path, VRS_FILE_NAME)
        if os.path.isfile(vrs_file_path):
            data_paths.vrs_file_path = vrs_file_path

        # Check for MPS folder
        mps_folder_path = os.path.join(self.sequence_path, MPS_FOLDER)
        if os.path.isdir(mps_folder_path):
            data_paths.mps_folder_path = mps_folder_path

        # Check for VRS health check file
        health_check_path = os.path.join(self.sequence_path, VRS_HEALTH_CHECK_FILE)
        if os.path.isfile(health_check_path):
            data_paths.vrs_health_check_path = health_check_path

        # Set heart rate results file path
        heart_rate_file = os.path.join(
            self.sequence_path, HEART_RATE_FOLDER, HEART_RATE_RESULTS_FILE
        )
        if os.path.isfile(heart_rate_file):
            data_paths.heart_rate_results_file_path = heart_rate_file

        # Set diarization results file path
        diarization_file = os.path.join(
            self.sequence_path, DIARIZATION_FOLDER, DIARIZATION_RESULTS_FILE
        )
        if os.path.isfile(diarization_file):
            data_paths.diarization_results_file_path = diarization_file

        # Set egocentric voxel lifting data paths
        evl_folder = os.path.join(self.sequence_path, EVL_FOLDER)
        if os.path.isdir(evl_folder):
            instances_file = os.path.join(evl_folder, EVL_INSTANCE_FILE)
            if os.path.isfile(instances_file):
                data_paths.evl_instances_file_path = instances_file
            bbox_3d_file = os.path.join(evl_folder, BBOX_3D_FILE)
            if os.path.isfile(bbox_3d_file):
                data_paths.evl_bbox_3d_file_path = bbox_3d_file
            scene_objects_file = os.path.join(evl_folder, SCENE_OBJECTS_FILE)
            if os.path.isfile(scene_objects_file):
                data_paths.evl_scene_objects_file_path = scene_objects_file
            bbox_2d_file = os.path.join(evl_folder, BBOX_2D_FILE)
            if os.path.isfile(bbox_2d_file):
                data_paths.evl_bbox_2d_file_path = bbox_2d_file

        # Set hand-object interaction results file path
        hoi_file = os.path.join(
            self.sequence_path,
            HAND_OBJECT_INTERACTION_FOLDER,
            HAND_OBJECT_INTERACTION_RESULTS_FILE,
        )
        if os.path.isfile(hoi_file):
            data_paths.hand_object_interaction_results_file_path = hoi_file

        # Set foundation stereo data folder path
        stereo_depth_folder = os.path.join(self.sequence_path, STEREO_DEPTH_FOLDER)
        if os.path.isdir(stereo_depth_folder):
            data_paths.stereo_depth_data_folder_path = stereo_depth_folder

        return data_paths if data_paths.is_valid() else None
