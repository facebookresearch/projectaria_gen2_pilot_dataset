# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import json
import logging
import os
from typing import List, Optional

import numpy as np
from projectaria_tools.core.calibration import CameraModelType, CameraProjection
from projectaria_tools.core.sensor_data import TimeQueryOptions
from projectaria_tools.core.sophus import SE3

from .aria_gen2_pilot_data_file_keys import (
    STEREO_DEPTH_DEPTH_SUBFOLDER,
    STEREO_DEPTH_PINHOLE_CAMERA_PARAMETERS_FILE,
    STEREO_DEPTH_RECTIFIED_IMAGES_SUBFOLDER,
)
from .aria_gen2_pilot_dataset_data_types import CameraIntrinsicsAndPose
from .utils import (
    check_valid_file,
    find_timestamp_index_by_time_query_option,
    load_image,
)


class StereoDepthDataProvider:
    """Foundation stereo data provider for Aria Gen2 Pilot Dataset."""

    def __init__(self, data_path: str):
        """Initialize with path to depth data directory."""
        self.data_path = data_path
        self.camera_intrinsics_and_pose_list: List[CameraIntrinsicsAndPose] = []
        self.timestamps_ns: List[int] = []
        self.sorted_to_original_indices: List[
            int
        ] = []  # Maps sorted index to original file index

        # Set up logger
        self.logger = logging.getLogger(self.__class__.__name__)

        self._load_data()
        self.depth_subfolder_path_ = os.path.join(
            self.data_path, STEREO_DEPTH_DEPTH_SUBFOLDER
        )

    def _load_data(self) -> None:
        """Load camera info from JSON file and build timestamp cache."""
        if not os.path.exists(self.data_path):
            raise RuntimeError(f"Depth data directory does not exist: {self.data_path}")

        json_path = os.path.join(
            self.data_path, STEREO_DEPTH_PINHOLE_CAMERA_PARAMETERS_FILE
        )
        check_valid_file(json_path)

        # Validate that required subfolders exist and contain files
        self._validate_subfolder(STEREO_DEPTH_DEPTH_SUBFOLDER)
        self._validate_subfolder(STEREO_DEPTH_RECTIFIED_IMAGES_SUBFOLDER)

        try:
            with open(json_path, "r") as f:
                all_frame_camera_json = json.load(f)

            if not all_frame_camera_json:
                raise RuntimeError(f"Depth camera JSON file is empty: {json_path}")

            for per_frame_camera_json in all_frame_camera_json:
                # Extract timestamp
                timestamp_ns = per_frame_camera_json["frameTimestampNs"]

                # Extract camera extrinsics as SE3 transform
                transform_world_camera_json = per_frame_camera_json["T_world_camera"]
                quat_w = transform_world_camera_json["QuaternionXYZW"][3]
                quat_xyz = np.array(
                    transform_world_camera_json["QuaternionXYZW"][:3]
                ).reshape(3, 1)
                translation_array = np.array(
                    transform_world_camera_json["Translation"]
                ).reshape(3, 1)
                transform_world_camera = SE3.from_quat_and_translation(
                    quat_w, quat_xyz, translation_array
                )

                # Extract camera intrinsics parameters
                camera_intrinsic_params = np.array(
                    per_frame_camera_json["camera"]["Parameters"]
                )

                # Extract camera model name
                camera_model_name = per_frame_camera_json["camera"]["ModelName"].split(
                    ":"
                )[0]
                if camera_model_name != "Linear":
                    raise RuntimeError("Only Linear camera model is supported.")

                camera_intrinsics_and_pose = CameraIntrinsicsAndPose(
                    timestamp_ns=timestamp_ns,
                    camera_projection=CameraProjection(
                        CameraModelType.LINEAR, camera_intrinsic_params
                    ),
                    transform_world_camera=transform_world_camera,
                )

                self.camera_intrinsics_and_pose_list.append(camera_intrinsics_and_pose)
                self.timestamps_ns.append(timestamp_ns)

            # Ensure data is sorted by timestamp for efficient querying
            if self.timestamps_ns and not all(
                self.timestamps_ns[i] < self.timestamps_ns[i + 1]
                for i in range(len(self.timestamps_ns) - 1)
            ):
                # Create mapping with original indices before sorting
                # indexed_data: (original_index, (timestamp, camera_intrinsics_and_pose))
                indexed_data = list(
                    enumerate(
                        zip(self.timestamps_ns, self.camera_intrinsics_and_pose_list)
                    )
                )

                # Sort all data by timestamp
                sorted_indexed_data = sorted(indexed_data, key=lambda x: x[1][0])

                # Extract sorted data and build original indices mapping
                self.sorted_to_original_indices = [
                    item[0] for item in sorted_indexed_data
                ]
                sorted_timestamps_and_poses = [item[1] for item in sorted_indexed_data]

                self.timestamps_ns, self.camera_intrinsics_and_pose_list = zip(
                    *sorted_timestamps_and_poses
                )
                self.timestamps_ns = list(self.timestamps_ns)
                self.camera_intrinsics_and_pose_list = list(
                    self.camera_intrinsics_and_pose_list
                )
            else:
                # Data is already sorted, create identity mapping
                self.sorted_to_original_indices = list(range(len(self.timestamps_ns)))

        except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as e:
            raise RuntimeError(
                f"Failed to load depth camera data from {json_path}: {e}"
            )
        if not self.timestamps_ns:
            raise RuntimeError(
                "No depth camera data found, can not initialize FoundationStereoDataProvider."
            )

    def _validate_subfolder(self, subfolder_name: str) -> List[str]:
        """Validate that a subfolder exists and contains PNG files."""
        subfolder_path = os.path.join(self.data_path, subfolder_name)

        if not os.path.exists(subfolder_path):
            raise RuntimeError(
                f"Foundation stereo {subfolder_path} subfolder does not exist!"
            )
        if not os.path.isdir(subfolder_path):
            raise RuntimeError(
                f"Foundation stereo {subfolder_path} path is not a directory!:"
            )

        png_files = [f for f in os.listdir(subfolder_path) if f.endswith(".png")]
        if not png_files:
            raise RuntimeError(f"No PNG files found in {subfolder_path} subfolder!")

    def get_stereo_depth_depth_map_by_index(self, index: int) -> Optional[np.ndarray]:
        """Get depth map by index.

        Returns depth map as uint16 array where 1 unit = 1mm.
        In camera coordinates:
        X = (x - cx) * depth / fx
        Y = (y - cy) * depth / fy
        Z = depth
        """
        if 0 <= index < len(self.camera_intrinsics_and_pose_list):
            # Use original file index to ensure correct timestamp-image correspondence
            original_file_index = self.sorted_to_original_indices[index]
            return load_image(
                folder_path=self.depth_subfolder_path_,
                filename_pattern="depth_{:08d}.png",
                index=original_file_index,
                dtype=np.uint16,
            )
        else:
            self.logger.warning(
                f"Invalid index {index}! No depth map found at this index."
            )
            return None

    def get_stereo_depth_depth_map_by_timestamp_ns(
        self,
        timestamp_ns: int,
        time_query_option: TimeQueryOptions = TimeQueryOptions.CLOSEST,
    ) -> Optional[np.ndarray]:
        """Get depth map at specified timestamp."""
        index = find_timestamp_index_by_time_query_option(
            self.timestamps_ns, timestamp_ns, time_query_option
        )
        return self.get_stereo_depth_depth_map_by_index(index)

    def get_stereo_depth_rectified_slam_front_left_by_index(
        self, index: int
    ) -> Optional[np.ndarray]:
        """Get rectified front-left SLAM image by index."""
        if 0 <= index < len(self.camera_intrinsics_and_pose_list):
            # Use original file index to ensure correct timestamp-image correspondence
            original_file_index = self.sorted_to_original_indices[index]
            rectified_images_path = os.path.join(
                self.data_path, STEREO_DEPTH_RECTIFIED_IMAGES_SUBFOLDER
            )
            return load_image(
                folder_path=rectified_images_path,
                filename_pattern="image_{:08d}.png",
                index=original_file_index,
                dtype=np.uint8,  # Rectified images are typically uint8, not uint16
            )
        else:
            self.logger.warning(
                f"Invalid index {index}! No rectified front-left SLAM image found at this index."
            )
            return None

    def get_stereo_depth_rectified_slam_front_left_by_timestamp_ns(
        self,
        timestamp_ns: int,
        time_query_option: TimeQueryOptions = TimeQueryOptions.CLOSEST,
    ) -> Optional[np.ndarray]:
        """Get rectified front-left SLAM image at specified timestamp."""
        index = find_timestamp_index_by_time_query_option(
            self.timestamps_ns, timestamp_ns, time_query_option
        )
        return self.get_stereo_depth_rectified_slam_front_left_by_index(index)

    def get_stereo_depth_camera_intrinsics_and_pose_by_index(
        self, index: int
    ) -> Optional[CameraIntrinsicsAndPose]:
        """Get depth camera info by index."""
        if 0 <= index < len(self.camera_intrinsics_and_pose_list):
            return self.camera_intrinsics_and_pose_list[index]
        else:
            self.logger.warning(
                "Index %d is out of range (0 to %d). Return None.",
                index,
                len(self.camera_intrinsics_and_pose_list) - 1,
            )
            return None

    def get_stereo_depth_camera_intrinsics_and_pose_by_timestamp_ns(
        self,
        timestamp_ns: int,
        time_query_option: TimeQueryOptions = TimeQueryOptions.CLOSEST,
    ) -> Optional[CameraIntrinsicsAndPose]:
        """Get depth camera info at specified timestamp."""
        index = find_timestamp_index_by_time_query_option(
            self.timestamps_ns, timestamp_ns, time_query_option
        )
        return self.get_stereo_depth_camera_intrinsics_and_pose_by_index(index)

    def get_depth_data_total_number(self) -> int:
        """Get total number of depth entries."""
        return len(self.camera_intrinsics_and_pose_list)
