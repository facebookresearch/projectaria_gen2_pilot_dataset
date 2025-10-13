# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import logging
import os
from typing import Dict, List, Optional

import numpy as np
from aria_gen2_pilot_dataset.data_provider.aria_gen2_pilot_data_file_keys import (
    MPS_CLOSED_LOOP_TRAJECTORY_FILE_NAME,
    MPS_HAND_TRACKING_RESULTS_FILE_NAME,
    MPS_HAND_TRACKING_SUBFOLDER,
    MPS_OPEN_LOOP_TRAJECTORY_FILE_NAME,
    MPS_SLAM_SUBFOLDER,
)
from projectaria_tools.core import mps
from projectaria_tools.core.calibration import DeviceCalibration

from projectaria_tools.core.data_provider import (
    create_vrs_data_provider,
    VrsDataProvider,
)
from projectaria_tools.core.mps import (
    ClosedLoopTrajectoryPose,
    EyeGaze,
    OpenLoopTrajectoryPose,
)
from projectaria_tools.core.mps.utils import (
    filter_points_from_confidence,
    filter_points_from_count,
)
from projectaria_tools.core.sensor_data import (
    AlsData,
    AudioData,
    AudioDataRecord,
    BarometerData,
    BluetoothBeaconData,
    FrontendOutput,
    GpsData,
    ImageData,
    ImageDataRecord,
    MotionData,
    PpgData,
    TemperatureData,
    TimeDomain,
    TimeQueryOptions,
    TimeSyncMode,
    WifiBeaconData,
)

from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.projects.adt import InstanceInfo

from .aria_gen2_pilot_data_paths import (
    AriaGen2PilotDataPaths,
    EgocentricVoxelLiftingDataPaths,
)
from .aria_gen2_pilot_data_paths_provider import AriaGen2PilotDataPathsProvider
from .aria_gen2_pilot_dataset_data_types import (
    BoundingBox2D,
    BoundingBox3D,
    CameraIntrinsicsAndPose,
    DiarizationData,
    HandObjectInteractionData,
    HeartRateData,
)
from .diarization_data_provider import DiarizationDataProvider
from .egocentric_voxel_lifting_data_provider import EgocentricVoxelLiftingDataProvider
from .hand_object_interaction_data_provider import HandObjectInteractionDataProvider
from .heart_rate_data_provider import HeartRateDataProvider
from .stereo_depth_data_provider import StereoDepthDataProvider


class AriaGen2PilotDataProvider:
    """Main data provider for Aria Gen2 Pilot Dataset sequences."""

    def __init__(self, sequence_folder_path: str) -> None:
        """Initialize the data provider with the given sequence folder path.

        Args:
            sequence_folder_path: Path to the sequence folder
        """
        self.RGB_CAMERA_LABEL = "camera-rgb"
        # Create data paths from the sequence folder path
        path_provider = AriaGen2PilotDataPathsProvider(sequence_folder_path)
        parsed_data_paths = path_provider.get_data_paths()
        if not parsed_data_paths:
            raise ValueError(
                f"Invalid data paths from sequence directory: {sequence_folder_path}"
            )
        self.data_paths_: AriaGen2PilotDataPaths = parsed_data_paths

        # Set up logger
        self.logger = logging.getLogger(self.__class__.__name__)

        self.vrs_data_provider_ = self.init_vrs_data_provider()
        self.mps_data_provider_ = self.init_mps_data_provider()
        self.stereo_depth_data_provider_ = self.init_stereo_depth_data_provider()
        self.hand_object_interaction_data_provider_ = (
            self.init_hand_object_interaction_data_provider()
        )
        self.egocentric_voxel_lifting_data_provider_ = (
            self.init_egocentric_voxel_lifting_data_provider()
        )
        self.heart_rate_data_provider_ = self.init_heart_rate_data_provider()
        self.diarization_data_provider_ = self.init_diarization_data_provider()

        # Initialize caches for MPS data
        self._cached_open_loop_trajectory: Optional[List[OpenLoopTrajectoryPose]] = None
        self._cached_closed_loop_trajectory: Optional[
            List[ClosedLoopTrajectoryPose]
        ] = None
        self._cached_hand_tracking_result_list: Optional[
            List[mps.hand_tracking.HandTrackingResult]
        ] = None

        # Report data provider availability
        data_types = [
            ("VRS", self.vrs_data_provider_ is not None),
            ("MPS", self.mps_data_provider_ is not None),
            ("Foundation Stereo", self.stereo_depth_data_provider_ is not None),
            ("Heart Rate", self.heart_rate_data_provider_ is not None),
            ("Diarization", self.diarization_data_provider_ is not None),
            (
                "Egocentric voxel lifting",
                self.egocentric_voxel_lifting_data_provider_ is not None,
            ),
            (
                "Hand Object Interaction",
                self.hand_object_interaction_data_provider_ is not None,
            ),
        ]

        available = sum(status for _, status in data_types)
        self.logger.info(
            "Data provider availability: %d/%d types", available, len(data_types)
        )
        for name, status in data_types:
            status_str = "√" if status else "X"
            if status:
                self.logger.info("   %s %s", status_str, name)
            else:
                self.logger.warning("   %s %s", status_str, name)

    def init_vrs_data_provider(self) -> VrsDataProvider:
        """Initialize and return VRS data provider."""
        if not self.data_paths_.is_vrs_data_path_valid():
            raise ValueError("VRS path not found in data paths: ", self.data_paths_)

        vrs_data_provider = create_vrs_data_provider(self.data_paths_.vrs_file_path)
        if vrs_data_provider is None:
            raise RuntimeError("VRS data provider not initialized correctly.")
        return vrs_data_provider

    def init_mps_data_provider(self) -> Optional[mps.MpsDataProvider]:
        """Initialize and return MPS data provider."""
        if not self.data_paths_.is_mps_data_path_valid():
            self.logger.warning(
                "MPS data not found in sequence folder: %s",
                self.data_paths_.sequence_path,
            )
            return None

        mps_paths_provider = mps.MpsDataPathsProvider(self.data_paths_.mps_folder_path)
        mps_data_paths = mps_paths_provider.get_data_paths()
        mps_data_provider = mps.MpsDataProvider(mps_data_paths)
        # check whether mps_data_provider has valid data needed for AriaGen2PilotDataProvider
        if (
            mps_data_provider.has_closed_loop_poses()
            or mps_data_provider.has_open_loop_poses()
            or mps_data_provider.has_semidense_point_cloud()
            or mps_data_provider.has_hand_tracking_results()
        ):
            return mps_data_provider
        else:
            self.logger.warning(
                "MPS data provider not initialized correctly. Missing required data.",
            )
            return None

    def init_stereo_depth_data_provider(
        self,
    ) -> Optional[StereoDepthDataProvider]:
        """Initialize and return Foundation Stereo data provider."""
        if not self.data_paths_.is_stereo_depth_data_path_valid():
            self.logger.warning(
                "Foundation Stereo data not found in sequence folder: %s",
                self.data_paths_.sequence_path,
            )
            return None

        return StereoDepthDataProvider(self.data_paths_.stereo_depth_data_folder_path)

    def init_hand_object_interaction_data_provider(
        self,
    ) -> Optional[HandObjectInteractionDataProvider]:
        """Initialize and return Hand Object Interaction data provider."""
        if not self.data_paths_.is_hand_object_interaction_data_path_valid():
            self.logger.warning(
                "Hand Object Interaction data not found in sequence folder: %s",
                self.data_paths_.sequence_path,
            )
            return None
        rgb_stream_id = self.vrs_data_provider_.get_stream_id_from_label(
            self.RGB_CAMERA_LABEL
        )
        rgb_width = self.vrs_data_provider_.get_image_configuration(
            rgb_stream_id
        ).image_width
        rgb_height = self.vrs_data_provider_.get_image_configuration(
            rgb_stream_id
        ).image_height

        return HandObjectInteractionDataProvider(
            self.data_paths_.hand_object_interaction_results_file_path,
            rgb_width=rgb_width,
            rgb_height=rgb_height,
        )

    def init_egocentric_voxel_lifting_data_provider(
        self,
    ) -> Optional[EgocentricVoxelLiftingDataProvider]:
        """Initialize and return Egocentric Voxel Lifting data provider."""
        if not self.data_paths_.is_egocentric_voxel_lifting_data_path_valid():
            self.logger.warning(
                "Egocentric Voxel Lifting data not found in sequence folder: %s",
                self.data_paths_.sequence_path,
            )
            return None

        evl_paths = EgocentricVoxelLiftingDataPaths(
            instances_file_path=self.data_paths_.evl_instances_file_path,
            bbox_3d_file_path=self.data_paths_.evl_bbox_3d_file_path,
            scene_objects_file_path=self.data_paths_.evl_scene_objects_file_path,
            bbox_2d_file_path=self.data_paths_.evl_bbox_2d_file_path,
        )

        # Create camera label to stream ID mapping
        camera_label_to_stream_ids = {}
        all_streams = self.vrs_data_provider_.get_all_streams()
        for stream_id in all_streams:
            label = self.vrs_data_provider_.get_label_from_stream_id(stream_id)
            if "camera" in label.lower() or "slam" in label.lower():
                camera_label_to_stream_ids[label] = stream_id

        return EgocentricVoxelLiftingDataProvider(evl_paths, camera_label_to_stream_ids)

    def init_heart_rate_data_provider(self) -> Optional[HeartRateDataProvider]:
        """Initialize and return Heart Rate data provider."""
        if not self.data_paths_.is_heart_rate_data_path_valid():
            self.logger.warning(
                "Heart Rate data not found in sequence folder: %s",
                self.data_paths_.sequence_path,
            )
            return None

        return HeartRateDataProvider(self.data_paths_.heart_rate_results_file_path)

    def init_diarization_data_provider(self) -> Optional[DiarizationDataProvider]:
        """Initialize and return Diarization data provider."""
        if not self.data_paths_.is_diarization_data_path_valid():
            self.logger.warning(
                "Diarization data not found in sequence folder: %s",
                self.data_paths_.sequence_path,
            )
            return None

        return DiarizationDataProvider(self.data_paths_.diarization_results_file_path)

    # ============================================================================
    #                                private utils
    # ============================================================================

    def _convert_time_domain_to_device_time_ns(
        self, time_domain: TimeDomain, timestamp_ns: int
    ) -> int:
        """Convert time domain to device time in nanoseconds."""
        if time_domain == TimeDomain.DEVICE_TIME:
            return timestamp_ns
        elif time_domain == TimeDomain.TIME_CODE:
            return self.vrs_data_provider_.convert_from_timecode_to_device_time_ns(
                timestamp_ns
            )
        elif time_domain == TimeDomain.TIC_SYNC:
            return self.vrs_data_provider_.convert_from_synctime_to_device_time_ns(
                timestamp_ns, TimeSyncMode.TIC_SYNC
            )
        elif time_domain == TimeDomain.SUBGHZ:
            return self.vrs_data_provider_.convert_from_synctime_to_device_time_ns(
                timestamp_ns, TimeSyncMode.SUBGHZ
            )
        else:
            raise ValueError(f"Unsupported time domain: {time_domain}")

    # ============================================================================
    #                                  VRS API
    # ============================================================================
    @property
    def vrs_data_provider(self) -> VrsDataProvider:
        """Get VRS data provider."""
        if self.vrs_data_provider_ is None:
            raise RuntimeError("VRS data provider not initialized correctly.")
        return self.vrs_data_provider_

    # On Device Data Access
    def get_vrs_all_streams(self) -> List[StreamId]:
        """Get all available streams from the vrs file."""
        return self.vrs_data_provider_.get_all_streams()

    def get_vrs_label_from_stream_id(self, stream_id: StreamId) -> str:
        """Get label from stream_id."""
        return self.vrs_data_provider_.get_label_from_stream_id(stream_id)

    def get_vrs_stream_id_from_label(self, label: str) -> Optional[StreamId]:
        """Get stream_id from label."""
        return self.vrs_data_provider_.get_stream_id_from_label(label)

    def get_vrs_device_calibration(self) -> Optional[DeviceCalibration]:
        """Get calibration of the device."""
        return self.vrs_data_provider_.get_device_calibration()

    def get_vrs_num_data(self, stream_id: StreamId) -> int:
        """Return number of collected sensor data of a stream."""
        return self.vrs_data_provider_.get_num_data(stream_id)

    def get_vrs_timestamps_ns(
        self, stream_id: StreamId, time_domain: TimeDomain
    ) -> List[int]:
        """Get all timestamps in nanoseconds for a stream."""
        return self.vrs_data_provider_.get_timestamps_ns(stream_id, time_domain)

    # Data getters by index
    def get_vrs_image_data_by_index(
        self, stream_id: StreamId, index: int
    ) -> tuple[ImageData, ImageDataRecord]:
        """Get image data by index."""
        return self.vrs_data_provider_.get_image_data_by_index(stream_id, index)

    def get_vrs_imu_data_by_index(
        self, stream_id: StreamId, index: int
    ) -> Optional[MotionData]:
        """Get IMU data by index."""
        return self.vrs_data_provider_.get_imu_data_by_index(stream_id, index)

    def get_vrs_gps_data_by_index(
        self, stream_id: StreamId, index: int
    ) -> Optional[GpsData]:
        """Get GPS data by index."""
        return self.vrs_data_provider_.get_gps_data_by_index(stream_id, index)

    def get_vrs_wps_data_by_index(
        self, stream_id: StreamId, index: int
    ) -> Optional[WifiBeaconData]:
        """Get WPS data by index."""
        return self.vrs_data_provider_.get_wps_data_by_index(stream_id, index)

    def get_vrs_audio_data_by_index(
        self, stream_id: StreamId, index: int
    ) -> tuple[AudioData, AudioDataRecord]:
        """Get audio data by index."""
        return self.vrs_data_provider_.get_audio_data_by_index(stream_id, index)

    def get_vrs_barometer_data_by_index(
        self, stream_id: StreamId, index: int
    ) -> Optional[BarometerData]:
        """Get barometer data by index."""
        return self.vrs_data_provider_.get_barometer_data_by_index(stream_id, index)

    def get_vrs_bluetooth_data_by_index(
        self, stream_id: StreamId, index: int
    ) -> Optional[BluetoothBeaconData]:
        """Get Bluetooth data by index."""
        return self.vrs_data_provider_.get_bluetooth_data_by_index(stream_id, index)

    def get_vrs_magnetometer_data_by_index(
        self, stream_id: StreamId, index: int
    ) -> Optional[MotionData]:
        """Get magnetometer data by index."""
        return self.vrs_data_provider_.get_magnetometer_data_by_index(stream_id, index)

    def get_vrs_ppg_data_by_index(
        self, stream_id: StreamId, index: int
    ) -> Optional[PpgData]:
        """Get PPG data by index."""
        return self.vrs_data_provider_.get_ppg_data_by_index(stream_id, index)

    def get_vrs_vio_data_by_index(
        self, stream_id: StreamId, index: int
    ) -> Optional[FrontendOutput]:
        """Get VIO data by index."""
        return self.vrs_data_provider_.get_vio_data_by_index(stream_id, index)

    def get_vrs_vio_high_freq_data_by_index(
        self, stream_id: StreamId, index: int
    ) -> Optional[OpenLoopTrajectoryPose]:
        """Get VIO high frequency data by index."""
        return self.vrs_data_provider_.get_vio_high_freq_data_by_index(stream_id, index)

    def get_vrs_eye_gaze_data_by_index(
        self, stream_id: StreamId, index: int
    ) -> Optional[EyeGaze]:
        """Get eye gaze data by index."""
        return self.vrs_data_provider_.get_eye_gaze_data_by_index(stream_id, index)

    def get_vrs_hand_pose_data_by_index(
        self, stream_id: StreamId, index: int
    ) -> Optional[mps.hand_tracking.HandTrackingResult]:
        """Get hand pose data by index."""
        return self.vrs_data_provider_.get_hand_pose_data_by_index(stream_id, index)

    def get_vrs_als_data_by_index(
        self, stream_id: StreamId, index: int
    ) -> Optional[AlsData]:
        """Get ALS data by index."""
        return self.vrs_data_provider_.get_als_data_by_index(stream_id, index)

    def get_vrs_temperature_data_by_index(
        self, stream_id: StreamId, index: int
    ) -> Optional[TemperatureData]:
        """Get temperature data by index."""
        return self.vrs_data_provider_.get_temperature_data_by_index(stream_id, index)

    # Data getters by timestamp
    def get_vrs_image_data_by_time_ns(
        self,
        stream_id: StreamId,
        time_ns: int,
        time_domain: TimeDomain,
        time_query_options: TimeQueryOptions = TimeQueryOptions.BEFORE,
    ) -> tuple[ImageData, ImageDataRecord]:
        """Get image data by timestamp."""
        return self.vrs_data_provider_.get_image_data_by_time_ns(
            stream_id, time_ns, time_domain, time_query_options
        )

    def get_vrs_imu_data_by_time_ns(
        self,
        stream_id: StreamId,
        time_ns: int,
        time_domain: TimeDomain,
        time_query_options: TimeQueryOptions = TimeQueryOptions.BEFORE,
    ) -> Optional[MotionData]:
        """Get IMU data by timestamp."""
        return self.vrs_data_provider_.get_imu_data_by_time_ns(
            stream_id, time_ns, time_domain, time_query_options
        )

    def get_vrs_gps_data_by_time_ns(
        self,
        stream_id: StreamId,
        time_ns: int,
        time_domain: TimeDomain,
        time_query_options: TimeQueryOptions = TimeQueryOptions.BEFORE,
    ) -> Optional[GpsData]:
        """Get GPS data by timestamp."""
        return self.vrs_data_provider_.get_gps_data_by_time_ns(
            stream_id, time_ns, time_domain, time_query_options
        )

    def get_vrs_wps_data_by_time_ns(
        self,
        stream_id: StreamId,
        time_ns: int,
        time_domain: TimeDomain,
        time_query_options: TimeQueryOptions = TimeQueryOptions.BEFORE,
    ) -> Optional[WifiBeaconData]:
        """Get WPS data by timestamp."""
        return self.vrs_data_provider_.get_wps_data_by_time_ns(
            stream_id, time_ns, time_domain, time_query_options
        )

    def get_vrs_audio_data_by_time_ns(
        self,
        stream_id: StreamId,
        time_ns: int,
        time_domain: TimeDomain,
        time_query_options: TimeQueryOptions = TimeQueryOptions.BEFORE,
    ) -> tuple[AudioData, AudioDataRecord]:
        """Get audio data by timestamp."""
        return self.vrs_data_provider_.get_audio_data_by_time_ns(
            stream_id, time_ns, time_domain, time_query_options
        )

    def get_vrs_barometer_data_by_time_ns(
        self,
        stream_id: StreamId,
        time_ns: int,
        time_domain: TimeDomain,
        time_query_options: TimeQueryOptions = TimeQueryOptions.BEFORE,
    ) -> Optional[BarometerData]:
        """Get barometer data by timestamp."""
        return self.vrs_data_provider_.get_barometer_data_by_time_ns(
            stream_id, time_ns, time_domain, time_query_options
        )

    def get_vrs_bluetooth_data_by_time_ns(
        self,
        stream_id: StreamId,
        time_ns: int,
        time_domain: TimeDomain,
        time_query_options: TimeQueryOptions = TimeQueryOptions.BEFORE,
    ) -> Optional[BluetoothBeaconData]:
        """Get Bluetooth data by timestamp."""
        return self.vrs_data_provider_.get_bluetooth_data_by_time_ns(
            stream_id, time_ns, time_domain, time_query_options
        )

    def get_vrs_magnetometer_data_by_time_ns(
        self,
        stream_id: StreamId,
        time_ns: int,
        time_domain: TimeDomain,
        time_query_options: TimeQueryOptions = TimeQueryOptions.BEFORE,
    ) -> Optional[MotionData]:
        """Get magnetometer data by timestamp."""
        return self.vrs_data_provider_.get_magnetometer_data_by_time_ns(
            stream_id, time_ns, time_domain, time_query_options
        )

    def get_vrs_ppg_data_by_time_ns(
        self,
        stream_id: StreamId,
        time_ns: int,
        time_domain: TimeDomain,
        time_query_options: TimeQueryOptions = TimeQueryOptions.BEFORE,
    ) -> Optional[PpgData]:
        """Get PPG data by timestamp."""
        return self.vrs_data_provider_.get_ppg_data_by_time_ns(
            stream_id, time_ns, time_domain, time_query_options
        )

    def get_vrs_vio_data_by_time_ns(
        self,
        stream_id: StreamId,
        time_ns: int,
        time_domain: TimeDomain,
        time_query_options: TimeQueryOptions = TimeQueryOptions.BEFORE,
    ) -> Optional[FrontendOutput]:
        """Get VIO data by timestamp."""
        return self.vrs_data_provider_.get_vio_data_by_time_ns(
            stream_id, time_ns, time_domain, time_query_options
        )

    def get_vrs_vio_high_freq_data_by_time_ns(
        self,
        stream_id: StreamId,
        time_ns: int,
        time_domain: TimeDomain,
        time_query_options: TimeQueryOptions = TimeQueryOptions.BEFORE,
    ) -> Optional[OpenLoopTrajectoryPose]:
        """Get VIO high frequency data by timestamp."""
        return self.vrs_data_provider_.get_vio_high_freq_data_by_time_ns(
            stream_id, time_ns, time_domain, time_query_options
        )

    def get_vrs_eye_gaze_data_by_time_ns(
        self,
        stream_id: StreamId,
        time_ns: int,
        time_domain: TimeDomain,
        time_query_options: TimeQueryOptions = TimeQueryOptions.BEFORE,
    ) -> Optional[EyeGaze]:
        """Get eye gaze data by timestamp."""
        return self.vrs_data_provider_.get_eye_gaze_data_by_time_ns(
            stream_id, time_ns, time_domain, time_query_options
        )

    def get_vrs_hand_pose_data_by_time_ns(
        self,
        stream_id: StreamId,
        time_ns: int,
        time_domain: TimeDomain,
        time_query_options: TimeQueryOptions = TimeQueryOptions.BEFORE,
    ) -> Optional[mps.hand_tracking.HandTrackingResult]:
        """Get hand pose data by timestamp."""
        return self.vrs_data_provider_.get_hand_pose_data_by_time_ns(
            stream_id, time_ns, time_domain, time_query_options
        )

    def get_vrs_interpolated_hand_pose_data(
        self,
        handtracking_stream_id: StreamId,
        device_time_ns: int,
        time_domain: TimeDomain,
    ) -> Optional[mps.hand_tracking.HandTrackingResult]:
        """Get hand pose data by timestamp."""
        vrs_data_provider = self.vrs_data_provider_
        if vrs_data_provider is None:
            raise RuntimeError("VRS data provider was not initialized.")
        if handtracking_stream_id not in vrs_data_provider.get_all_streams():
            raise ValueError
        return vrs_data_provider.get_interpolated_hand_pose_data(
            handtracking_stream_id, device_time_ns, time_domain
        )

    def get_vrs_als_data_by_time_ns(
        self,
        stream_id: StreamId,
        time_ns: int,
        time_domain: TimeDomain,
        time_query_options: TimeQueryOptions = TimeQueryOptions.BEFORE,
    ) -> Optional[AlsData]:
        """Get ALS data by timestamp."""
        return self.vrs_data_provider_.get_als_data_by_time_ns(
            stream_id, time_ns, time_domain, time_query_options
        )

    def get_vrs_temperature_data_by_time_ns(
        self,
        stream_id: StreamId,
        time_ns: int,
        time_domain: TimeDomain,
        time_query_options: TimeQueryOptions = TimeQueryOptions.BEFORE,
    ) -> Optional[TemperatureData]:
        """Get temperature data by timestamp."""
        return self.vrs_data_provider_.get_temperature_data_by_time_ns(
            stream_id, time_ns, time_domain, time_query_options
        )

    # ============================================================================
    #                                  MPS API
    # ============================================================================
    @property
    def mps_data_provider(self) -> mps.MpsDataProvider:
        """Get VRS data provider."""
        if self.mps_data_provider_ is None:
            raise RuntimeError("MPS data provider not initialized correctly.")
        return self.mps_data_provider_

    def has_mps_data(self) -> bool:
        """Return whether the MPS data provider exists."""
        return self.mps_data_provider_ is not None

    def get_mps_open_loop_trajectory(self) -> List[OpenLoopTrajectoryPose]:
        if self.mps_data_provider_ is None:
            raise RuntimeError("MPS data provider was not initialized.")

        # Return cached data if available
        if self._cached_open_loop_trajectory is not None:
            return self._cached_open_loop_trajectory

        # Load and cache the data
        self._cached_open_loop_trajectory = mps.read_open_loop_trajectory(
            os.path.join(
                self.data_paths_.mps_folder_path,
                MPS_SLAM_SUBFOLDER,
                MPS_OPEN_LOOP_TRAJECTORY_FILE_NAME,
            )
        )
        return self._cached_open_loop_trajectory

    def get_mps_closed_loop_trajectory(self) -> List[ClosedLoopTrajectoryPose]:
        if self.mps_data_provider_ is None:
            raise RuntimeError("MPS data provider was not initialized.")

        # Return cached data if available
        if self._cached_closed_loop_trajectory is not None:
            return self._cached_closed_loop_trajectory

        # Load and cache the data
        self._cached_closed_loop_trajectory = mps.read_closed_loop_trajectory(
            os.path.join(
                self.data_paths_.mps_folder_path,
                MPS_SLAM_SUBFOLDER,
                MPS_CLOSED_LOOP_TRAJECTORY_FILE_NAME,
            )
        )
        return self._cached_closed_loop_trajectory

    def get_mps_open_loop_pose(
        self,
        timestamp_ns: int,
        time_domain: TimeDomain,
        time_query_options: TimeQueryOptions = TimeQueryOptions.CLOSEST,
    ) -> Optional[OpenLoopTrajectoryPose]:
        """Query MPS for OpenLoopTrajectoryPose at a specific timestamp."""
        if self.mps_data_provider_ is None:
            raise RuntimeError("MPS data provider was not initialized.")
        device_timestamp_ns = self._convert_time_domain_to_device_time_ns(
            time_domain, timestamp_ns
        )
        return self.mps_data_provider_.get_open_loop_pose(
            device_timestamp_ns, time_query_options
        )

    def get_mps_closed_loop_pose(
        self,
        timestamp_ns: int,
        time_domain: TimeDomain,
        time_query_options: TimeQueryOptions = TimeQueryOptions.CLOSEST,
    ) -> Optional[ClosedLoopTrajectoryPose]:
        """Query MPS for ClosedLoopTrajectoryPose at a specific timestamp."""
        if self.mps_data_provider_ is None:
            raise RuntimeError("MPS data provider was not initialized.")
        device_timestamp_ns = self._convert_time_domain_to_device_time_ns(
            time_domain, timestamp_ns
        )
        return self.mps_data_provider_.get_closed_loop_pose(
            device_timestamp_ns, time_query_options
        )

    def get_mps_interpolated_closed_loop_pose(
        self,
        timestamp_ns: int,
        time_domain: TimeDomain,
    ) -> Optional[ClosedLoopTrajectoryPose]:
        """Query MPS for interpolated ClosedLoopTrajectoryPose at a specific timestamp."""
        if self.mps_data_provider_ is None:
            raise RuntimeError("MPS data provider was not initialized.")
        device_timestamp_ns = self._convert_time_domain_to_device_time_ns(
            time_domain, timestamp_ns
        )
        return self.mps_data_provider_.get_interpolated_closed_loop_pose(
            device_timestamp_ns
        )

    def get_mps_semidense_point_cloud(self) -> List[mps.GlobalPointPosition]:
        """Get the MPS semidense point cloud."""
        if self.mps_data_provider_ is None:
            raise RuntimeError("MPS data provider was not initialized.")
        return self.mps_data_provider_.get_semidense_point_cloud()

    def get_mps_semidense_point_cloud_filtered(
        self, filter_confidence: bool = True, max_point_count: Optional[int] = None
    ) -> List[mps.GlobalPointPosition]:
        """Get the MPS semidense point cloud with filtering applied.

        Args:
            filter_confidence: If True, filter out low confidence points
            max_point_count: Maximum number of points to return (for downsampling)

        Returns:
            Filtered list of GlobalPointPosition objects
        """
        if self.mps_data_provider_ is None:
            raise RuntimeError("MPS data provider was not initialized.")

        # Get the raw point cloud data
        points_data = self.mps_data_provider_.get_semidense_point_cloud()

        if not points_data:
            return []

        # Apply confidence filtering if requested
        if filter_confidence:
            points_data = filter_points_from_confidence(points_data)

        # Apply count-based downsampling if requested
        if max_point_count is not None and len(points_data) > max_point_count:
            points_data = filter_points_from_count(points_data, max_point_count)

        return points_data

    def get_mps_hand_tracking_result_list(
        self,
    ) -> List[mps.hand_tracking.HandTrackingResult]:
        """Get the MPS hand tracking results."""
        if self.mps_data_provider_ is None:
            raise RuntimeError("MPS data provider was not initialized.")

        # Return cached data if available
        if self._cached_hand_tracking_result_list is not None:
            return self._cached_hand_tracking_result_list

        # Load and cache the data
        self._cached_hand_tracking_result_list = (
            mps.hand_tracking.read_hand_tracking_results(
                os.path.join(
                    self.data_paths_.mps_folder_path,
                    MPS_HAND_TRACKING_SUBFOLDER,
                    MPS_HAND_TRACKING_RESULTS_FILE_NAME,
                )
            )
        )
        return self._cached_hand_tracking_result_list

    def get_mps_hand_tracking_result(
        self,
        timestamp_ns: int,
        time_domain: TimeDomain,
        time_query_options: TimeQueryOptions = TimeQueryOptions.CLOSEST,
    ) -> Optional[mps.hand_tracking.HandTrackingResult]:
        """Get the MPS hand tracking result."""
        if self.mps_data_provider_ is None:
            raise RuntimeError("MPS data provider was not initialized.")
        device_timestamp_ns = self._convert_time_domain_to_device_time_ns(
            time_domain, timestamp_ns
        )
        return self.mps_data_provider_.get_hand_tracking_result(
            device_timestamp_ns, time_query_options
        )

    def get_mps_interpolated_hand_tracking_result(
        self,
        timestamp_ns: int,
        time_domain: TimeDomain,
    ) -> Optional[mps.hand_tracking.HandTrackingResult]:
        """Get the MPS hand tracking result."""
        if self.mps_data_provider_ is None:
            raise RuntimeError("MPS data provider was not initialized.")
        device_timestamp_ns = self._convert_time_domain_to_device_time_ns(
            time_domain, timestamp_ns
        )
        return self.mps_data_provider_.get_interpolated_hand_tracking_result(
            device_timestamp_ns
        )

    # =======================================================
    #                    Heart Rate API
    # =======================================================
    def has_heart_rate_data(self) -> bool:
        """Return whether the heart rate algorithm data provider exists."""
        return self.heart_rate_data_provider_ is not None

    def get_heart_rate_by_index(self, index: int) -> Optional[HeartRateData]:
        """Get heart rate data by index."""
        if self.heart_rate_data_provider_ is None:
            raise RuntimeError("Heart Rate data provider was not initialized.")
        return self.heart_rate_data_provider_.get_heart_rate_by_index(index)

    def get_heart_rate_by_timestamp_ns(
        self,
        timestamp_ns: int,
        time_domain: TimeDomain,
        time_query_options: TimeQueryOptions = TimeQueryOptions.CLOSEST,
    ) -> Optional[HeartRateData]:
        """Get heart rate data at specified timestamp."""
        if self.heart_rate_data_provider_ is None:
            raise RuntimeError("Heart Rate data provider was not initialized.")
        device_timestamp_ns = self._convert_time_domain_to_device_time_ns(
            time_domain, timestamp_ns
        )
        return self.heart_rate_data_provider_.get_heart_rate_by_timestamp_ns(
            device_timestamp_ns, time_query_options
        )

    def get_heart_rate_total_number(self) -> int:
        """Get total number of heart rate entries."""
        if self.heart_rate_data_provider_ is None:
            raise RuntimeError("Heart Rate data provider was not initialized.")
        return self.heart_rate_data_provider_.get_heart_rate_total_number()

    # =======================================================
    #                    Diarization API
    # =======================================================
    def has_diarization_data(self) -> bool:
        """Return whether the diarization algorithm data provider exists."""
        return self.diarization_data_provider_ is not None

    def get_diarization_data_by_index(self, index: int) -> Optional[DiarizationData]:
        """Get diarization data by index."""
        if self.diarization_data_provider_ is None:
            raise RuntimeError("Diarization data provider was not initialized.")
        return self.diarization_data_provider_.get_diarization_data_by_index(index)

    def get_diarization_data_by_timestamp_ns(
        self,
        timestamp_ns: int,
        time_domain: TimeDomain,
    ) -> List[DiarizationData]:
        """Get diarization data containing timestamp."""
        if self.diarization_data_provider_ is None:
            raise RuntimeError("Diarization data provider was not initialized.")
        device_timestamp_ns = self._convert_time_domain_to_device_time_ns(
            time_domain, timestamp_ns
        )
        return self.diarization_data_provider_.get_diarization_data_by_timestamp_ns(
            device_timestamp_ns
        )

    def get_diarization_data_by_start_and_end_timestamps(
        self,
        start_timestamp_ns: int,
        end_timestamp_ns: int,
        time_domain: TimeDomain,
    ) -> List[DiarizationData]:
        """Get diarization data overlapping with time period."""
        if self.diarization_data_provider_ is None:
            raise RuntimeError("Diarization data provider was not initialized.")
        device_start_ns = self._convert_time_domain_to_device_time_ns(
            time_domain, start_timestamp_ns
        )
        device_end_ns = self._convert_time_domain_to_device_time_ns(
            time_domain, end_timestamp_ns
        )
        return self.diarization_data_provider_.get_diarization_data_by_start_and_end_timestamps(
            device_start_ns, device_end_ns
        )

    def get_diarization_data_total_number(self) -> int:
        """Get total number of diarization entries."""
        if self.diarization_data_provider_ is None:
            raise RuntimeError("Diarization data provider was not initialized.")
        return self.diarization_data_provider_.get_diarization_data_total_number()

    # =======================================================
    #              Egocentric Voxel Lifting API
    # =======================================================
    def has_egocentric_voxel_lifting_data(self) -> bool:
        """Return whether the egocentric voxel lifting algorithm data provider exists."""
        return self.egocentric_voxel_lifting_data_provider_ is not None

    def get_evl_3d_bounding_boxes_by_timestamp_ns(
        self,
        timestamp_ns: int,
        time_domain: TimeDomain,
        time_query_options: TimeQueryOptions = TimeQueryOptions.CLOSEST,
    ) -> Optional[Dict[int, BoundingBox3D]]:
        """Get 3D bounding boxes from EVL data provider at specified timestamp."""
        if self.egocentric_voxel_lifting_data_provider_ is None:
            raise RuntimeError(
                "Egocentric Voxel Lifting data provider was not initialized."
            )
        device_timestamp_ns = self._convert_time_domain_to_device_time_ns(
            time_domain, timestamp_ns
        )
        return self.egocentric_voxel_lifting_data_provider_.get_evl_3d_bounding_boxes_by_timestamp_ns(
            device_timestamp_ns,
            time_query_options,
        )

    def get_evl_2d_bounding_boxes_by_timestamp_ns(
        self,
        timestamp_ns: int,
        time_domain: TimeDomain,
        camera_label: str,
    ) -> Optional[Dict[int, BoundingBox2D]]:
        """Get 2D bounding boxes from EVL data provider at specified timestamp for a specific camera."""
        if self.egocentric_voxel_lifting_data_provider_ is None:
            raise RuntimeError(
                "Egocentric Voxel Lifting data provider was not initialized."
            )
        device_timestamp_ns = self._convert_time_domain_to_device_time_ns(
            time_domain, timestamp_ns
        )
        return self.egocentric_voxel_lifting_data_provider_.get_evl_2d_bounding_boxes_by_timestamp_ns(
            device_timestamp_ns, camera_label
        )

    def get_evl_instance_info_by_id(self, instance_id: int) -> Optional[InstanceInfo]:
        """Get instance information (category, name) for given EVL instance ID."""
        if self.egocentric_voxel_lifting_data_provider_ is None:
            raise RuntimeError(
                "Egocentric Voxel Lifting data provider was not initialized."
            )
        return self.egocentric_voxel_lifting_data_provider_.get_evl_instance_info_by_id(
            instance_id
        )

    # =======================================================
    #                Hand Object Interaction API
    # =======================================================
    def has_hand_object_interaction_data(self) -> bool:
        """Return whether the hand object interaction algorithm data provider exists."""
        return self.hand_object_interaction_data_provider_ is not None

    def get_hoi_data_by_timestamp_ns(
        self,
        timestamp_ns: int,
        time_domain: TimeDomain,
        time_query_options: TimeQueryOptions = TimeQueryOptions.CLOSEST,
    ) -> Optional[List[HandObjectInteractionData]]:
        """Get hand-object interaction data at specified timestamp."""
        if self.hand_object_interaction_data_provider_ is None:
            raise RuntimeError(
                "Hand Object Interaction data provider was not initialized."
            )
        device_timestamp_ns = self._convert_time_domain_to_device_time_ns(
            time_domain, timestamp_ns
        )
        return self.hand_object_interaction_data_provider_.get_hoi_data_by_timestamp_ns(
            device_timestamp_ns, time_query_options
        )

    def get_hoi_data_by_index(
        self, index: int
    ) -> Optional[List[HandObjectInteractionData]]:
        """Get hand-object interaction data by index."""
        if self.hand_object_interaction_data_provider_ is None:
            raise RuntimeError(
                "Hand Object Interaction data provider was not initialized."
            )
        return self.hand_object_interaction_data_provider_.get_hoi_data_by_index(index)

    def get_hoi_total_number(self) -> int:
        """Get total number of hand-object interaction timestamps."""
        if self.hand_object_interaction_data_provider_ is None:
            raise RuntimeError(
                "Hand Object Interaction data provider was not initialized."
            )
        return self.hand_object_interaction_data_provider_.get_hoi_total_number()

    def has_hoi_data(self) -> bool:
        """Check if hand-object interaction data exists."""
        return self.hand_object_interaction_data_provider_ is not None

    # =======================================================
    #                Foundation Stereo API
    # =======================================================
    def has_stereo_depth_data(self) -> bool:
        """Return whether the foundation stereo algorithm data provider exists."""
        return self.stereo_depth_data_provider_ is not None

    def get_stereo_depth_depth_map_by_index(self, index: int) -> Optional[np.ndarray]:
        """Get foundation stereo depth map by index."""
        if self.stereo_depth_data_provider_ is None:
            raise RuntimeError("Foundation Stereo data provider was not initialized.")
        return self.stereo_depth_data_provider_.get_stereo_depth_depth_map_by_index(
            index
        )

    def get_stereo_depth_depth_map_by_timestamp_ns(
        self,
        timestamp_ns: int,
        time_domain: TimeDomain,
        time_query_option: TimeQueryOptions = TimeQueryOptions.CLOSEST,
    ) -> Optional[np.ndarray]:
        """Get foundation stereo depth map at specified timestamp."""
        if self.stereo_depth_data_provider_ is None:
            raise RuntimeError("Foundation Stereo data provider was not initialized.")
        device_timestamp_ns = self._convert_time_domain_to_device_time_ns(
            time_domain, timestamp_ns
        )
        return (
            self.stereo_depth_data_provider_.get_stereo_depth_depth_map_by_timestamp_ns(
                device_timestamp_ns, time_query_option
            )
        )

    def get_stereo_depth_rectified_slam_front_left_by_index(
        self, index: int
    ) -> Optional[np.ndarray]:
        """Get foundation stereo rectified front-left SLAM image by index."""
        if self.stereo_depth_data_provider_ is None:
            raise RuntimeError("Foundation Stereo data provider was not initialized.")
        return self.stereo_depth_data_provider_.get_stereo_depth_rectified_slam_front_left_by_index(
            index
        )

    def get_stereo_depth_rectified_slam_front_left_by_timestamp_ns(
        self,
        timestamp_ns: int,
        time_domain: TimeDomain,
        time_query_option: TimeQueryOptions = TimeQueryOptions.CLOSEST,
    ) -> Optional[np.ndarray]:
        """Get foundation stereo rectified front-left SLAM image at specified timestamp."""
        if self.stereo_depth_data_provider_ is None:
            raise RuntimeError("Foundation Stereo data provider was not initialized.")
        device_timestamp_ns = self._convert_time_domain_to_device_time_ns(
            time_domain, timestamp_ns
        )
        return self.stereo_depth_data_provider_.get_stereo_depth_rectified_slam_front_left_by_timestamp_ns(
            device_timestamp_ns, time_query_option
        )

    def get_stereo_depth_camera_intrinsics_and_pose_by_index(
        self, index: int
    ) -> Optional[CameraIntrinsicsAndPose]:
        """Get foundation stereo depth camera info by index."""
        if self.stereo_depth_data_provider_ is None:
            raise RuntimeError("Foundation Stereo data provider was not initialized.")
        return self.stereo_depth_data_provider_.get_stereo_depth_camera_intrinsics_and_pose_by_index(
            index
        )

    def get_stereo_depth_camera_intrinsics_and_pose_by_timestamp_ns(
        self,
        timestamp_ns: int,
        time_domain: TimeDomain,
        time_query_option: TimeQueryOptions = TimeQueryOptions.CLOSEST,
    ) -> Optional[CameraIntrinsicsAndPose]:
        """Get foundation stereo depth camera info at specified timestamp."""
        if self.stereo_depth_data_provider_ is None:
            raise RuntimeError("Foundation Stereo data provider was not initialized.")
        device_timestamp_ns = self._convert_time_domain_to_device_time_ns(
            time_domain, timestamp_ns
        )
        return self.stereo_depth_data_provider_.get_stereo_depth_camera_intrinsics_and_pose_by_timestamp_ns(
            device_timestamp_ns, time_query_option
        )

    def get_stereo_depth_data_total_number(self) -> int:
        """Get total number of foundation stereo entries."""
        if self.stereo_depth_data_provider_ is None:
            raise RuntimeError("Foundation Stereo data provider was not initialized.")
        return self.stereo_depth_data_provider_.get_depth_data_total_number()
