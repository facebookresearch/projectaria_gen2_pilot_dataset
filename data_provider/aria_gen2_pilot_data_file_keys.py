# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""
AriaGen2PilotDataset File Keys and Constants

This module defines core file names, paths, and metadata keys used in the Aria Gen2 Pilot Dataset
for VRS data access. TODO: add MPS and algorithm-specific file keys.
"""

# ==== Core File Name Constants ====
# VRS files
VRS_FILE_NAME = "video.vrs"

VRS_HEALTH_CHECK_FILE = "vrs_health_check_results.json"

# mps
MPS_FOLDER = "mps"
MPS_SLAM_SUBFOLDER = "slam"
MPS_HAND_TRACKING_SUBFOLDER = "hand_tracking"
MPS_OPEN_LOOP_TRAJECTORY_FILE_NAME = "open_loop_trajectory.csv"
MPS_CLOSED_LOOP_TRAJECTORY_FILE_NAME = "closed_loop_trajectory.csv"
MPS_HAND_TRACKING_RESULTS_FILE_NAME = "hand_tracking_results.csv"

# Algorithm folders and files
HEART_RATE_FOLDER = "heart_rate"
HEART_RATE_RESULTS_FILE = "heart_rate_results.csv"
DIARIZATION_FOLDER = "diarization"
DIARIZATION_RESULTS_FILE = "diarization_results.csv"
EVL_FOLDER = "scene"
EVL_INSTANCE_FILE = "instances.json"
BBOX_3D_FILE = "3d_bounding_box.csv"
SCENE_OBJECTS_FILE = "scene_objects.csv"
BBOX_2D_FILE = "2d_bounding_box.csv"
HAND_OBJECT_INTERACTION_FOLDER = "hand_object_interaction"
HAND_OBJECT_INTERACTION_RESULTS_FILE = "hand_object_interaction_results.json"

# Foundation Stereo folder and files
STEREO_DEPTH_FOLDER = "depth"
STEREO_DEPTH_DEPTH_SUBFOLDER = "depth"
STEREO_DEPTH_RECTIFIED_IMAGES_SUBFOLDER = "rectified_images"
STEREO_DEPTH_PINHOLE_CAMERA_PARAMETERS_FILE = "pinhole_camera_parameters.json"

# Summary files
SUMMARY_FILE = "summary.json"

# ==== Time Domain Constants ====
DEVICE_TIME_DOMAIN = "DeviceTime"
TIME_CODE_DOMAIN = "TimeCode"

# ==== Default Values ====
INVALID_TIMESTAMP_NS = -1
DEFAULT_TIME_QUERY_OPTION = "closest"

# ==== Supported File Extensions ====
SUPPORTED_VRS_EXTENSIONS = [".vrs"]
SUPPORTED_METADATA_EXTENSIONS = [".json"]

# ==== Stream Labels (Common Aria Gen2 streams) ====
RGB_CAMERA_LABEL = "camera-rgb"
SLAM_LEFT_CAMERA_LABEL = "camera-slam-left"
SLAM_RIGHT_CAMERA_LABEL = "camera-slam-right"
ET_CAMERA_LABEL = "camera-et"
IMU_RIGHT_LABEL = "imu-right"
IMU_LEFT_LABEL = "imu-left"
GPS_LABEL = "gps"
WPS_LABEL = "wps"
AUDIO_LABEL = "mic"
BAROMETER_LABEL = "baro"
BLUETOOTH_LABEL = "bluetooth"
MAGNETOMETER_LABEL = "mag"
PPG_LABEL = "ppg"

# ==== Sensor Data Types ====
IMAGE_DATA_TYPE = "ImageData"
IMU_DATA_TYPE = "ImuData"
GPS_DATA_TYPE = "GpsData"
WPS_DATA_TYPE = "WpsData"
AUDIO_DATA_TYPE = "AudioData"
BAROMETER_DATA_TYPE = "BarometerData"
BLUETOOTH_DATA_TYPE = "BluetoothData"
MAGNETOMETER_DATA_TYPE = "MagnetometerData"
PPG_DATA_TYPE = "PpgData"
