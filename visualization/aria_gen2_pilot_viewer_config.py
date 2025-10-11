# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from dataclasses import dataclass


@dataclass
class AriaGen2PilotViewerConfig:
    """Configuration class for AriaGen2PilotDataVisualizer."""

    # number of point cloud points to visualize
    point_cloud_max_point_count: int = 30000
    rgb_jpeg_quality: int = 30
    depth_and_slam_jpeg_quality: int = 10

    # new_image_width = image_width // depth_image_downsample_factor, new_image_height = image_height // depth_image_downsample_factor
    depth_image_downsample_factor_3d: int = 4
