# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass


@dataclass
class AriaGen2PilotViewerConfig:
    """Configuration class for AriaGen2PilotDataVisualizer."""

    # === Memory Optimization Settings ===
    # Reduce image quality for memory savings
    rgb_jpeg_quality: int = 50
    depth_and_slam_jpeg_quality: int = 50

    # Downsample images before logging
    rgb_downsample_factor: int = 2
    slam_downsample_factor: int = 4
    depth_image_downsample_factor: int = 4

    # Point cloud memory optimization
    point_cloud_max_point_count: int = 30000
