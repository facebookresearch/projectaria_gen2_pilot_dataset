# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from dataclasses import dataclass


@dataclass
class AriaGen2PilotViewerConfig:
    """Configuration class for AriaGen2PilotDataVisualizer."""

    # number of point cloud points to visualize
    point_cloud_max_point_count: int = 50000
    jpeg_quality: int = 50
