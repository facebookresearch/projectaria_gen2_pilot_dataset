# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from dataclasses import dataclass
from enum import Enum

from types import MappingProxyType
from typing import Optional

from . import plot_color


class PlotEntity(Enum):
    # MPS related
    HAND_TRACKING_LEFT_HAND_LANDMARKS = "HAND_TRACKING_LEFT_HAND_LANDMARKS"
    HAND_TRACKING_LEFT_HAND_SKELETON = "HAND_TRACKING_LEFT_HAND_SKELETON"
    HAND_TRACKING_RIGHT_HAND_LANDMARKS = "HAND_TRACKING_RIGHT_HAND_LANDMARKS"
    HAND_TRACKING_RIGHT_HAND_SKELETON = "HAND_TRACKING_RIGHT_HAND_SKELETON"
    TRAJECTORY = "TRAJECTORY"
    SEMI_DENSE_POINT_CLOUD = "SEMI_DENSE_POINT_CLOUD"

    # Algorithm related
    EVL_BBOX_3D = "EVL_BBOX_3D"
    EVL_BBOX_PROJECTED_3D = "EVL_BBOX_PROJECTED_3D"
    HOI_LEFT_HAND = "HOI_LEFT_HAND"
    HOI_RIGHT_HAND = "HOI_RIGHT_HAND"
    HOI_INTERACTING_OBJECT = "HOI_INTERACTING_OBJECT"
    DIARIZATION_TEXT = "DIARIZATION_TEXT"
    HEART_RATE = "HEART_RATE"
    STEREO_DEPTH = "STEREO_DEPTH"


@dataclass
class PlotStyle:
    color: Optional[list] = None  # RGB or RGBA
    plot_2d_size: Optional[int] = None
    plot_3d_size: Optional[int] = None
    label: Optional[str] = None


_REGISTRY = {
    PlotEntity.SEMI_DENSE_POINT_CLOUD: PlotStyle(
        label="semi_dense_point_cloud",
        color=plot_color.GRAY,
        plot_3d_size=0.006,
    ),
    PlotEntity.TRAJECTORY: PlotStyle(
        label="trajectory", color=[173, 216, 255], plot_3d_size=0.015
    ),
    PlotEntity.HAND_TRACKING_LEFT_HAND_LANDMARKS: PlotStyle(
        label="landmarks",
        color=[255, 64, 0],
        plot_2d_size=6,
        plot_3d_size=0.005,
    ),
    PlotEntity.HAND_TRACKING_LEFT_HAND_SKELETON: PlotStyle(
        label="skeleton",
        color=plot_color.GREEN,
        plot_2d_size=3,
        plot_3d_size=0.003,
    ),
    PlotEntity.HAND_TRACKING_RIGHT_HAND_LANDMARKS: PlotStyle(
        label="landmarks",
        color=plot_color.YELLOW,
        plot_2d_size=6,
        plot_3d_size=0.005,
    ),
    PlotEntity.HAND_TRACKING_RIGHT_HAND_SKELETON: PlotStyle(
        label="skeleton",
        color=plot_color.GREEN,
        plot_2d_size=3,
        plot_3d_size=0.003,
    ),
    PlotEntity.HOI_LEFT_HAND: PlotStyle(
        label="hoi_left_hand", color=[119, 172, 48, 128]
    ),
    PlotEntity.HOI_RIGHT_HAND: PlotStyle(
        label="hoi_right_hand", color=[217, 83, 255, 128]
    ),
    PlotEntity.HOI_INTERACTING_OBJECT: PlotStyle(
        label="hoi_interacting_object", color=[237, 177, 32, 128]
    ),
    PlotEntity.DIARIZATION_TEXT: PlotStyle(
        label="diarization",
        color=plot_color.WHITE,
        plot_2d_size=10,
    ),
    PlotEntity.EVL_BBOX_3D: PlotStyle(
        label="evl_3d_bboxes",
        color=plot_color.GREEN_ALPHA,
        plot_3d_size=0.005,
    ),
    PlotEntity.EVL_BBOX_PROJECTED_3D: PlotStyle(
        label="evl_3d_bboxes_projected",
        color=plot_color.GREEN,
        plot_2d_size=1.5,
    ),
    PlotEntity.STEREO_DEPTH: PlotStyle(
        label="stereo_depth_depth_camera",
        plot_3d_size=0.3,
    ),
}

REGISTRY = MappingProxyType(_REGISTRY)


def get_plot_style(entity: PlotEntity) -> PlotStyle:
    return REGISTRY[entity]
