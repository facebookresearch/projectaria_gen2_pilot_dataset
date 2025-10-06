# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import logging
from typing import Dict, Optional

from projectaria_tools.core.sensor_data import TimeQueryOptions

from projectaria_tools.projects.adt import (
    AriaDigitalTwinDataPaths,
    AriaDigitalTwinDataProvider,
    InstanceInfo,
)

from .aria_gen2_pilot_data_paths import EgocentricVoxelLiftingDataPaths
from .aria_gen2_pilot_dataset_data_types import BoundingBox2D, BoundingBox3D


class EgocentricVoxelLiftingDataProvider:
    def __init__(
        self,
        data_paths: EgocentricVoxelLiftingDataPaths,
        camera_label_to_stream_ids: Dict[str, str],
    ) -> None:
        self.data_paths = data_paths
        self.camera_label_to_stream_ids = camera_label_to_stream_ids

        # Set up logger
        self.logger = logging.getLogger(self.__class__.__name__)

        adt_data_paths = AriaDigitalTwinDataPaths()
        adt_data_paths.instances_filepath = self.data_paths.instances_file_path
        adt_data_paths.object_boundingbox_3d_filepath = (
            self.data_paths.bbox_3d_file_path
        )
        adt_data_paths.boundingboxes_2d_filepath = self.data_paths.bbox_2d_file_path
        adt_data_paths.object_trajectories_filepath = (
            self.data_paths.scene_objects_file_path
        )

        self.adt_gt_provider = AriaDigitalTwinDataProvider(adt_data_paths)

    def get_evl_3d_bounding_boxes_by_timestamp_ns(
        self,
        timestamp_ns: int,
        time_query_options: TimeQueryOptions = TimeQueryOptions.CLOSEST,
    ) -> Optional[Dict[int, BoundingBox3D]]:  # [Dict[instance_id, BoundingBox3D]]
        bbox3d_with_dt = (
            self.adt_gt_provider.get_object_3d_boundingboxes_by_timestamp_ns(
                timestamp_ns,
                time_query_options,
            )
        )
        if not bbox3d_with_dt.is_valid():
            self.logger.warning(
                "Cannot obtain valid 3d bbox data at %d, or the nearest valid bb3d is too far away.",
                timestamp_ns,
            )
            return None
        bbox3d_instance_ids = bbox3d_with_dt.data().keys()
        device_timestamp_ns = bbox3d_with_dt.dt_ns() + timestamp_ns

        bbox3d_id_to_3d_bboxes = {}

        for bbox3d_instance_id in bbox3d_instance_ids:
            bbox3d_id_to_3d_bboxes[bbox3d_instance_id] = BoundingBox3D(
                device_timestamp_ns, bbox3d_with_dt.data()[bbox3d_instance_id]
            )
        return bbox3d_id_to_3d_bboxes

    def get_evl_2d_bounding_boxes_by_timestamp_ns(
        self,
        timestamp_ns: int,
        camera_label: str,
        time_query_options: TimeQueryOptions = TimeQueryOptions.CLOSEST,
    ) -> Optional[Dict[int, BoundingBox2D]]:
        bbox2d_with_dt = (
            self.adt_gt_provider.get_object_2d_boundingboxes_by_timestamp_ns(
                timestamp_ns,
                self.camera_label_to_stream_ids[camera_label],
                time_query_options,
            )
        )
        if not bbox2d_with_dt.is_valid():
            self.logger.warning(
                "Cannot obtain valid 2d bbox data at %d, or the nearest valid bb2d is too far away.",
                timestamp_ns,
            )
            return None
        device_timestamp_ns = bbox2d_with_dt.dt_ns() + timestamp_ns

        bbox2d_id_to_2d_bboxes = {}

        for bbox2d_instance_id in bbox2d_with_dt.data().keys():
            bbox2d_id_to_2d_bboxes[bbox2d_instance_id] = BoundingBox2D(
                device_timestamp_ns, bbox2d_with_dt.data()[bbox2d_instance_id]
            )
        return bbox2d_id_to_2d_bboxes

    def get_evl_instance_info_by_id(self, instance_id: int) -> Optional[InstanceInfo]:
        """Get instance information (category, name) for given instance ID."""
        return (
            self.adt_gt_provider.get_instance_info_by_id(instance_id)
            if self.adt_gt_provider.has_instance_id(instance_id)
            else None
        )
