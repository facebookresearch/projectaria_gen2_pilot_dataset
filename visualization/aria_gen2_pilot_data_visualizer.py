# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import logging
from functools import partial
from typing import Dict

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from aria_gen2_pilot_dataset import AriaGen2PilotDataProvider

from aria_gen2_pilot_dataset.data_provider.aria_gen2_pilot_dataset_data_types import (
    BoundingBox3D,
    CameraIntrinsicsAndPose,
    HandObjectInteractionData,
    HeartRateData,
)
from PIL import Image
from projectaria_tools.core import mps
from projectaria_tools.core.sensor_data import SensorDataType, TimeDomain
from projectaria_tools.utils.rerun_helpers import (
    AriaGlassesOutline,
    create_hand_skeleton_from_landmarks,
    ToTransform3D,
)
from rerun.blueprint.archetypes import LineGrid3D
from tqdm import tqdm

from . import plot_color

from .aria_gen2_pilot_viewer_config import AriaGen2PilotViewerConfig
from .plot_style import get_plot_style, PlotEntity, PlotStyle
from .plot_utils import extract_bbox_projection_data, project_3d_bbox_to_2d_camera


class AriaGen2PilotDataVisualizer:
    """
    Visualization library for Aria Gen2 Pilot Dataset.
    """

    def __init__(
        self,
        data_provider: AriaGen2PilotDataProvider,
        config: AriaGen2PilotViewerConfig = None,
    ) -> None:
        """
        Initialize visualizer following aria_data_plotter blueprint
        """
        self.config = config if config is not None else AriaGen2PilotViewerConfig()
        self.pd_provider = data_provider

        self.logger = logging.getLogger(self.__class__.__name__)

        self.RGB_CAMERA_LABEL = "camera-rgb"
        self.SLAM_CAMERA_LABELS = [
            "slam-front-left",
            "slam-front-right",
            "slam-side-left",
            "slam-side-right",
        ]

        self.device_calibration = (
            self.pd_provider.vrs_data_provider.get_device_calibration()
        )
        if self.device_calibration is None:
            raise RuntimeError(
                "device_calibration is None. Cannot initialize visualizer."
            )
        self.rgb_camera_calibration = self.device_calibration.get_camera_calib(
            self.RGB_CAMERA_LABEL
        )
        self.slam_left_front_camera_calibration = (
            self.device_calibration.get_camera_calib(self.SLAM_CAMERA_LABELS[0])
        )
        if not self.rgb_camera_calibration:
            raise RuntimeError(
                "rgb_camera_calibration is None. Cannot initialize visualizer."
            )
        self.last_heart_rate_data = None
        self.slam_to_rgb_plotting_ratio = self._set_slam_to_rgb_plotting_ratio()
        self.closed_loop_trajectory_pose_cache = []
        self.rgb_frame_interval_ns = (
            0  # set later, rgb_frame_interval_ns = int (1 / rgb_frame_rate * 1e9)
        )
        # Store original image dimensions for scaling calculations
        self.original_rgb_width, self.original_rgb_height = (
            self.rgb_camera_calibration.get_image_size()
        )
        self.original_slam_width, self.original_slam_height = (
            self.slam_left_front_camera_calibration.get_image_size()
        )

    def initialize_rerun_and_blueprint(self, rrd_output_path: str = ""):
        """
        Initialize rerun and set up blueprint after all data is loaded
        """
        if rrd_output_path:
            self.logger.info(
                f"Initializing Rerun and saving to {rrd_output_path}... The Rerun window will not be shown when saving to a file."
            )
            rr.init("AriaGen2PilotDataViewer", spawn=False)
            rr.save(rrd_output_path)
        else:
            self.logger.info("Initializing Rerun visualization window...")
            rr.init("AriaGen2PilotDataViewer", spawn=True)

        # === Top Row Views ===
        rgb_view = rrb.Spatial2DView(
            name=self.RGB_CAMERA_LABEL,
            origin=f"{self.RGB_CAMERA_LABEL}",
        )

        world_3d_view = rrb.Spatial3DView(
            name="World View",
            origin="world",
            background=plot_color.BLACK,
            line_grid=LineGrid3D(visible=False),
        )

        # === Bottom Row Views ===
        heart_rate_view = rrb.TimeSeriesView(
            name="Heart Rate",
            origin="heart_rate_bpm",
        )

        depth_image_view = rrb.Spatial2DView(
            name="Depth",
            origin="world/stereo_depth_depth_camera",
        )

        rectified_slam_front_left_view = rrb.Spatial2DView(
            name="Rectified SLAM Front Left",
            origin="rectified_slam_front_left",
        )

        slam_views = [
            rrb.Spatial2DView(
                name=label,
                origin=f"{label}",
            )
            for label in self.SLAM_CAMERA_LABELS
        ]

        # Assemble the Layout
        blueprint = rrb.Blueprint(
            rrb.Vertical(
                rrb.Horizontal(rgb_view, world_3d_view),
                rrb.Horizontal(
                    heart_rate_view,
                    depth_image_view,
                    rectified_slam_front_left_view,
                    *slam_views,
                ),
                row_shares=[3, 1],
            ),
            rrb.SelectionPanel(state="collapsed"),
            rrb.BlueprintPanel(state="collapsed"),
        )

        rr.send_blueprint(blueprint, make_active=True)

    def _set_slam_to_rgb_plotting_ratio(self) -> float:
        """Calculate and set the scale ratio for converting plot sizes from RGB camera space to SLAM camera space.

        This ratio is based on the height ratio between SLAM and RGB camera resolutions.
        Used to ensure plot elements (markers, gaze points, etc.) appear proportionally
        sized across different camera views with different resolutions.
        """
        if self.device_calibration is None:
            raise RuntimeError(
                "device_calibration is None. Can not set slam_to_rgb_plotting_ratio."
            )
        rgb_image_height = self.device_calibration.get_camera_calib(
            self.RGB_CAMERA_LABEL
        ).get_image_size()[1]
        slam_image_height = self.device_calibration.get_camera_calib(
            self.SLAM_CAMERA_LABELS[0]
        ).get_image_size()[1]
        if rgb_image_height == 0 or slam_image_height == 0:
            raise RuntimeError(
                "RGB or SLAM image height is 0. Cannot set slam_to_rgb_plotting_ratio."
            )

        return slam_image_height / rgb_image_height

    def plot_static_components(self):
        """Plot static components, including device extrinsics and semidense point cloud."""

        # Plot device extrinsics
        self.plot_device_extrinsics()

        # Plot semidense point cloud
        if self.pd_provider.has_mps_data():
            filtered_mps_semidense_point_cloud_data = (
                self.pd_provider.get_mps_semidense_point_cloud_filtered(
                    filter_confidence=True,
                    max_point_count=self.config.point_cloud_max_point_count,
                )
            )
            self.plot_mps_semidense_point_cloud(filtered_mps_semidense_point_cloud_data)

    def plot_sequence(self):
        """Plot sequence, including sensor data, mps, algorithm outputs, etc."""
        # config deliver options
        vrs_data_provider = self.pd_provider.vrs_data_provider
        deliver_option = vrs_data_provider.get_default_deliver_queued_options()
        deliver_option.deactivate_stream_all()

        # Activate RGB and SLAM camera streams
        for label in [self.RGB_CAMERA_LABEL] + self.SLAM_CAMERA_LABELS:
            stream_id = vrs_data_provider.get_stream_id_from_label(label)
            deliver_option.activate_stream(stream_id)

        # render progress bar
        rgb_stream_id = vrs_data_provider.get_stream_id_from_label(
            self.RGB_CAMERA_LABEL
        )
        rgb_frame_count = vrs_data_provider.get_num_data(rgb_stream_id)
        progress_bar = tqdm(total=rgb_frame_count, desc="Processing frames")

        # set rgb_frame_rate
        self.rgb_frame_interval_ns = int(
            1 / vrs_data_provider.get_nominal_rate_hz(rgb_stream_id) * 1e9
        )

        # Plot static components
        self.plot_static_components()

        for data in vrs_data_provider.deliver_queued_sensor_data(deliver_option):
            device_time_ns = data.get_time_ns(TimeDomain.DEVICE_TIME)
            rr.set_time_nanos("device_time", device_time_ns)

            # Extract frame from SensorData and log RGB image with overlays
            if data.sensor_data_type() == SensorDataType.IMAGE:
                frame = data.image_data_and_record()[0].to_numpy_array()
                stream_label = vrs_data_provider.get_label_from_stream_id(
                    data.stream_id()
                )

                # plot images
                self.plot_image(
                    frame=frame,
                    camera_label=stream_label,
                    jpeg_quality=self.config.rgb_jpeg_quality
                    if stream_label == self.RGB_CAMERA_LABEL
                    else self.config.depth_and_slam_jpeg_quality,
                )

                # Project and plot hand tracking result for each image frame
                if self.pd_provider.has_mps_data():
                    mps_hand_tracking_result = (
                        self.pd_provider.get_mps_interpolated_hand_tracking_result(
                            device_time_ns, TimeDomain.DEVICE_TIME
                        )
                    )
                    self.plot_mps_hand_tracking_result_2d(
                        mps_hand_tracking_result,
                        stream_label,
                    )

            else:
                raise RuntimeError(
                    f"Unsupported sensor data type in visualization: {data.sensor_data_type()}"
                )

            if stream_label == self.RGB_CAMERA_LABEL:
                progress_bar.update(1)
                # plot heart rate data
                if self.pd_provider.has_heart_rate_data():
                    heart_rate_data = self.pd_provider.get_heart_rate_by_timestamp_ns(
                        device_time_ns, TimeDomain.DEVICE_TIME
                    )
                    self.plot_heart_rate_bpm(heart_rate_data, device_time_ns)

                # plot hand object interaction data
                if self.pd_provider.has_hoi_data():
                    hoi_data_list = self.pd_provider.get_hoi_data_by_timestamp_ns(
                        device_time_ns, TimeDomain.DEVICE_TIME
                    )
                    self.plot_hand_object_interaction_data(
                        hoi_data_list, device_time_ns
                    )

                # Plot diarization text overlay
                if self.pd_provider.has_diarization_data():
                    diarization_data_list = (
                        self.pd_provider.get_diarization_data_by_timestamp_ns(
                            device_time_ns, TimeDomain.DEVICE_TIME
                        )
                    )
                    self.plot_diarization_text_overlay(diarization_data_list)

                # Plot EVL 3D bounding boxes
                if self.pd_provider.has_egocentric_voxel_lifting_data():
                    evl_3d_bboxes = (
                        self.pd_provider.get_evl_3d_bounding_boxes_by_timestamp_ns(
                            device_time_ns, TimeDomain.DEVICE_TIME
                        )
                    )
                    self.plot_evl_3d_bounding_boxes(evl_3d_bboxes)
                    if self.pd_provider.has_mps_data():
                        self.plot_evl_3d_bboxes_projected_to_rgb_camera(device_time_ns)

            if stream_label == self.SLAM_CAMERA_LABELS[0]:
                if self.pd_provider.has_mps_data():
                    trajectory = self.pd_provider.get_mps_interpolated_closed_loop_pose(
                        device_time_ns, TimeDomain.DEVICE_TIME
                    )
                    self.plot_closed_loop_pose(trajectory)
                    self.plot_mps_hand_tracking_result_3d(
                        mps_hand_tracking_result,
                    )

                # plot stereo depth data
                if self.pd_provider.has_stereo_depth_data():
                    camera_intrinsics_and_pose = self.pd_provider.get_stereo_depth_camera_intrinsics_and_pose_by_timestamp_ns(
                        device_time_ns, TimeDomain.DEVICE_TIME
                    )
                    rectified_slam_front_left_image = self.pd_provider.get_stereo_depth_rectified_slam_front_left_by_timestamp_ns(
                        device_time_ns, TimeDomain.DEVICE_TIME
                    )
                    depth_map = (
                        self.pd_provider.get_stereo_depth_depth_map_by_timestamp_ns(
                            device_time_ns, TimeDomain.DEVICE_TIME
                        )
                    )
                    self.plot_stereo_depth_data(
                        camera_intrinsics_and_pose,
                        depth_map,
                        rectified_slam_front_left_image,
                        device_time_ns,
                    )

    # === Sensor related plotting functions ===
    def plot_device_extrinsics(self) -> None:
        """Plot device sensor extrinsics"""
        # A helper to log components with timeless = True
        log_static = partial(rr.log, static=True)

        if not self.device_calibration:
            raise RuntimeError(
                "Device calibration is None. Cannot plot device extrinsics!"
            )

        # Aria glasses outline for device reference
        aria_glasses_outline = AriaGlassesOutline(
            self.device_calibration, use_cad_calib=False
        )
        log_static(
            "world/device/glasses_outline",
            rr.LineStrips3D([aria_glasses_outline]),
            static=True,
        )

    def _resize_image(self, image: np.ndarray, downsample_factor: int) -> np.ndarray:
        """Downsample image by given factor using PIL, handles both regular and uint16 images."""
        if downsample_factor <= 1:
            return image

        new_size = (
            image.shape[1] // downsample_factor,
            image.shape[0] // downsample_factor,
        )

        # Handle uint16 images (not directly supported by PIL)
        if image.dtype == np.uint16:
            image_f = Image.fromarray(image.astype(np.float32), mode="F")
            resized_image = image_f.resize(new_size, resample=Image.LANCZOS)
            return np.clip(np.array(resized_image), 0, 65535).astype(np.uint16)

        # Handle regular images (uint8, RGB, etc.)
        pil_image = Image.fromarray(image)
        return np.array(pil_image.resize(new_size, Image.LANCZOS))

    def _get_camera_scale_factor(self, camera_label: str) -> float:
        """Get the scaling factor for overlays on this camera due to downsampling."""
        if camera_label == self.RGB_CAMERA_LABEL:
            return 1.0 / self.config.rgb_downsample_factor
        elif camera_label in self.SLAM_CAMERA_LABELS:
            return 1.0 / self.config.slam_downsample_factor
        else:
            return 1.0  # No scaling for unknown cameras

    def plot_image(
        self,
        frame: np.array,
        camera_label: str,
        jpeg_quality: int,
    ) -> None:
        """Plot image from a camera with downsampling"""
        # Apply downsampling based on camera type
        if camera_label == self.RGB_CAMERA_LABEL:
            downsample_factor = self.config.rgb_downsample_factor
        elif (
            camera_label in self.SLAM_CAMERA_LABELS
            or camera_label == "rectified_slam_front_left"
        ):
            downsample_factor = self.config.slam_downsample_factor
        else:
            downsample_factor = 1

        # Downsample the image
        if downsample_factor > 1:
            frame = self._resize_image(frame, downsample_factor)

        rr.log(
            f"{camera_label}",
            rr.Clear.recursive(),
        )
        rr.log(
            f"{camera_label}",
            rr.Image(frame).compress(jpeg_quality),
        )

    # === MPS related plotting functions ===
    def plot_mps_semidense_point_cloud(
        self, point_cloud_data: list[mps.GlobalPointPosition]
    ) -> None:
        if point_cloud_data == []:
            return
        points_array = np.array(
            [
                point.position_world
                for point in point_cloud_data
                if hasattr(point, "position_world")
            ]
        )
        plot_style = get_plot_style(PlotEntity.SEMI_DENSE_POINT_CLOUD)
        rr.log(
            f"world/{plot_style.label}",
            rr.Points3D(
                positions=points_array,
                colors=[plot_style.color] * len(points_array),
                radii=plot_style.plot_3d_size,
            ),
            static=True,
        )

    def plot_closed_loop_pose(
        self,
        closed_loop_trajectory_pose: mps.ClosedLoopTrajectoryPose,
    ) -> None:
        """Plot MPS closed loop trajectory"""
        if not closed_loop_trajectory_pose:
            return
        # Get transform and add to trajectory cache
        T_world_device = closed_loop_trajectory_pose.transform_world_device
        self.closed_loop_trajectory_pose_cache.append(T_world_device.translation()[0])

        # Plot device pose
        rr.log(
            "world/device",
            ToTransform3D(T_world_device, axis_length=0.05),
        )

        # Plot accumulated trajectory
        if len(self.closed_loop_trajectory_pose_cache) > 1:
            plot_style = get_plot_style(PlotEntity.TRAJECTORY)
            rr.log(
                f"world/{plot_style.label}",
                rr.LineStrips3D(
                    [self.closed_loop_trajectory_pose_cache],
                    colors=[plot_style.color],
                    radii=plot_style.plot_3d_size,
                ),
            )

    def _plot_single_hand_3d(
        self,
        hand_joints_in_device: list[np.array],
        hand_label: str,
    ) -> None:
        """
        Plot single hand data in 3D and 2D camera views
        """
        landmarks_style, skeleton_style = self._get_hand_plot_style(hand_label)
        if hand_joints_in_device is None:
            return
        # Plot 3D hand markers and skeleton
        hand_skeleton_3d = create_hand_skeleton_from_landmarks(hand_joints_in_device)
        rr.log(
            f"world/device/hand-tracking/{hand_label}/{landmarks_style.label}",
            rr.Points3D(
                positions=hand_joints_in_device,
                colors=[landmarks_style.color],
                radii=landmarks_style.plot_3d_size,
            ),
        )
        rr.log(
            f"world/device/hand-tracking/{hand_label}/{skeleton_style.label}",
            rr.LineStrips3D(
                hand_skeleton_3d,
                colors=[skeleton_style.color],
                radii=skeleton_style.plot_3d_size,
            ),
        )

    def _get_hand_plot_style(self, hand_label: str) -> PlotStyle:
        if hand_label == "left":
            landmarks_plot_entity = PlotEntity.HAND_TRACKING_LEFT_HAND_LANDMARKS
            skeleton_plot_entity = PlotEntity.HAND_TRACKING_LEFT_HAND_SKELETON
        else:
            landmarks_plot_entity = PlotEntity.HAND_TRACKING_RIGHT_HAND_LANDMARKS
            skeleton_plot_entity = PlotEntity.HAND_TRACKING_RIGHT_HAND_SKELETON

        return get_plot_style(landmarks_plot_entity), get_plot_style(
            skeleton_plot_entity
        )

    def _plot_single_hand_2d(
        self, hand_joints_in_device: list[np.array], hand_label: str, camera_label: str
    ) -> None:
        """
        Plot single hand data in 2D camera views
        """
        if hand_joints_in_device is None:
            return

        landmarks_style, skeleton_style = self._get_hand_plot_style(hand_label)

        # Project joint locations to camera view
        camera_calib = self.device_calibration.get_camera_calib(camera_label)

        # project into camera frame, and also create line segments
        hand_joints_in_camera = []
        for pt_in_device in hand_joints_in_device:
            pt_in_camera = (
                camera_calib.get_transform_device_camera().inverse() @ pt_in_device
            )
            pixel = camera_calib.project(pt_in_camera)
            hand_joints_in_camera.append(pixel)

        # Create hand skeleton in 2D image space
        hand_skeleton = create_hand_skeleton_from_landmarks(hand_joints_in_camera)

        # Apply downsampling scaling to 2D coordinates
        downsample_scale = self._get_camera_scale_factor(camera_label)
        if downsample_scale != 1.0:
            hand_joints_in_camera = [
                [pt[0] * downsample_scale, pt[1] * downsample_scale]
                if pt is not None
                else None
                for pt in hand_joints_in_camera
            ]
            hand_skeleton = [
                [
                    [pt[0] * downsample_scale, pt[1] * downsample_scale]
                    for pt in strip
                    if pt is not None
                ]
                for strip in hand_skeleton
            ]

        # Remove "None" markers from hand joints in camera. This is intentionally done AFTER the hand skeleton creation
        hand_joints_in_camera = list(
            filter(lambda x: x is not None, hand_joints_in_camera)
        )

        # Apply both SLAM-to-RGB ratio AND downsampling scale factor
        scale_ratio = (
            downsample_scale
            if camera_label == self.RGB_CAMERA_LABEL
            else downsample_scale * self.slam_to_rgb_plotting_ratio
        )
        rr.log(
            f"{camera_label}/hand-tracking/{hand_label}/{landmarks_style.label}",
            rr.Points2D(
                positions=hand_joints_in_camera,
                colors=[landmarks_style.color],
                radii=landmarks_style.plot_2d_size * scale_ratio,
            ),
        )
        rr.log(
            f"{camera_label}/hand-tracking/{hand_label}/{skeleton_style.label}",
            rr.LineStrips2D(
                hand_skeleton,
                colors=[skeleton_style.color],
                radii=skeleton_style.plot_2d_size * scale_ratio,
            ),
        )

    def plot_mps_hand_tracking_result_2d(
        self,
        hand_pose_data: mps.hand_tracking.HandTrackingResult,
        camera_label: str,
    ) -> None:
        """
        Project and plot hand pose data to 2D camera image
        """
        rr.log(
            f"{camera_label}/hand-tracking",
            rr.Clear.recursive(),
        )
        if hand_pose_data is None:
            return

        if hand_pose_data.left_hand is not None:
            self._plot_single_hand_2d(
                hand_joints_in_device=hand_pose_data.left_hand.landmark_positions_device,
                hand_label="left",
                camera_label=camera_label,
            )
        if hand_pose_data.right_hand is not None:
            self._plot_single_hand_2d(
                hand_joints_in_device=hand_pose_data.right_hand.landmark_positions_device,
                hand_label="right",
                camera_label=camera_label,
            )

    def plot_mps_hand_tracking_result_3d(
        self,
        hand_pose_data: mps.hand_tracking.HandTrackingResult,
    ) -> None:
        """
        Plot hand pose data within 3D world view
        """
        rr.log(
            "world/device/hand-tracking",
            rr.Clear.recursive(),
        )
        if hand_pose_data is None:
            return

        if hand_pose_data.left_hand is not None:
            self._plot_single_hand_3d(
                hand_joints_in_device=hand_pose_data.left_hand.landmark_positions_device,
                hand_label="left",
            )
        if hand_pose_data.right_hand is not None:
            self._plot_single_hand_3d(
                hand_joints_in_device=hand_pose_data.right_hand.landmark_positions_device,
                hand_label="right",
            )

    # === Algorithm related plotting functions ===
    def plot_heart_rate_bpm(
        self, heart_rate_data: HeartRateData, query_timestamp_ns: int
    ):
        if (
            heart_rate_data is None
            or abs(heart_rate_data.timestamp_ns - query_timestamp_ns)
            > self.rgb_frame_interval_ns / 2
            or (
                self.last_heart_rate_data
                and heart_rate_data.timestamp_ns
                == self.last_heart_rate_data.timestamp_ns
            )
        ):
            return
        self.last_heart_rate_data = heart_rate_data
        rr.log(
            "heart_rate_bpm",
            rr.Scalar(heart_rate_data.heart_rate_bpm),
        )

    def plot_diarization_text_overlay(self, diarization_data_list: list):
        """Plot diarization text overlay"""
        if not diarization_data_list:
            return

        # Get plot style for diarization
        diarization_style = get_plot_style(PlotEntity.DIARIZATION_TEXT)

        # Clear previous diarization overlays
        rr.log(
            f"{self.RGB_CAMERA_LABEL}/{diarization_style.label}", rr.Clear.recursive()
        )

        # Get image dimensions for positioning (use original dimensions, then scale)
        width, height = self.rgb_camera_calibration.get_image_size()

        # Apply downsampling scaling
        downsample_scale = self._get_camera_scale_factor(self.RGB_CAMERA_LABEL)
        scaled_width = width * downsample_scale
        scaled_height = height * downsample_scale

        if diarization_data_list:
            for i, conv_data in enumerate(diarization_data_list):
                text_content = f"{conv_data.speaker}: {conv_data.content}"
                text_x = scaled_width // 2
                text_y = (
                    scaled_height
                    - scaled_height / 15
                    - (i * diarization_style.plot_2d_size * downsample_scale * 7)
                )

                rr.log(
                    f"{self.RGB_CAMERA_LABEL}/{diarization_style.label}/conversation_text_{i}",
                    rr.Points2D(
                        positions=[[text_x, text_y]],
                        labels=[text_content],
                        colors=[diarization_style.color],
                        radii=diarization_style.plot_2d_size * downsample_scale,
                    ),
                )

    def plot_evl_3d_bounding_boxes(self, evl_3d_bboxes: Dict[int, BoundingBox3D]):
        """Visualize EVL 3D bounding box data using plot style"""
        # Clear previous 3D bounding boxes
        plot_style = get_plot_style(PlotEntity.EVL_BBOX_3D)
        rr.log(f"world/{plot_style.label}", rr.Clear.recursive())

        if not evl_3d_bboxes:
            return

        # Get plot style for EVL 3D bounding boxes
        plot_style = get_plot_style(PlotEntity.EVL_BBOX_3D)

        bb3d_sizes = []
        bb3d_centers = []
        bb3d_quats_xyzw = []
        bb3d_labels = []

        for instance_id, boundingBox3d in evl_3d_bboxes.items():
            # Extract BoundingBox3dData from our BoundingBox3D wrapper
            bbox3d_data = boundingBox3d.bbox3d

            # Get AABB in object's local coordinates: [xmin, xmax, ymin, ymax, zmin, zmax]
            aabb = bbox3d_data.aabb

            # Calculate dimensions
            object_dimensions = np.array(
                [
                    aabb[1] - aabb[0],  # width (xmax - xmin)
                    aabb[3] - aabb[2],  # height (ymax - ymin)
                    aabb[5] - aabb[4],  # depth (zmax - zmin)
                ]
            )

            # Get world center and rotation from transform_scene_object
            T_scene_object = bbox3d_data.transform_scene_object
            quat_and_translation = np.squeeze(T_scene_object.to_quat_and_translation())
            quaternion_wxyz = quat_and_translation[0:4]  # [w, x, y, z]
            world_center = quat_and_translation[4:7]  # [x, y, z]

            # Convert quaternion to ReRun format [x, y, z, w]
            quat_xyzw = [
                quaternion_wxyz[1],
                quaternion_wxyz[2],
                quaternion_wxyz[3],
                quaternion_wxyz[0],
            ]

            # Get label
            label = f"instance_{instance_id}"
            instance_info = self.pd_provider.get_evl_instance_info_by_id(instance_id)
            if instance_info:
                if hasattr(instance_info, "category") and instance_info.category:
                    label = instance_info.category
                elif hasattr(instance_info, "name") and instance_info.name:
                    label = instance_info.name

            # Add to lists
            bb3d_centers.append(world_center)
            bb3d_sizes.append(object_dimensions)
            bb3d_quats_xyzw.append(quat_xyzw)
            bb3d_labels.append(label)

        # Visualize using ReRun Boxes3D with plot style
        if bb3d_sizes:
            # Split into batches of 20 (ReRun limitation, if batch size too large, the labels can not be displayed)
            MAX_BOXES_PER_BATCH = 20
            batch_id = 0

            while batch_id * MAX_BOXES_PER_BATCH < len(bb3d_sizes):
                start_idx = batch_id * MAX_BOXES_PER_BATCH
                end_idx = min(len(bb3d_sizes), start_idx + MAX_BOXES_PER_BATCH)
                rr.log(
                    f"world/{plot_style.label}/batch_{batch_id}",
                    rr.Boxes3D(
                        sizes=bb3d_sizes[start_idx:end_idx],
                        centers=bb3d_centers[start_idx:end_idx],
                        rotations=bb3d_quats_xyzw[start_idx:end_idx],
                        labels=bb3d_labels[start_idx:end_idx],
                        colors=[plot_style.color],
                        radii=plot_style.plot_3d_size,
                        show_labels=False,
                    ),
                )
                batch_id += 1

    def plot_evl_3d_bboxes_projected_to_rgb_camera(self, device_time_ns: int):
        """Project EVL 3D bounding boxes onto 2D RGB camera view with labels"""
        plot_style = get_plot_style(PlotEntity.EVL_BBOX_PROJECTED_3D)

        rr.log(
            f"{self.RGB_CAMERA_LABEL}/{plot_style.label}",
            rr.Clear.recursive(),
        )

        # retrieve data to plot
        evl_3d_bboxes = self.pd_provider.get_evl_3d_bounding_boxes_by_timestamp_ns(
            device_time_ns, TimeDomain.DEVICE_TIME
        )
        if not evl_3d_bboxes:
            return

        trajectory_pose = self.pd_provider.get_mps_closed_loop_pose(
            device_time_ns, TimeDomain.DEVICE_TIME
        )
        if (
            not trajectory_pose
            or abs(
                int(trajectory_pose.tracking_timestamp.total_seconds() * 1e9)
                - device_time_ns
            )
            > self.rgb_frame_interval_ns / 2
        ):
            return

        # Get transforms and image dimensions
        T_world_device = trajectory_pose.transform_world_device
        T_device_camera = self.rgb_camera_calibration.get_transform_device_camera()
        T_world_camera = T_world_device @ T_device_camera

        # Get original image dimensions then scale for downsampling
        image_width, image_height = self.rgb_camera_calibration.get_image_size()
        downsample_scale = self._get_camera_scale_factor(self.RGB_CAMERA_LABEL)

        # Extract bbox data for projection using utility function
        projection_data = extract_bbox_projection_data(self.pd_provider, evl_3d_bboxes)

        # Collect all projection results for batching
        all_projected_lines = []
        all_line_colors = []
        label_positions = []
        label_texts = []
        label_colors = []

        # Project each bounding box using utility function
        for data in projection_data:
            projection_result = project_3d_bbox_to_2d_camera(
                corners_in_world=data["corners_world"],
                T_world_camera=T_world_camera,
                camera_calibration=self.rgb_camera_calibration,
                image_width=image_width,  # Use original dimensions for projection
                image_height=image_height,
                label=data["label"],
            )
            if not projection_result:
                continue
            projected_lines, line_colors, label_position = projection_result

            # Collect projection data for batching
            if projected_lines:
                # Scale projected line coordinates for downsampling
                if downsample_scale != 1.0:
                    scaled_lines = []
                    for line_strip in projected_lines:
                        scaled_strip = [
                            [pt[0] * downsample_scale, pt[1] * downsample_scale]
                            for pt in line_strip
                        ]
                        scaled_lines.append(scaled_strip)
                    projected_lines = scaled_lines

                all_projected_lines.extend(projected_lines)
                if line_colors and len(line_colors) >= len(projected_lines):
                    all_line_colors.extend(line_colors[: len(projected_lines)])
                else:
                    all_line_colors.extend(
                        [self.config.evl_3d_bbox_color] * len(projected_lines)
                    )

                if label_position and data["label"]:
                    # Scale label position for downsampling
                    if downsample_scale != 1.0:
                        label_position = [
                            label_position[0] * downsample_scale,
                            label_position[1] * downsample_scale,
                        ]

                    label_positions.append(label_position)
                    label_texts.append(data["label"])
                    label_colors.append(plot_style.color)

        # Batch and log all results with scaling
        if all_projected_lines:
            rr.log(
                f"{self.RGB_CAMERA_LABEL}/{plot_style.label}/wireframes",
                rr.LineStrips2D(
                    all_projected_lines,
                    colors=all_line_colors,
                    radii=plot_style.plot_2d_size
                    * downsample_scale,  # Scale line thickness
                ),
            )

        if label_positions:
            rr.log(
                f"{self.RGB_CAMERA_LABEL}/{plot_style.label}/labels",
                rr.Points2D(
                    positions=label_positions,
                    labels=label_texts,
                    colors=plot_style.color,
                ),
            )

    def plot_hand_object_interaction_data(
        self, hoi_data_list: list[HandObjectInteractionData], device_time_ns: int
    ):
        """Query HOI data and overlay segmentation masks on RGB image"""
        rr.log(
            f"{self.RGB_CAMERA_LABEL}/hoi_overlay",
            rr.Clear.recursive(),
        )

        # filter out HOI data too far away from the current frame
        if (
            not hoi_data_list
            or abs(hoi_data_list[0].timestamp_ns - device_time_ns)
            > self.rgb_frame_interval_ns / 2
        ):
            return

        category_to_plot_style = {
            1: get_plot_style(PlotEntity.HOI_LEFT_HAND),
            2: get_plot_style(PlotEntity.HOI_RIGHT_HAND),
            3: get_plot_style(PlotEntity.HOI_INTERACTING_OBJECT),
        }

        # Determine mask shape from the first valid mask
        mask_shape = next(
            (
                mask.shape
                for hoi_data in hoi_data_list
                for mask in hoi_data.masks
                if mask is not None and mask.size > 0
            ),
            None,
        )
        if mask_shape is None:
            # No valid masks found, nothing to plot
            return

        # Initialize combined RGBA overlay
        combined_rgba_overlay = np.zeros((*mask_shape, 4), dtype=np.uint8)

        # Overlay each category's mask with its color
        for hoi_data in hoi_data_list:
            category_id = hoi_data.category_id
            plot_style = category_to_plot_style.get(category_id, None)
            if not plot_style:
                raise ValueError(
                    f"Unknown category ID {category_id} for HOI data. Cannot plot."
                )
            for mask in hoi_data.masks:
                if mask is None or mask.size == 0:
                    continue
                foreground_pixels = mask > 0
                combined_rgba_overlay[foreground_pixels] = plot_style.color

        # Apply downsampling to HOI overlay if needed
        downsample_scale = self._get_camera_scale_factor(self.RGB_CAMERA_LABEL)
        if downsample_scale != 1.0:
            combined_rgba_overlay = self._resize_image(
                combined_rgba_overlay, self.config.rgb_downsample_factor
            )

        # Log the combined segmentation overlay as an image
        rr.log(
            f"{self.RGB_CAMERA_LABEL}/hoi_overlay/combined",
            rr.Image(combined_rgba_overlay),
        )

    def plot_stereo_depth_3d_(
        self,
        depth_map: np.ndarray,
        camera_intrinsics_and_pose: CameraIntrinsicsAndPose,
    ) -> None:
        """Project 2D depth map as 3D point cloud."""
        # Clear previous depth image
        # Set up depth camera in world coordinate system
        plot_style = get_plot_style(PlotEntity.STEREO_DEPTH)
        if depth_map is None or camera_intrinsics_and_pose is None:
            return

        # Get original camera intrinsics
        original_fx, original_fy = (
            camera_intrinsics_and_pose.camera_projection.get_focal_lengths()
        )
        original_ux, original_uy = (
            camera_intrinsics_and_pose.camera_projection.get_principal_point()
        )
        factor = self.config.depth_image_downsample_factor

        # Scale intrinsics to match the downsampled (subsampled) image size
        scaled_fx = original_fx / factor
        scaled_fy = original_fy / factor
        scaled_ux = original_ux / factor
        scaled_uy = original_uy / factor

        subsampled_depth_map = self._resize_image(depth_map, factor)

        rr.log(
            f"world/{plot_style.label}",
            rr.Pinhole(
                resolution=[
                    subsampled_depth_map.shape[1],
                    subsampled_depth_map.shape[0],
                ],
                focal_length=[scaled_fx, scaled_fy],
                principal_point=[scaled_ux, scaled_uy],
            ),
            static=True,
        )
        rr.log(
            f"world/{plot_style.label}",
            ToTransform3D(
                camera_intrinsics_and_pose.transform_world_camera, axis_length=0.02
            ),
        )

        # Define the scaling factor: depth data is in millimeters, ReRun expects meters, 1m = 1000mm
        DEPTH_IMAGE_SCALING = 1000

        # Log the depth image with proper world positioning and scaling
        rr.log(
            f"world/{plot_style.label}",
            rr.DepthImage(
                subsampled_depth_map,
                meter=DEPTH_IMAGE_SCALING,
                colormap="Magma",
                point_fill_ratio=plot_style.plot_3d_size,
            ),
        )

    def plot_stereo_depth_data(
        self,
        camera_intrinsics_and_pose: CameraIntrinsicsAndPose,
        depth_map: np.ndarray,
        rectified_slam_front_left_image: np.ndarray,
        query_timestamp_ns: int,
    ) -> None:
        """Plot the stereo depth data."""
        plot_style = get_plot_style(PlotEntity.STEREO_DEPTH)
        rr.log(
            "depth_image",
            rr.Clear.recursive(),
        )
        rr.log(
            "rectified_slam_front_left",
            rr.Clear.recursive(),
        )
        rr.log(
            f"world/{plot_style.label}",
            rr.Clear.recursive(),
        )
        if (
            depth_map is None
            or rectified_slam_front_left_image is None
            or rectified_slam_front_left_image is None
            or abs(camera_intrinsics_and_pose.timestamp_ns - query_timestamp_ns)
            > (self.rgb_frame_interval_ns / 2)
        ):
            return
        self.plot_image(
            frame=rectified_slam_front_left_image,
            camera_label="rectified_slam_front_left",
            jpeg_quality=self.config.depth_and_slam_jpeg_quality,
        )
        self.plot_stereo_depth_3d_(depth_map, camera_intrinsics_and_pose)
