# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest
from unittest.mock import DEFAULT, MagicMock, patch

import numpy as np
import rerun as rr
from aria_gen2_pilot_dataset.data_provider.aria_gen2_pilot_dataset_data_types import (
    HandObjectInteractionData,
)
from aria_gen2_pilot_dataset.visualization.aria_gen2_pilot_data_visualizer import (
    AriaGen2PilotDataVisualizer,
    HOI_BACKGROUND_CLASS_ID,
    HOI_CATEGORY_TO_PLOT_ENTITY,
)
from aria_gen2_pilot_dataset.visualization.aria_gen2_pilot_viewer_config import (
    AriaGen2PilotViewerConfig,
)

RR = "aria_gen2_pilot_dataset.visualization.aria_gen2_pilot_data_visualizer.rr"
RR_LOG = f"{RR}.log"

RGB_IMAGE_SIZE = (2560, 1920)
SLAM_IMAGE_SIZE = (1408, 1408)
RGB_FRAME_INTERVAL_NS = 100_000_000  # 10 Hz
TIMESTAMP_NS = 5_093_867_000_000


def _make_visualizer() -> AriaGen2PilotDataVisualizer:
    rgb_calib = MagicMock()
    rgb_calib.get_image_size.return_value = RGB_IMAGE_SIZE
    slam_calib = MagicMock()
    slam_calib.get_image_size.return_value = SLAM_IMAGE_SIZE

    device_calibration = MagicMock()
    device_calibration.get_camera_calib.side_effect = (
        lambda label: rgb_calib if label == "camera-rgb" else slam_calib
    )

    data_provider = MagicMock()
    data_provider.vrs_data_provider.get_device_calibration.return_value = (
        device_calibration
    )

    visualizer = AriaGen2PilotDataVisualizer(data_provider, AriaGen2PilotViewerConfig())
    visualizer.rgb_frame_interval_ns = RGB_FRAME_INTERVAL_NS
    return visualizer


def _make_hoi_data(
    masks: list[np.ndarray], category_id: int = 1
) -> list[HandObjectInteractionData]:
    return [
        HandObjectInteractionData(
            timestamp_ns=TIMESTAMP_NS,
            category_id=category_id,
            masks=masks,
            bboxes=[[100.0, 100.0, 100.0, 100.0]] * len(masks),
            scores=[1.0] * len(masks),
        )
    ]


def _make_mask(row_start: int = 100) -> np.ndarray:
    mask = np.zeros((RGB_IMAGE_SIZE[1], RGB_IMAGE_SIZE[0]), dtype=np.uint8)
    mask[row_start : row_start + 100, 100:200] = 1
    return mask


class TestHandObjectInteractionOverlay(unittest.TestCase):
    """The HOI overlay is a child image of `camera-rgb` and must not hide it."""

    def setUp(self) -> None:
        self.visualizer = _make_visualizer()

    def _log_hoi(self, hoi_data: list[HandObjectInteractionData]) -> dict:
        with patch(RR_LOG) as mock_log:
            self.visualizer.plot_hand_object_interaction_data(hoi_data, TIMESTAMP_NS)
        return {call.args[0]: call.args[1] for call in mock_log.call_args_list}

    def test_overlay_is_a_segmentation_image(self) -> None:
        logged = self._log_hoi(_make_hoi_data([_make_mask()]))

        overlay = logged.get("camera-rgb/hoi_overlay/combined")
        self.assertIsNotNone(overlay, "HOI overlay was not logged")
        # A full-frame RGBA `rr.Image` child hides the camera frame: rerun paints its
        # transparent background as opaque black over the parent entity's image.
        self.assertNotIsInstance(overlay, rr.Image)
        self.assertIsInstance(overlay, rr.SegmentationImage)

    def test_downsampling_does_not_invent_class_ids(self) -> None:
        left_hand = _make_hoi_data([_make_mask(row_start=100)], category_id=1)
        interacting_object = _make_hoi_data([_make_mask(row_start=400)], category_id=3)

        logged = self._log_hoi(left_hand + interacting_object)

        overlay = logged["camera-rgb/hoi_overlay/combined"]
        buffer = overlay.buffer.as_arrow_array().to_pylist()[0]
        class_ids = np.frombuffer(bytes(buffer), dtype=np.uint8)
        # Interpolating labels would blend ids 1 and 3 into the unannotated category 2.
        self.assertEqual(
            set(np.unique(class_ids).tolist()), {HOI_BACKGROUND_CLASS_ID, 1, 3}
        )

    def test_no_overlay_logged_when_masks_are_empty(self) -> None:
        logged = self._log_hoi(_make_hoi_data([]))

        self.assertIn("camera-rgb/hoi_overlay", logged)
        self.assertNotIn("camera-rgb/hoi_overlay/combined", logged)

    def test_no_overlay_logged_when_masks_are_zero_sized(self) -> None:
        logged = self._log_hoi(_make_hoi_data([np.empty((0, 0), dtype=np.uint8)]))

        self.assertNotIn("camera-rgb/hoi_overlay/combined", logged)

    def test_no_overlay_logged_for_stale_hoi_data(self) -> None:
        stale = _make_hoi_data([_make_mask()])
        stale[0].timestamp_ns = TIMESTAMP_NS + RGB_FRAME_INTERVAL_NS

        logged = self._log_hoi(stale)

        self.assertNotIn("camera-rgb/hoi_overlay/combined", logged)

    def test_unknown_category_is_rejected(self) -> None:
        unknown = _make_hoi_data([_make_mask()])
        unknown[0].category_id = 99

        with patch(RR_LOG):
            with self.assertRaises(ValueError):
                self.visualizer.plot_hand_object_interaction_data(unknown, TIMESTAMP_NS)


class TestHoiAnnotationContext(unittest.TestCase):
    """Class 0 must resolve to a fully transparent color, or the overlay hides the frame."""

    def setUp(self) -> None:
        self.visualizer = _make_visualizer()

    def _log_context(self):
        with patch(RR_LOG) as mock_log:
            self.visualizer.plot_hoi_annotation_context()
        return mock_log

    @staticmethod
    def _class_id_to_color(context) -> dict[int, int]:
        """Class id -> color packed as 0xRRGGBBAA."""
        entries = context.context.as_arrow_array().to_pylist()[0]
        return {
            entry["class_description"]["info"]["id"]: entry["class_description"][
                "info"
            ]["color"]
            for entry in entries
        }

    def test_context_is_logged_statically_on_the_camera_entity(self) -> None:
        mock_log = self._log_context()

        mock_log.assert_called_once()
        call = mock_log.call_args
        self.assertEqual(call.args[0], "camera-rgb")
        self.assertIsInstance(call.args[1], rr.AnnotationContext)
        self.assertTrue(call.kwargs["static"])

    def test_background_class_is_fully_transparent(self) -> None:
        colors = self._class_id_to_color(self._log_context().call_args.args[1])

        self.assertEqual(colors[HOI_BACKGROUND_CLASS_ID] & 0xFF, 0)
        for category_id in HOI_CATEGORY_TO_PLOT_ENTITY:
            self.assertEqual(colors[category_id] & 0xFF, 0xFF)

    def test_every_hoi_category_has_a_color(self) -> None:
        colors = self._class_id_to_color(self._log_context().call_args.args[1])

        self.assertEqual(
            set(colors), {HOI_BACKGROUND_CLASS_ID} | set(HOI_CATEGORY_TO_PLOT_ENTITY)
        )


class TestRerunViewerSpawn(unittest.TestCase):
    """The viewer must spawn the buck-provided rerun CLI, not one found on PATH."""

    def setUp(self) -> None:
        self.visualizer = _make_visualizer()

    def _init_rerun(self, rrd_output_path: str = "") -> dict:
        with patch.multiple(
            RR, init=DEFAULT, spawn=DEFAULT, save=DEFAULT, send_blueprint=DEFAULT
        ) as mocks:
            self.visualizer.initialize_rerun_and_blueprint(rrd_output_path)
        return mocks

    def test_spawn_uses_rerun_path_from_environment(self) -> None:
        cli_path = "/buck-out/gen/third-party/pypi/rerun-sdk/rerun"

        with patch.dict(os.environ, {"RERUN_PATH": cli_path}):
            mocks = self._init_rerun()

        self.assertFalse(mocks["init"].call_args.kwargs["spawn"])
        mocks["spawn"].assert_called_once_with(executable_path=cli_path)
        mocks["save"].assert_not_called()

    def test_spawn_falls_back_to_path_lookup_when_unset(self) -> None:
        with patch.dict(os.environ):
            os.environ.pop("RERUN_PATH", None)
            mocks = self._init_rerun()

        mocks["spawn"].assert_called_once_with(executable_path=None)

    def test_rrd_output_path_saves_without_spawning(self) -> None:
        mocks = self._init_rerun("/tmp/out.rrd")

        mocks["save"].assert_called_once_with("/tmp/out.rrd")
        mocks["spawn"].assert_not_called()
