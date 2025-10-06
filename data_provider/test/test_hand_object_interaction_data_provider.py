# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import json
import os
import tempfile
import unittest

import numpy as np
from aria_gen2_pilot_dataset.data_provider.aria_gen2_pilot_dataset_data_types import (
    HandObjectInteractionData,
)
from aria_gen2_pilot_dataset.data_provider.hand_object_interaction_data_provider import (
    HandObjectInteractionDataProvider,
)
from projectaria_tools.core.sensor_data import TimeQueryOptions


class TestHandObjectInteractionDataProvider(unittest.TestCase):
    """Test suite for HandObjectInteractionDataProvider."""

    def setUp(self):
        """Set up test data and create temporary test file."""
        # Test data mimicking the structure from manifold example
        self.test_data = [
            {
                "segmentation": {
                    "size": [1512, 2016],
                    "counts": "TUk`13U_10L8L0K7N0UJO1O1N2M3L4K5J5J5I6H7G7G8F8F9D;E;D;E;D<C<C=B=B>A>A?@?@?@?@A>A>B=B=C<C<D;D;E:E:F9F9G8G8H7H7I6I6J5J5K4K4L3L3M2M2N1N1O0O0O0O0O1N1N2M2M3L3L4K4K5J5J6I6I7H7H8G8G9F9F:E:E;D;D<C<C=B=B>A>A?@?@A>A>B=B=C<C<D;D;E:F9F9G8G8H7H7I6I6J5J5K4K4L3L3M2M2N1N1O0O0",
                },
                "bbox": [1050.18, 738.11, 325.16, 535.08],
                "score": 1.0,
                "image_id": 2620886,
                "category_id": 3,
            },
            {
                "segmentation": {
                    "size": [1512, 2016],
                    "counts": "aab`13U_10M7M0L6O0TKO1N1O2M3L4K5J5J5I6H7G7G8F8F9E9E:D:D;C;C<B<B=A=A>@>@?@?@A>A>B=B=C<C<D;D;E:E:F9F9G8G8H7H7I6I6J5J5K4K4L3L3M2M2N1N1O0O0",
                },
                "bbox": [589.23, 543.67, 298.45, 412.89],
                "score": 0.95,
                "image_id": 2620886,
                "category_id": 1,
            },
            {
                "segmentation": {
                    "size": [1512, 2016],
                    "counts": "xyz`13U_10P4P0O3Q0RLO1O1O2N3M4L5K6J7I8H9G:F;E<D=C>B?A@A@B?B?C>C>D=D=E<E<F;F;G:G:H9H9I8I8J7J7K6K6L5L5M4M4N3N3O2O2O1O1",
                },
                "bbox": [1234.56, 890.12, 156.78, 234.90],
                "score": 0.88,
                "image_id": 2620887,
                "category_id": 2,
            },
            {
                "segmentation": {
                    "size": [1512, 2016],
                    "counts": "def`13U_10Q3Q0P2R0SMO1N1O2M3L4K5J6I7H8G9F:E;D<C=B>A?@@A?A?B>B>C=C=D<D<E;E;F:F:G9G9H8H8I7I7J6J6K5K5L4L4M3M3N2N2O1O1",
                },
                "bbox": [456.78, 123.45, 345.67, 456.78],
                "score": 0.92,
                "image_id": 2620887,
                "category_id": 3,
            },
        ]

        # Create temporary file with test data
        self.temp_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        )
        json.dump(self.test_data, self.temp_file)
        self.temp_file.close()

        # Initialize data provider
        self.provider = HandObjectInteractionDataProvider(self.temp_file.name)

    def tearDown(self):
        """Clean up temporary files."""
        os.unlink(self.temp_file.name)

    def test_get_hoi_data_by_timestamp_ns(self):
        """Test querying by timestamp."""
        # Test exact timestamp match
        timestamp_ns = 2620886000000
        data = self.provider.get_hoi_data_by_timestamp_ns(
            timestamp_ns, TimeQueryOptions.CLOSEST
        )

        self.assertIsNotNone(data)
        self.assertEqual(len(data), 2)  # 2 categories at this timestamp

        # Check category distribution
        categories = [item.category_id for item in data]
        self.assertIn(1, categories)  # left hand
        self.assertIn(3, categories)  # object

        # Verify timestamp conversion and new structure
        for item in data:
            self.assertEqual(item.timestamp_ns, timestamp_ns)

            # Check new decoded format
            self.assertIsInstance(item.masks, list)
            self.assertIsInstance(item.bboxes, list)
            self.assertIsInstance(item.scores, list)

            # Each category should have at least one mask/bbox/score
            self.assertGreater(len(item.masks), 0)
            self.assertEqual(len(item.masks), len(item.bboxes))
            self.assertEqual(len(item.masks), len(item.scores))

            # Check mask properties
            for mask in item.masks:
                self.assertIsInstance(mask, np.ndarray)
                self.assertEqual(mask.shape, (1512, 2016))  # from test data
                self.assertEqual(mask.dtype, np.uint8)

    def test_get_hoi_data_by_index(self):
        """Test querying by index."""
        # Test valid index
        data = self.provider.get_hoi_data_by_index(0)
        self.assertIsNotNone(data)
        self.assertIsInstance(data, list)

        # Test invalid index
        data = self.provider.get_hoi_data_by_index(999)
        self.assertIsNone(data)

    def test_time_query_options(self):
        """Test different time query options."""
        # Test CLOSEST with exact match
        exact_timestamp = 2620886000000
        data_closest = self.provider.get_hoi_data_by_timestamp_ns(
            exact_timestamp, TimeQueryOptions.CLOSEST
        )
        self.assertIsNotNone(data_closest)

        # Test CLOSEST with approximate timestamp
        approx_timestamp = 2620886500000  # Between two timestamps
        data_approx = self.provider.get_hoi_data_by_timestamp_ns(
            approx_timestamp, TimeQueryOptions.CLOSEST
        )
        self.assertIsNotNone(data_approx)

        # Test BEFORE - should return data from before the queried timestamp
        data_before = self.provider.get_hoi_data_by_timestamp_ns(
            2620887500000,
            TimeQueryOptions.BEFORE,  # Query between two timestamps
        )
        self.assertIsNotNone(data_before)
        # Should return data from timestamp 2620887000000 (the latest before query)
        self.assertEqual(data_before[0].timestamp_ns, 2620887000000)

        # Test AFTER - should return data from after the queried timestamp
        data_after = self.provider.get_hoi_data_by_timestamp_ns(
            2620885000000,
            TimeQueryOptions.AFTER,  # Query before first timestamp
        )
        self.assertIsNotNone(data_after)
        # Should return data from timestamp 2620886000000 (the first after query)
        self.assertEqual(data_after[0].timestamp_ns, 2620886000000)

    def test_data_structure_validation(self):
        """Test that loaded data structures are correctly formed."""
        timestamp_ns = 2620886000000
        data = self.provider.get_hoi_data_by_timestamp_ns(timestamp_ns)

        self.assertIsNotNone(data)

        for item in data:
            self.assertIsInstance(item, HandObjectInteractionData)

            # Check required fields for new structure
            self.assertIsInstance(item.timestamp_ns, int)
            self.assertIsInstance(item.category_id, int)
            self.assertIsInstance(item.masks, list)
            self.assertIsInstance(item.bboxes, list)
            self.assertIsInstance(item.scores, list)

            # Check valid categories
            self.assertIn(item.category_id, [1, 2, 3])

            # Check that masks, bboxes, and scores have same length
            self.assertEqual(len(item.masks), len(item.bboxes))
            self.assertEqual(len(item.masks), len(item.scores))
            self.assertGreater(len(item.masks), 0)  # At least one instance per category

            # Check bbox format [x, y, width, height]
            for bbox in item.bboxes:
                self.assertEqual(len(bbox), 4)
                for val in bbox:
                    self.assertIsInstance(val, (int, float))
                    self.assertGreaterEqual(val, 0)

            # Check score range
            for score in item.scores:
                self.assertIsInstance(score, float)
                self.assertGreaterEqual(score, 0.0)
                self.assertLessEqual(score, 1.0)

            # Check mask properties
            for mask in item.masks:
                self.assertIsInstance(mask, np.ndarray)
                self.assertEqual(mask.dtype, np.uint8)
                self.assertEqual(len(mask.shape), 2)  # 2D array

    def test_empty_data_handling(self):
        """Test behavior with empty data file."""
        # Create empty test file
        empty_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump([], empty_file)
        empty_file.close()

        try:
            with self.assertRaises(RuntimeError):
                HandObjectInteractionDataProvider(empty_file.name)
        finally:
            os.unlink(empty_file.name)

    def test_missing_file_handling(self):
        """Test behavior with missing file."""
        with self.assertRaises(FileNotFoundError):
            HandObjectInteractionDataProvider("/nonexistent/path.json")
