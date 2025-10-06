# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import pathlib
import tempfile
import unittest

from aria_gen2_pilot_dataset.data_provider.heart_rate_data_provider import (
    HeartRateDataProvider,
)

from projectaria_tools.core.sensor_data import TimeQueryOptions


class TestHeartRateDataProvider(unittest.TestCase):
    """Test heart rate data provider functionality."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Create test CSV data with unsorted timestamps to test sorting
        self.test_data = (
            "timestamp_ns,heart_rate_bpm\n"
            "3000000000,68\n"
            "1000000000,72\n"
            "4000000000,80\n"
            "2000000000,75\n"
            "5000000000,65\n"
        )

        # Create temporary file with test data
        self.temp_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        )
        self.temp_file.write(self.test_data)
        self.temp_file.close()

    def tearDown(self) -> None:
        """Clean up test fixtures."""
        import pathlib

        pathlib.Path(self.temp_file.name).unlink(missing_ok=True)

    def test_provider_initialization(self):
        """Test heart rate data provider initialization."""
        provider = HeartRateDataProvider(self.temp_file.name)
        self.assertIsNotNone(provider)
        self.assertEqual(provider.get_heart_rate_total_number(), 5)

    def test_get_heart_rate_by_index(self):
        """Test getting heart rate data by index."""
        provider = HeartRateDataProvider(self.temp_file.name)

        # Test valid indices (data should be sorted)
        heart_rate_data = provider.get_heart_rate_by_index(0)
        self.assertIsNotNone(heart_rate_data)
        self.assertEqual(heart_rate_data.timestamp_ns, 1000000000)
        self.assertEqual(heart_rate_data.heart_rate_bpm, 72)

        heart_rate_data = provider.get_heart_rate_by_index(2)
        self.assertIsNotNone(heart_rate_data)
        self.assertEqual(heart_rate_data.timestamp_ns, 3000000000)
        self.assertEqual(heart_rate_data.heart_rate_bpm, 68)

        # Test invalid indices
        self.assertIsNone(provider.get_heart_rate_by_index(-1))
        self.assertIsNone(provider.get_heart_rate_by_index(10))

    def test_get_heart_rate_by_timestamp_closest(self):
        """Test getting heart rate data by timestamp with closest option."""
        provider = HeartRateDataProvider(self.temp_file.name)

        # Test exact match
        heart_rate_data = provider.get_heart_rate_by_timestamp_ns(
            2000000000, TimeQueryOptions.CLOSEST
        )
        self.assertIsNotNone(heart_rate_data)
        self.assertEqual(heart_rate_data.heart_rate_bpm, 75)

        # Test closest match
        heart_rate_data = provider.get_heart_rate_by_timestamp_ns(
            2600000000, TimeQueryOptions.CLOSEST
        )
        self.assertIsNotNone(heart_rate_data)
        self.assertEqual(heart_rate_data.timestamp_ns, 3000000000)
        self.assertEqual(heart_rate_data.heart_rate_bpm, 68)

    def test_get_heart_rate_by_timestamp_before(self):
        """Test getting heart rate data by timestamp with before option."""
        provider = HeartRateDataProvider(self.temp_file.name)

        # Test before match
        heart_rate_data = provider.get_heart_rate_by_timestamp_ns(
            3500000000, TimeQueryOptions.BEFORE
        )
        self.assertIsNotNone(heart_rate_data)
        self.assertEqual(heart_rate_data.timestamp_ns, 3000000000)
        self.assertEqual(heart_rate_data.heart_rate_bpm, 68)

        # Test no match before first timestamp
        heart_rate_data = provider.get_heart_rate_by_timestamp_ns(
            500000000, TimeQueryOptions.BEFORE
        )
        self.assertIsNone(heart_rate_data)

    def test_get_heart_rate_by_timestamp_after(self):
        """Test getting heart rate data by timestamp with after option."""
        provider = HeartRateDataProvider(self.temp_file.name)

        # Test after match
        heart_rate_data = provider.get_heart_rate_by_timestamp_ns(
            3500000000, TimeQueryOptions.AFTER
        )
        self.assertIsNotNone(heart_rate_data)
        self.assertEqual(heart_rate_data.timestamp_ns, 4000000000)
        self.assertEqual(heart_rate_data.heart_rate_bpm, 80)

        # Test no match after last timestamp
        heart_rate_data = provider.get_heart_rate_by_timestamp_ns(
            6000000000, TimeQueryOptions.AFTER
        )
        self.assertIsNone(heart_rate_data)

    def test_get_heart_rate_total_number(self):
        """Test getting total number of heart rate entries."""
        provider = HeartRateDataProvider(self.temp_file.name)
        self.assertEqual(provider.get_heart_rate_total_number(), 5)

    def test_empty_csv_file(self):
        """Test provider with empty CSV file throws error."""

        empty_data = ""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(empty_data)
            f.close()

        try:
            # Should throw RuntimeError for empty data file via check_valid_file utility
            with self.assertRaises(RuntimeError) as context:
                HeartRateDataProvider(f.name)

            # Verify error message mentions empty file
            error_msg = str(context.exception).lower()
            self.assertTrue(
                "empty" in error_msg,
                f"Expected empty file error, got: {context.exception}",
            )
        finally:
            pathlib.Path(f.name).unlink(missing_ok=True)

    def test_nonexistent_file(self):
        """Test provider with nonexistent file throws error."""
        # Should throw RuntimeError for missing file via check_valid_file utility
        with self.assertRaises(RuntimeError) as context:
            HeartRateDataProvider("/nonexistent/path.csv")

        # Verify error message mentions file not found or similar
        error_msg = str(context.exception).lower()
        self.assertTrue(
            "not found" in error_msg
            or "does not exist" in error_msg
            or "no such file" in error_msg
            or "file does not exist" in error_msg,
            f"Expected file not found error, got: {context.exception}",
        )
