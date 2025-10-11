# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import tempfile
import unittest

from aria_gen2_pilot_dataset.data_provider.diarization_data_provider import (
    DiarizationDataProvider,
)


class TestDiarizationDataProvider(unittest.TestCase):
    """Test suite for DiarizationDataProvider."""

    def setUp(self):
        """Set up test data and create temporary CSV file."""
        # Comprehensive test data with overlapping segments and edge cases
        self.csv_data = """start_timestamp_ns,end_timestamp_ns,speaker,content
1000000000,2000000000,Speaker1,Hello world
1500000000,2500000000,Speaker2,Good morning overlapping
2500000000,3500000000,Speaker2,How are you
3000000000,4000000000,Speaker1,I am fine thank you
4000000000,5000000000,Speaker1,Nice weather today
5500000000,6000000000,Speaker3,See you later
6500000000,7000000000,Speaker1,Goodbye everyone
"""
        # Create temp file with test data
        self.temp_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        )
        self.temp_file.write(self.csv_data)
        self.temp_file.close()

        # Create provider
        self.provider = DiarizationDataProvider(self.temp_file.name)

    def tearDown(self):
        """Clean up temp files."""
        import os

        os.unlink(self.temp_file.name)

    def test_data_loading_and_structure(self):
        """Test that data loads correctly and maintains structure."""
        # Test data loading
        self.assertEqual(len(self.provider.diarization_data), 7)

        # Test that data is sorted by start timestamp
        start_timestamps = [
            data.start_timestamp_ns for data in self.provider.diarization_data
        ]
        self.assertEqual(start_timestamps, sorted(start_timestamps))

        # Test start timestamp list matches
        self.assertEqual(self.provider.start_timestamp_ns_list, start_timestamps)

        # Test data structure
        first_segment = self.provider.diarization_data[0]
        self.assertEqual(first_segment.start_timestamp_ns, 1000000000)
        self.assertEqual(first_segment.end_timestamp_ns, 2000000000)
        self.assertEqual(first_segment.speaker, "Speaker1")
        self.assertEqual(first_segment.content, "Hello world")

    def test_get_diarization_data_by_timestamp_ns_single_segment(self):
        """Test timestamp search with single matching segment."""
        # Test timestamp within first segment
        result = self.provider.get_diarization_data_by_timestamp_ns(1500000000)
        self.assertEqual(len(result), 2)  # Should match first two overlapping segments
        speakers = [seg.speaker for seg in result]
        self.assertIn("Speaker1", speakers)
        self.assertIn("Speaker2", speakers)

        # Test timestamp at segment boundary (start)
        result = self.provider.get_diarization_data_by_timestamp_ns(1000000000)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].speaker, "Speaker1")

        # Test timestamp at segment boundary (end - inclusive)
        result = self.provider.get_diarization_data_by_timestamp_ns(2000000000)
        self.assertGreaterEqual(
            len(result), 1
        )  # Should include segment ending at this time

    def test_get_diarization_data_by_timestamp_ns_overlapping_segments(self):
        """Test timestamp search with overlapping segments."""
        # Test timestamp in overlapping region (1500-2000ns)
        result = self.provider.get_diarization_data_by_timestamp_ns(1750000000)
        self.assertEqual(len(result), 2)

        # Check both speakers are present
        speakers = [seg.speaker for seg in result]
        self.assertIn("Speaker1", speakers)
        self.assertIn("Speaker2", speakers)

        # Test content matches expected segments
        contents = [seg.content for seg in result]
        self.assertIn("Hello world", contents)
        self.assertIn("Good morning overlapping", contents)

    def test_get_diarization_data_by_timestamp_ns_no_match(self):
        """Test timestamp search with no matching segments."""
        # Test timestamp before all segments
        result = self.provider.get_diarization_data_by_timestamp_ns(500000000)
        self.assertEqual(len(result), 0)

        # Test timestamp after all segments
        result = self.provider.get_diarization_data_by_timestamp_ns(8000000000)
        self.assertEqual(len(result), 0)

        # Test timestamp in gap between segments
        result = self.provider.get_diarization_data_by_timestamp_ns(
            5250000000
        )  # Between 5000-5500
        self.assertEqual(len(result), 0)

    def test_get_diarization_data_by_start_and_end_timestamps_full_overlap(self):
        """Test range search with full segment overlap."""
        # Query range that fully contains a segment
        result = self.provider.get_diarization_data_by_start_and_end_timestamps(
            500000000, 2500000000
        )
        self.assertGreaterEqual(len(result), 2)  # Should include first two segments

        # Check that returned segments are actually within or overlapping the range
        for seg in result:
            # Segment overlaps if: seg_start <= query_end AND seg_end >= query_start
            self.assertTrue(
                seg.start_timestamp_ns <= 2500000000
            )  # seg_start <= query_end
            self.assertTrue(seg.end_timestamp_ns >= 500000000)  # seg_end >= query_start

    def test_get_diarization_data_by_start_and_end_timestamps_partial_overlap(self):
        """Test range search with partial segment overlap."""
        # Query range that partially overlaps segments
        result = self.provider.get_diarization_data_by_start_and_end_timestamps(
            1500000000, 3000000000
        )

        # Should include segments that overlap with this range
        self.assertGreater(len(result), 0)

        # Check overlap logic for each returned segment
        for seg in result:
            # Verify actual overlap: segment overlaps if seg_start <= query_end AND seg_end >= query_start
            overlaps = (
                seg.start_timestamp_ns <= 3000000000
                and seg.end_timestamp_ns >= 1500000000
            )
            self.assertTrue(
                overlaps,
                f"Segment ({seg.start_timestamp_ns}-{seg.end_timestamp_ns}) should overlap with query (1500000000-3000000000)",
            )

    def test_get_diarization_data_by_start_and_end_timestamps_no_overlap(self):
        """Test range search with no overlapping segments."""
        # Query range in gap between segments
        result = self.provider.get_diarization_data_by_start_and_end_timestamps(
            5100000000, 5400000000
        )
        self.assertEqual(len(result), 0)

        # Query range before all segments
        result = self.provider.get_diarization_data_by_start_and_end_timestamps(
            100000000, 500000000
        )
        self.assertEqual(len(result), 0)

        # Query range after all segments
        result = self.provider.get_diarization_data_by_start_and_end_timestamps(
            8000000000, 9000000000
        )
        self.assertEqual(len(result), 0)

    def test_get_diarization_data_by_start_and_end_timestamps_edge_cases(self):
        """Test range search edge cases."""
        # Query with start == end (point query)
        result = self.provider.get_diarization_data_by_start_and_end_timestamps(
            1500000000, 1500000000
        )
        self.assertGreater(len(result), 0)  # Should still find overlapping segments

        # Query with end < start (invalid range)
        result = self.provider.get_diarization_data_by_start_and_end_timestamps(
            3000000000, 2000000000
        )
        self.assertEqual(len(result), 0)

        # Query spanning entire dataset
        result = self.provider.get_diarization_data_by_start_and_end_timestamps(
            0, 10000000000
        )
        self.assertEqual(len(result), 7)  # Should return all segments

    def test_speaker_filtering_and_content_verification(self):
        """Test speaker identification and content verification."""
        # Get all segments from Speaker1
        all_segments = []
        for timestamp in [1500000000, 3500000000, 4500000000, 6750000000]:
            segments = self.provider.get_diarization_data_by_timestamp_ns(timestamp)
            all_segments.extend(segments)

        speaker1_segments = [seg for seg in all_segments if seg.speaker == "Speaker1"]
        self.assertGreater(len(speaker1_segments), 0)

        # Verify content for specific speaker
        speaker1_contents = [seg.content for seg in speaker1_segments]
        expected_contents = [
            "Hello world",
            "I am fine thank you",
            "Nice weather today",
            "Goodbye everyone",
        ]

        for content in expected_contents:
            # Check if this content appears in any Speaker1 segment
            found = any(content == seg_content for seg_content in speaker1_contents)
            self.assertTrue(
                found, f"Expected content '{content}' not found in Speaker1 segments"
            )

    def test_temporal_ordering_and_binary_search_efficiency(self):
        """Test that temporal ordering is maintained and binary search works efficiently."""
        # Test that segments are returned in chronological order for range queries
        result = self.provider.get_diarization_data_by_start_and_end_timestamps(
            1000000000, 5000000000
        )

        # Verify segments are sorted by start timestamp
        start_times = [seg.start_timestamp_ns for seg in result]
        self.assertEqual(start_times, sorted(start_times))

        # Test binary search boundary conditions
        # Query exactly at segment boundaries
        boundary_tests = [
            (1000000000, 1000000000),  # Exact start of first segment
            (2000000000, 2000000000),  # Exact end of first segment
            (7000000000, 7000000000),  # Exact end of last segment
        ]

        for start_ts, end_ts in boundary_tests:
            result = self.provider.get_diarization_data_by_start_and_end_timestamps(
                start_ts, end_ts
            )
            # Should find segments that include these boundaries
            if start_ts <= 7000000000:  # Within data range
                self.assertGreaterEqual(len(result), 0)

    def test_data_integrity_and_type_validation(self):
        """Test data integrity and type validation."""
        # Verify all segments have correct data types
        for seg in self.provider.diarization_data:
            self.assertIsInstance(seg.start_timestamp_ns, int)
            self.assertIsInstance(seg.end_timestamp_ns, int)
            self.assertIsInstance(seg.speaker, str)
            self.assertIsInstance(seg.content, str)

            # Verify temporal consistency (start < end)
            self.assertLess(seg.start_timestamp_ns, seg.end_timestamp_ns)

            # Verify non-empty content
            self.assertTrue(len(seg.speaker.strip()) > 0)
            self.assertTrue(len(seg.content.strip()) > 0)

    def test_unsorted_data_handling(self):
        """Test handling of unsorted input data."""
        # Create unsorted CSV data
        unsorted_csv = """start_timestamp_ns,end_timestamp_ns,speaker,content
4000000000,5000000000,Speaker1,Third segment
1000000000,2000000000,Speaker2,First segment
3000000000,4000000000,Speaker1,Second segment
"""

        # Create temp file with unsorted data
        unsorted_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        )
        unsorted_file.write(unsorted_csv)
        unsorted_file.close()

        try:
            # Create provider with unsorted data
            unsorted_provider = DiarizationDataProvider(unsorted_file.name)

            # Verify data is sorted after loading
            start_times = [
                seg.start_timestamp_ns for seg in unsorted_provider.diarization_data
            ]
            self.assertEqual(start_times, sorted(start_times))

            # Verify functionality still works
            result = unsorted_provider.get_diarization_data_by_timestamp_ns(1500000000)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].content, "First segment")

        finally:
            import os

            os.unlink(unsorted_file.name)

    def test_empty_file_handling(self):
        """Test handling of empty CSV file."""
        # Create empty CSV (header only)
        empty_csv = ""

        empty_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        empty_file.write(empty_csv)
        empty_file.close()

        try:
            # Should throw RuntimeError for empty file via check_valid_csv utility
            with self.assertRaises(RuntimeError) as context:
                DiarizationDataProvider(empty_file.name)

            # Verify error message mentions empty file
            error_msg = str(context.exception).lower()
            self.assertTrue(
                "empty" in error_msg,
                f"Expected empty file error, got: {context.exception}",
            )

        finally:
            import os

            os.unlink(empty_file.name)

    def test_missing_file_handling(self):
        """Test handling of missing file."""
        # Should throw RuntimeError for missing file via check_valid_csv utility
        with self.assertRaises(RuntimeError) as context:
            DiarizationDataProvider("/nonexistent/path.csv")

        # Verify error message mentions file not found or similar
        error_msg = str(context.exception).lower()
        self.assertTrue(
            "not found" in error_msg
            or "does not exist" in error_msg
            or "no such file" in error_msg
            or "file does not exist" in error_msg,
            f"Expected file not found error, got: {context.exception}",
        )

    def test_malformed_csv_handling(self):
        """Test handling of malformed CSV data."""
        # Create CSV with missing required columns (missing end_timestamp_ns)
        malformed_csv = """start_timestamp_ns,speaker,content
1000000000,Speaker1,Missing end timestamp
"""

        malformed_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        )
        malformed_file.write(malformed_csv)
        malformed_file.close()

        try:
            # Should raise RuntimeError for malformed headers via check_valid_csv utility
            with self.assertRaises(RuntimeError) as context:
                DiarizationDataProvider(malformed_file.name)

            # Verify error message mentions header mismatch or CSV headers
            error_msg = str(context.exception).lower()
            self.assertTrue(
                "headers don't match" in error_msg
                or "csv headers" in error_msg
                or "expected" in error_msg,
                f"Expected CSV header validation error, got: {context.exception}",
            )

        finally:
            import os

            os.unlink(malformed_file.name)

    def test_invalid_data_types_handling(self):
        """Test handling of invalid data types in CSV."""
        # Create CSV with invalid timestamp format
        invalid_csv = """start_timestamp_ns,end_timestamp_ns,speaker,content
not_a_number,2000000000,Speaker1,Invalid start timestamp
1000000000,also_not_number,Speaker2,Invalid end timestamp
"""

        invalid_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        )
        invalid_file.write(invalid_csv)
        invalid_file.close()

        try:
            # Should raise RuntimeError for invalid data types
            with self.assertRaises(RuntimeError):
                DiarizationDataProvider(invalid_file.name)

        finally:
            import os

            os.unlink(invalid_file.name)

    def test_large_dataset_performance(self):
        """Test performance with larger dataset."""
        # Generate larger test dataset
        large_csv_lines = ["start_timestamp_ns,end_timestamp_ns,speaker,content"]

        # Create 100 segments with various overlaps
        for i in range(100):
            start_ts = i * 1000000000  # 1 second intervals
            end_ts = start_ts + 1500000000  # 1.5 second duration (creates overlaps)
            speaker = f"Speaker{i % 5}"  # 5 different speakers
            content = f"Content segment {i}"
            large_csv_lines.append(f"{start_ts},{end_ts},{speaker},{content}")

        large_csv = "\n".join(large_csv_lines)

        large_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        large_file.write(large_csv)
        large_file.close()

        try:
            # Create provider with large dataset
            large_provider = DiarizationDataProvider(large_file.name)

            # Verify data loaded correctly
            self.assertEqual(len(large_provider.diarization_data), 100)

            # Test timestamp search performance (should be fast with binary search)
            import time

            start_time = time.time()

            # Perform multiple searches
            for i in range(0, 50):
                timestamp = i * 1000000000 + 500000000  # Middle of segments
                result = large_provider.get_diarization_data_by_timestamp_ns(timestamp)
                self.assertGreater(len(result), 0)  # Should find overlapping segments

            elapsed_time = time.time() - start_time

            # Should complete quickly (less than 0.1 seconds for 50 searches)
            self.assertLess(elapsed_time, 0.1, "Binary search performance is too slow")

            # Test range search performance
            start_time = time.time()

            # Perform range searches
            for i in range(0, 10):
                start_ts = i * 10000000000
                end_ts = start_ts + 5000000000
                result = (
                    large_provider.get_diarization_data_by_start_and_end_timestamps(
                        start_ts, end_ts
                    )
                )
                # Results should be reasonable
                self.assertLessEqual(len(result), 100)

            elapsed_time = time.time() - start_time
            self.assertLess(elapsed_time, 0.1, "Range search performance is too slow")

        finally:
            import os

            os.unlink(large_file.name)

    def test_complex_overlapping_scenarios(self):
        """Test complex overlapping scenarios that stress the binary search logic."""
        # Create data with multiple overlapping segments at the same time
        complex_csv = """start_timestamp_ns,end_timestamp_ns,speaker,content
1000000000,3000000000,Speaker1,Long segment 1
1500000000,2500000000,Speaker2,Nested segment
2000000000,4000000000,Speaker3,Overlapping segment
2000000000,2500000000,Speaker4,Same start time as Speaker3
2500000000,2500000000,Speaker5,Zero duration segment"""

        complex_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        )
        complex_file.write(complex_csv)
        complex_file.close()

        try:
            complex_provider = DiarizationDataProvider(complex_file.name)

            # Test timestamp at maximum overlap point (2250000000)
            result = complex_provider.get_diarization_data_by_timestamp_ns(2250000000)
            self.assertGreaterEqual(
                len(result), 3
            )  # At least 3 speakers should overlap

            # Test exact boundary conditions
            result = complex_provider.get_diarization_data_by_timestamp_ns(2000000000)
            speakers = [seg.speaker for seg in result]
            self.assertIn("Speaker1", speakers)  # Should include Speaker1 (ongoing)
            self.assertIn("Speaker3", speakers)  # Should include Speaker3 (starting)

            # Test zero-duration segment handling
            result = complex_provider.get_diarization_data_by_timestamp_ns(2500000000)
            speakers = [seg.speaker for seg in result]
            # Should handle zero-duration segment appropriately

        finally:
            import os

            os.unlink(complex_file.name)
