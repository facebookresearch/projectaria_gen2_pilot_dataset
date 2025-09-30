# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Unit tests for utils.py module."""

import unittest

from aria_gen2_pilot_dataset.data_provider.utils import (
    find_timestamp_index_by_time_query_option,
)

from projectaria_tools.core.sensor_data import TimeQueryOptions


class TestFindTimestampIndexByTimeQueryOption(unittest.TestCase):
    """Test cases for find_timestamp_index_by_time_query_option function."""

    def test_all_scenarios(self) -> None:
        """Test all scenarios using data arrays."""
        # Test data format: (timestamps, target_timestamp, option, expected_result, description)
        # Each test case verified by manual calculation based on bisect logic
        test_cases = [
            # Empty list tests - function returns -1 for empty lists
            ([], 300, TimeQueryOptions.BEFORE, -1, "empty_before"),
            ([], 300, TimeQueryOptions.AFTER, -1, "empty_after"),
            ([], 300, TimeQueryOptions.CLOSEST, -1, "empty_closest"),
            # BEFORE option tests - finds last timestamp <= target
            # timestamps: [100, 200, 300, 400, 500] (indices 0, 1, 2, 3, 4)
            (
                [100, 200, 300, 400, 500],
                300,
                TimeQueryOptions.BEFORE,
                2,
                "before_exact_match",
            ),  # 300 at index 2
            (
                [100, 200, 300, 400, 500],
                250,
                TimeQueryOptions.BEFORE,
                1,
                "before_between_values",
            ),  # last <= 250 is 200 at index 1
            (
                [100, 200, 300, 400, 500],
                50,
                TimeQueryOptions.BEFORE,
                -1,
                "before_too_small",
            ),  # no timestamp <= 50
            (
                [100, 200, 300, 400, 500],
                600,
                TimeQueryOptions.BEFORE,
                4,
                "before_after_all",
            ),  # last <= 600 is 500 at index 4
            # AFTER option tests - finds first timestamp >= target
            (
                [100, 200, 300, 400, 500],
                300,
                TimeQueryOptions.AFTER,
                2,
                "after_exact_match",
            ),  # 300 at index 2
            (
                [100, 200, 300, 400, 500],
                250,
                TimeQueryOptions.AFTER,
                2,
                "after_between_values",
            ),  # first >= 250 is 300 at index 2
            (
                [100, 200, 300, 400, 500],
                50,
                TimeQueryOptions.AFTER,
                0,
                "after_before_all",
            ),  # first >= 50 is 100 at index 0
            (
                [100, 200, 300, 400, 500],
                600,
                TimeQueryOptions.AFTER,
                -1,
                "after_too_large",
            ),  # no timestamp >= 600
            # CLOSEST option tests - finds closest timestamp, prefer earlier on ties
            (
                [100, 200, 300, 400, 500],
                300,
                TimeQueryOptions.CLOSEST,
                2,
                "closest_exact_match",
            ),  # exact match at index 2
            (
                [100, 200, 300, 400, 500],
                220,
                TimeQueryOptions.CLOSEST,
                1,
                "closest_to_200",
            ),  # |220-200|=20 < |220-300|=80
            (
                [100, 200, 300, 400, 500],
                280,
                TimeQueryOptions.CLOSEST,
                2,
                "closest_to_300",
            ),  # |280-300|=20 < |280-200|=80
            (
                [100, 200, 300, 400, 500],
                250,
                TimeQueryOptions.CLOSEST,
                1,
                "closest_equidistant",
            ),  # |250-200|=50 = |250-300|=50, prefer earlier
            (
                [100, 200, 300, 400, 500],
                50,
                TimeQueryOptions.CLOSEST,
                0,
                "closest_before_all",
            ),  # closest is 100 at index 0
            (
                [100, 200, 300, 400, 500],
                600,
                TimeQueryOptions.CLOSEST,
                4,
                "closest_after_all",
            ),  # closest is 500 at index 4
            # Edge cases - single element
            (
                [300],
                300,
                TimeQueryOptions.BEFORE,
                0,
                "single_exact_before",
            ),  # 300 <= 300, return index 0
            (
                [300],
                200,
                TimeQueryOptions.BEFORE,
                -1,
                "single_too_small_before",
            ),  # no timestamp <= 200
            (
                [300],
                400,
                TimeQueryOptions.AFTER,
                -1,
                "single_too_large_after",
            ),  # no timestamp >= 400
            (
                [300],
                400,
                TimeQueryOptions.CLOSEST,
                0,
                "single_closest",
            ),  # only option is index 0
            # Duplicates - test bisect behavior with repeated values
            # timestamps: [100, 200, 200, 300] (indices 0, 1, 2, 3)
            (
                [100, 200, 200, 300],
                200,
                TimeQueryOptions.BEFORE,
                2,
                "duplicate_before",
            ),  # bisect_right finds rightmost position
            (
                [100, 200, 200, 300],
                200,
                TimeQueryOptions.AFTER,
                1,
                "duplicate_after",
            ),  # bisect_left finds leftmost position
        ]

        for timestamps, target, option, expected, description in test_cases:
            with self.subTest(case=description):
                result = find_timestamp_index_by_time_query_option(
                    timestamps, target, option
                )
                self.assertEqual(
                    result,
                    expected,
                    f"Failed for {description}: expected {expected}, got {result}",
                )
