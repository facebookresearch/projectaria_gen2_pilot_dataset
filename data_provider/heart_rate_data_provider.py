# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import csv
import logging
from typing import List, Optional, Tuple

from projectaria_tools.core.sensor_data import TimeQueryOptions

from .aria_gen2_pilot_dataset_data_types import HeartRateData
from .utils import check_valid_csv, find_data_by_timestamp_ns


class HeartRateDataProvider:
    """Heart rate data provider for Aria Gen2 Pilot Dataset."""

    def __init__(self, data_path: str):
        """Initialize with path to heart_rate_results.csv."""
        self.data_path = data_path
        self.heart_rate_data_list: List[Tuple[int, HeartRateData]] = []

        # Set up logger
        self.logger = logging.getLogger(self.__class__.__name__)

        self._load_data()

    def _load_data(self) -> None:
        """Load heart rate data from CSV file."""
        check_valid_csv(self.data_path, "timestamp_ns,heart_rate_bpm")

        try:
            self.heart_rate_data_list = []
            with open(self.data_path, "r") as csvfile:
                reader = csv.DictReader(csvfile)
                rows = list(reader)
                for row in rows:
                    timestamp_ns = int(row["timestamp_ns"])
                    heart_rate_bpm = int(row["heart_rate_bpm"])

                    heart_rate_data = HeartRateData(timestamp_ns, heart_rate_bpm)
                    self.heart_rate_data_list.append((timestamp_ns, heart_rate_data))

            # Ensure data is strictly increasing by timestamp for bisect operations
            if self.heart_rate_data_list and not all(
                self.heart_rate_data_list[i][0] < self.heart_rate_data_list[i + 1][0]
                for i in range(len(self.heart_rate_data_list) - 1)
            ):
                self.heart_rate_data_list.sort(key=lambda x: x[0])
                # Check for duplicate timestamps
                duplicates = [
                    self.heart_rate_data_list[i][0]
                    for i in range(len(self.heart_rate_data_list) - 1)
                    if self.heart_rate_data_list[i][0]
                    == self.heart_rate_data_list[i + 1][0]
                ]
                if duplicates:
                    raise ValueError(
                        f"Duplicate timestamp(s) found in heart rate data: {duplicates}"
                    )

        except (FileNotFoundError, KeyError, ValueError) as e:
            raise RuntimeError(
                f"Failed to load heart rate data from {self.data_path}: {e}"
            )

    def get_heart_rate_by_index(self, index: int) -> Optional[HeartRateData]:
        """Get heart rate data by index."""
        if 0 <= index < len(self.heart_rate_data_list):
            return self.heart_rate_data_list[index][1]
        else:
            self.logger.warning(
                "Index %d is out of range (0 to %d). Return None.",
                index,
                len(self.heart_rate_data_list) - 1,
            )
        return None

    def get_heart_rate_by_timestamp_ns(
        self,
        timestamp_ns: int,
        time_query_options: TimeQueryOptions = TimeQueryOptions.CLOSEST,
    ) -> Optional[HeartRateData]:
        """Get heart rate data at specified timestamp."""
        return find_data_by_timestamp_ns(
            self.heart_rate_data_list, timestamp_ns, time_query_options
        )

    def get_heart_rate_timestamps_ns(self) -> List[int]:
        """Get all heart rate timestamps."""
        return [timestamp for timestamp, _ in self.heart_rate_data_list]

    def get_heart_rate_total_number(self) -> int:
        """Get total number of heart rate entries."""
        return len(self.heart_rate_data_list)

    def get_heart_rate_all_data(self) -> List[HeartRateData]:
        """Get all heart rate data."""
        return [
            heart_rate_data.copy() for _, heart_rate_data in self.heart_rate_data_list
        ]
