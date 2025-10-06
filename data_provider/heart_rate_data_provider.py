# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import csv
import logging
from typing import List, Optional

from projectaria_tools.core.sensor_data import TimeQueryOptions

from .aria_gen2_pilot_dataset_data_types import HeartRateData
from .utils import check_valid_csv, find_timestamp_index_by_time_query_option


class HeartRateDataProvider:
    """Heart rate data provider for Aria Gen2 Pilot Dataset."""

    def __init__(self, data_path: str):
        """Initialize with path to heart_rate_results.csv."""
        self.data_path = data_path
        self.heart_rate_data_list: List[HeartRateData] = []
        self.timestamps_ns: List[int] = []

        # Set up logger
        self.logger = logging.getLogger(self.__class__.__name__)

        self._load_data()

    def _load_data(self) -> None:
        """Load heart rate data from CSV file."""
        check_valid_csv(self.data_path, "timestamp_ns,heart_rate_bpm")

        try:
            with open(self.data_path, "r") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    timestamp_ns = int(row["timestamp_ns"])
                    heart_rate_bpm = int(row["heart_rate_bpm"])

                    heart_rate_data = HeartRateData(timestamp_ns, heart_rate_bpm)
                    self.heart_rate_data_list.append(heart_rate_data)
                    self.timestamps_ns.append(timestamp_ns)

            # Ensure data is sorted by timestamp for efficient querying
            if self.timestamps_ns and not all(
                self.timestamps_ns[i] < self.timestamps_ns[i + 1]
                for i in range(len(self.timestamps_ns) - 1)
            ):
                # Sort all data by timestamp
                sorted_data = sorted(
                    zip(self.timestamps_ns, self.heart_rate_data_list),
                    key=lambda x: x[0],
                )
                self.timestamps_ns, self.heart_rate_data_list = zip(*sorted_data)
                self.timestamps_ns = list(self.timestamps_ns)
                self.heart_rate_data_list = list(self.heart_rate_data_list)

                # Check for duplicate timestamps
                duplicates = [
                    self.timestamps_ns[i]
                    for i in range(len(self.timestamps_ns) - 1)
                    if self.timestamps_ns[i] == self.timestamps_ns[i + 1]
                ]
                if duplicates:
                    raise ValueError(
                        f"Duplicate timestamp(s) found in heart rate data: {duplicates}"
                    )

        except Exception as e:
            raise RuntimeError(
                f"Failed to load heart rate data from {self.data_path}: {e}"
            )
        if not self.heart_rate_data_list:
            raise RuntimeError(
                "No heart rate data found, can not initialize HeartRateDataProvider."
            )

    def get_heart_rate_by_index(self, index: int) -> Optional[HeartRateData]:
        """Get heart rate data by index."""
        if 0 <= index < len(self.heart_rate_data_list):
            return self.heart_rate_data_list[index]
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
        index = find_timestamp_index_by_time_query_option(
            self.timestamps_ns, timestamp_ns, time_query_options
        )
        return self.get_heart_rate_by_index(index)

    def get_heart_rate_total_number(self) -> int:
        """Get total number of heart rate entries."""
        return len(self.heart_rate_data_list)
