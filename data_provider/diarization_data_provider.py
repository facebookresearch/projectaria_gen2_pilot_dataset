# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import bisect
import csv
from typing import List

from .aria_gen2_pilot_dataset_data_types import DiarizationData
from .utils import check_valid_csv


class DiarizationDataProvider:
    """Diarization data provider for Aria Gen2 Pilot Dataset."""

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.diarization_data: List[DiarizationData] = []
        self.start_timestamp_ns_list: List[int] = []
        self._load_data()

    def _load_data(self) -> None:
        check_valid_csv(
            self.data_path, "start_timestamp_ns,end_timestamp_ns,speaker,content"
        )
        try:
            with open(self.data_path, "r") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    self.diarization_data.append(
                        DiarizationData(
                            int(row["start_timestamp_ns"]),
                            int(row["end_timestamp_ns"]),
                            str(row["speaker"]),
                            str(row["content"]),
                        )
                    )

            # Only sort if not already sorted
            if any(
                self.diarization_data[i].start_timestamp_ns
                > self.diarization_data[i + 1].start_timestamp_ns
                for i in range(len(self.diarization_data) - 1)
            ):
                self.diarization_data.sort(key=lambda x: x.start_timestamp_ns)

            self.start_timestamp_ns_list = [
                data.start_timestamp_ns for data in self.diarization_data
            ]

        except (FileNotFoundError, KeyError, ValueError) as e:
            raise RuntimeError(
                f"Failed to load diarization data from {self.data_path}: {e}"
            )
        if not self.start_timestamp_ns_list:
            raise RuntimeError(
                "No diarization data found, can not initialize DiarizationDataProvider."
            )

    def get_diarization_data_by_index(self, index: int) -> DiarizationData:
        """Get utterance by index."""
        if index < 0 or index >= len(self.diarization_data):
            raise IndexError(f"Index {index} out of range.")
        return self.diarization_data[index]

    def get_diarization_data_by_timestamp_ns(
        self, timestamp_ns: int
    ) -> List[DiarizationData]:
        """Get utterances spanning timestamp.
        Specifically, this function will find all utterances that satisfy: data.start_timestamp <= query_timestamp and data.end_timestamp >= query_timestamp.

        Returns:
            List of DiarizationData sorted by start timestamp.
        """
        if not self.diarization_data:
            return []

        # Find segments with start <= timestamp
        right = bisect.bisect_right(self.start_timestamp_ns_list, timestamp_ns)

        result = []
        for i in range(right):
            data = self.diarization_data[i]
            if timestamp_ns <= data.end_timestamp_ns:
                result.append(data)
        return result

    def get_diarization_data_by_start_and_end_timestamps(
        self, start_timestamp_ns: int, end_timestamp_ns: int
    ) -> List[DiarizationData]:
        """Retrieve all utterances that overlap with the specified time interval.
        This function returns all utterances that satisfies ANY of the following conditions:
        1. `data.start_timestamp <= query_start_timestamp`.
        2. `data.end_timestamp >= query_end_timestamp`.

        Returns:
            List of DiarizationData sorted by start timestamp.
        """
        if not self.diarization_data:
            return []

        # Binary search: find first segment that could overlap (segment_start <= query_end)
        start_idx = bisect.bisect_right(self.start_timestamp_ns_list, end_timestamp_ns)

        result = []
        for i in range(start_idx):
            data = self.diarization_data[i]
            # Check full overlap
            if start_timestamp_ns <= data.end_timestamp_ns:
                result.append(data)
        return result

    def get_diarization_data_total_number(self) -> int:
        return len(self.diarization_data)
