# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import argparse
import logging

from aria_gen2_pilot_dataset import AriaGen2PilotDataProvider

from .aria_gen2_pilot_data_visualizer import AriaGen2PilotDataVisualizer
from .aria_gen2_pilot_viewer_config import AriaGen2PilotViewerConfig


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels"""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Orange/Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset
    }

    def format(self, record):
        # Get color for the log level
        color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
        # Apply color to the level name
        original_levelname = record.levelname
        record.levelname = f"{color}{record.levelname}{self.COLORS['RESET']}"

        # Format the message with the parent formatter
        formatted_message = super().format(record)

        # Restore the original level name for potential other handlers
        record.levelname = original_levelname

        return formatted_message


def setup_logging_format() -> None:
    # Configure root logger first
    handler = logging.StreamHandler()

    # Use colored formatter
    colored_formatter = ColoredFormatter(
        fmt="%(name)-20s - %(levelname)-8s - %(message)s",
    )
    handler.setFormatter(colored_formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sequence-path", type=str, required=True, help="path to sequence folder"
    )
    parser.add_argument(
        "--rrd-output-path",
        type=str,
        default="",
        help="path to save .rrd file (if not provided, will spawn viewer window)",
    )
    return parser.parse_args()


def main():
    setup_logging_format()
    # Set up named logger for the main application
    logger = logging.getLogger("AriaGen2PilotVisualization")

    args = parse_args()

    logger.info("Starting Aria Gen2 Pilot Dataset Visualization")

    # Load data provider directly from sequence path
    data_provider = AriaGen2PilotDataProvider(args.sequence_path)

    # Create and initialize visualizer
    config = AriaGen2PilotViewerConfig()
    visualizer = AriaGen2PilotDataVisualizer(data_provider, config)

    logger.info("Initializing visualization...")
    visualizer.initialize_rerun_and_blueprint(args.rrd_output_path)
    visualizer.plot_sequence()

    logger.info("Aria Gen2 Pilot Dataset Visualization finished successfully.")


if __name__ == "__main__":
    main()
