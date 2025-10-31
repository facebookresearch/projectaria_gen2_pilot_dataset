# Aria Gen2 Pilot Dataset

A Python package for loading, processing, and visualizing data from the Aria Gen2 Pilot Dataset. This dataset contains multimodal sensor data from Project Aria Gen2 glasses, including raw sensor streams, real-time machine perception outputs, and post-processed algorithm results.

## Features

- **VRS Data Loading**: Access raw sensor data (RGB/SLAM cameras, IMU, audio, GPS, etc.)
- **Machine Perception Services (MPS)**: Load SLAM trajectories and hand tracking results
- **Algorithm Outputs**: Process heart rate monitoring, diarization, hand-object interaction, egocentric voxel lifting, and stereo depth data
- **Visualization Tools**: Built-in viewer with Rerun integration for 3D visualization
- **Unified API**: Single `AriaGen2PilotDataProvider` interface for all data types

## Installation

```bash
# Deactivate conda if active
conda deactivate

# Remove existing environment if it exists
rm -rf ~/projectaria_gen2_python_env

# Create new Python virtual environment
python3.12 -m venv ~/projectaria_gen2_python_env

# Activate the environment
source ~/projectaria_gen2_python_env/bin/activate

# Upgrade pip
python3 -m pip install --upgrade pip

# Install the package with all dependencies
python3 -m pip install projectaria-gen2-pilot-dataset'[all]'
```
## Wiki

Please see the full documentation on the [Aria Gen2 Pilot Dataset wiki page](https://facebookresearch.github.io/projectaria_tools/gen2/research-tools/dataset/pilot/content).

## Tutorials

Get started with these comprehensive tutorials:
- [VRS Data Loading](examples/tutorial_1_vrs_data_loading.ipynb) - Loading raw sensor data
- [MPS Data Loading](examples/tutorial_2_mps_data_loading.ipynb) - Loading Machine Perception Services data
- [Algorithm Data Loading](examples/tutorial_3_algorithm_data_loading.ipynb) - Loading algorithm output data
- [Multiple Sequences Timestamp Alignment](examples/tutorial_4_multi_sequences_timestamp_alignment.ipynb) - Synchronizing data between multiple devices

## Dataset Structure

Each sequence contains:
- **Raw VRS files**: Sensor streams from Aria Gen2 devices
- **MPS outputs**: SLAM trajectories and hand tracking results
- **Algorithm data**: Heart rate, diarization, hand-object interaction, depth estimation, and 3D scene reconstruction

## Visualization

Launch the interactive viewer:
```bash
aria_gen2_pilot_dataset_viewer --sequence-path /path/to/sequence
```

## License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0). See the [LICENSE](LICENSE) file for details.
