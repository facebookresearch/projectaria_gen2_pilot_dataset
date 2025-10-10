# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import os

from setuptools import setup

# Read the contents of your README file if it exists
this_directory = os.path.abspath(os.path.dirname(__file__))
readme_path = os.path.join(this_directory, "README.md")
if os.path.exists(readme_path):
    with open(readme_path, encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "Aria Gen2 Pilot Dataset - Data provider and visualization tools for Aria Gen2 pilot dataset sequences"

setup(
    name="projectaria-gen2-pilot-dataset",
    author="Meta Reality Labs Research",
    version="1.0.0",
    license="CC BY-NC 4.0",
    description="Aria Gen2 Pilot Dataset",
    long_description="Data provider and visualization tools for Aria Gen2 pilot dataset sequences",
    packages=[
        "aria_gen2_pilot_dataset",
        "aria_gen2_pilot_dataset.data_provider",
        "aria_gen2_pilot_dataset.visualization",
    ],
    package_dir={"aria_gen2_pilot_dataset": "."},
    url="https://github.com/facebookresearch/projectaria_gen2_pilot_dataset",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Multimedia :: Video",
    ],
    python_requires=">=3.8",
    install_requires=["projectaria-tools==2.0.0a5", "pillow"],
    extras_require={"all": ["jupyter", "rerun-notebook==0.22.1"]},
    entry_points={
        "console_scripts": [
            "aria_gen2_pilot_dataset_viewer=aria_gen2_pilot_dataset.visualization.aria_gen2_pilot_dataset_viewer:main",
        ],
    },
    zip_safe=False,
)
