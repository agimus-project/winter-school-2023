#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2023-12-10
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from happypose.toolbox.datasets.scene_dataset import CameraData


def get_perception_data_dir(data_dir: Path | None = None) -> Path:
    """Load perception data dir, check it exists and has the data in it. If data_dir
    not specified, assume default location."""
    if data_dir is None:
        data_dir = Path(__file__).parent / "data"
    path_camera = data_dir / "cam_d435_640_happypose.json"
    scene1_dir = data_dir /"scene1_obj_14"
    # Handle the case where aws_tracker_videos fodler was copied whole into data
    if not path_camera.exists():
        data_dir = data_dir / "aws_tracker_videos"
        path_camera = data_dir / "cam_d435_640_happypose.json"
        scene1_dir = data_dir / "scene1_obj_14"
    assert (
        path_camera.exists() and scene1_dir.exists()
    ), f"{path_camera.as_posix()} and {scene1_dir.as_posix()} not found. See README.md for data download."
    return data_dir


def load_camera_data_color(data_dir: Path | None = None) -> CameraData:
    """Load json camera data from the data file."""
    data_dir = get_perception_data_dir(data_dir)
    path_camera = data_dir / "cam_d435_640_happypose.json"
    return CameraData.from_json(path_camera.read_text())


def load_rgb_images_for_scene(scene_id: int, data_dir: Path | None = None):
    """Get iterator that yields images for the given scene id."""
    data_dir = get_perception_data_dir(data_dir)
    scene_name = f"scene{scene_id}"
    if scene_id in [1, 2]:
        scene_name += "_obj_14"
    elif scene_id in [3, 4]:
        scene_name += "_obj_05"
    scene_path = data_dir / scene_name
    for path_image in sorted(scene_path.glob("color_*")):
        img_bgr = cv2.imread(str(path_image))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        rgb = np.array(img_rgb, dtype=np.uint8)
        yield rgb
