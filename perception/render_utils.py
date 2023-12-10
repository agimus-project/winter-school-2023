#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2023-12-10
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#
import numpy as np
import cv2
from happypose.toolbox.renderer import Panda3dLightData
from happypose.toolbox.visualization.utils import make_contour_overlay


def render_overlay(img: np.ndarray, renderer, predictions) -> np.ndarray:
    """Render overlay"""

    light_datas = [[Panda3dLightData(light_type="ambient", color=(1.0, 1.0, 1.0, 1))]]

    # todo overlay for all detections
    renderings = renderer.render(
        labels=predictions.infos.label[:1],
        TCO=predictions.poses[:1],
        K=predictions.K[:1],
        resolution=img.shape[:2],
        light_datas=light_datas,
    )
    contour_overlay = make_contour_overlay(
        img,
        renderings.rgbs[0].numpy().transpose(1, 2, 0),
        dilate_iterations=1,
        color=(0, 255, 0),
    )["img"]
    return contour_overlay


def draw_bounding_boxes(img: np.ndarray, detections) -> np.ndarray:
    """Draw bounding boxes based on the detections."""
    output_image = img.copy()
    for bb, confidence in zip(detections.bboxes, detections.infos["score"]):
        xmin, ymin, xmax, ymax = [int(v) for v in bb]
        cv2.rectangle(
            output_image, [xmin, ymin], [xmax, ymax], color=[255, 0, 0], thickness=2
        )
    return output_image
