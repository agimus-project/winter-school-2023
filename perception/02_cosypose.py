#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2023-12-10
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#
# The purpose of this example is to detect the object of interest on the input image
# and to estimate its 6D pose.
# We will use objects from YCBV dataset, namely red-cup with object ID: 14 and
# mustard with object ID: 05.
#
# To use this example, you need to specify environment variable HAPPYPOSE_DATA_DIR to
# point to your data.
#
import cv2
import numpy as np

from happypose.toolbox.inference.types import ObservationTensor

from perception.aws_cosypose import AWSCosyPose
from perception.loading_utils import load_rgb_images_for_scene, load_camera_data_color
from perception.render_utils import render_overlay, draw_bounding_boxes

if __name__ == "__main__":
    camera_data = load_camera_data_color()
    image = next(load_rgb_images_for_scene(scene_id=1))

    # HappyPose works with observation tensor that combines RGB with camera matrix
    observation = ObservationTensor.from_numpy(rgb=image, K=camera_data.K)

    # Load and use the detector
    detector = AWSCosyPose.load_detector()
    detections = detector.get_detections(observation=observation, detection_th=0.9)

    # filterdetection based on object id interested in, i.e. object_id = 14, the cup
    detections = detections[np.where(detections.infos["label"] == "ycbv-obj_000014")]

    pose_estimator = AWSCosyPose.load_pose_estimator()

    preds, preds_extra = pose_estimator.run_inference_pipeline(
        detections=detections,
        observation=observation,
        run_detector=False,
        n_coarse_iterations=1,
        n_refiner_iterations=1,
    )
    renderer = pose_estimator.refiner_model.renderer
    output_image = render_overlay(image, renderer, predictions=preds)
    output_image = draw_bounding_boxes(output_image, detections)

    # todo: visualize all predictions, after coarse, after 1 refiner, ...

    cv2.imshow("Poses overlay", cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
    cv2.waitKey()
    cv2.destroyWindow("Poses overlay")
