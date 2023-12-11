#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2023-12-11
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
from happypose.pose_estimators.megapose.config import LOCAL_DATA_DIR
from happypose.toolbox.datasets.bop_object_datasets import BOPObjectDataset

from happypose.toolbox.inference.types import ObservationTensor
from happypose.toolbox.utils.load_model import NAMED_MODELS, load_named_model

from perception.aws_cosypose import AWSCosyPose
from perception.loading_utils import load_rgb_images_for_scene, load_camera_data_color
from perception.render_utils import render_overlay, draw_bounding_boxes

if __name__ == "__main__":
    camera_data = load_camera_data_color()
    image = next(load_rgb_images_for_scene(scene_id=3))

    # HappyPose works with observation tensor that combines RGB with camera matrix
    observation = ObservationTensor.from_numpy(rgb=image, K=camera_data.K)

    # Load and use the detector
    detector = AWSCosyPose.load_detector()
    detections = detector.get_detections(observation=observation, detection_th=0.0)
    detections = detections[np.where(detections.infos["label"] == "ycbv-obj_000005")]

    # Load MegaPose
    object_dataset = BOPObjectDataset(
        LOCAL_DATA_DIR / "bop_datasets/ycbv/models_bop-compat",
        label_format="ycbv-{label}",
    )
    model_info = NAMED_MODELS["megapose-1.0-RGB"]
    pose_estimator = load_named_model(
        "megapose-1.0-RGB", object_dataset, n_workers=1, bsz_images=1
    ).to("cpu")
    pose_estimator._SO3_grid = pose_estimator._SO3_grid[::10]  # let's speed up
    preds, preds_extra = pose_estimator.run_inference_pipeline(
        observation,
        detections=detections,
        **model_info["inference_parameters"],
    )

    renderer = pose_estimator.refiner_model.renderer
    output_image = render_overlay(image, renderer, predictions=preds)
    output_image = draw_bounding_boxes(output_image, detections)

    # cv2.imwrite("/tmp/megapose.png", cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
    cv2.imshow("Poses overlay", cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
    cv2.waitKey()
    cv2.destroyWindow("Poses overlay")
