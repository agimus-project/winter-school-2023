#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2023-12-10
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#
# The purpose of this example is to detect the object of interest on the input image.
# We will use objects from YCBV dataset, namely red-cup with object ID: 14 and
# mustard with object ID: 05.
#
# To use this example, you need to specify environment variable HAPPYPOSE_DATA_DIR to
# point to your data.
#
import cv2
from happypose.toolbox.inference.types import ObservationTensor

from perception.aws_cosypose import AWSCosyPose
from perception.loading_utils import load_rgb_images_for_scene, load_camera_data_color

if __name__ == "__main__":
    camera_data = load_camera_data_color()
    image = next(load_rgb_images_for_scene(scene_id=1))

    # Let's show the image via opencv
    cv2.imshow("Input image", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.waitKey()
    cv2.destroyWindow("Input image")

    # HappyPose works with observation tensor that combines RGB with camera matrix
    observation = ObservationTensor.from_numpy(rgb=image, K=camera_data.K)

    # Load and use the detector
    detector = AWSCosyPose.load_detector()
    detections = detector.get_detections(observation=observation, detection_th=0.0)

    output_image = image.copy()
    # Draw all detection bounding boxes
    for bb, confidence in zip(detections.bboxes, detections.infos["score"]):
        # todo: filter the detections based on the object id, i.e. use prior information
        xmin, ymin, xmax, ymax = [int(v) for v in bb]
        cv2.rectangle(
            output_image, [xmin, ymin], [xmax, ymax], color=[255, 0, 0], thickness=2
        )

    # cv2.imwrite("/tmp/detections.png", cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
    cv2.imshow("Detections", cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
    cv2.waitKey()
    cv2.destroyWindow("Detections")
