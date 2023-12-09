import cv2
import numpy as np
import quaternion
from pathlib import Path
import argparse

import pym3t


def inv_SE3(T: np.ndarray):
    """
    Inverse of an SE(3) 4x4 array.
    """
    Tinv = np.eye(4)
    Tinv[:3,:3] = T[:3,:3].T
    Tinv[:3,3] = -T[:3,:3].T@T[:3,3]
    return Tinv


def tq_to_SE3(t, q):
    """
    :param t: translation as list or array
    :param q: quaternion as list or array, expected order: xyzw
    :return: 4x4 array representing the SE(3) transformation
    """
    T = np.eye(4)
    T[:3,3] = t
    # np.quaternion constructor uses wxyz convention
    quat = np.quaternion(q[3], q[0], q[1], q[2]).normalized()
    T[:3,:3] = quaternion.as_rotation_matrix(quat)
    return T


def setup_single_object_tracker(args: argparse.Namespace, cam_intrinsics: dict):
    """
    Setup and example pym3t object tracker.

    Handles a one camera, one object tracking and is only meant for 

    :param args: cli arguments from argparse
        should have at least these attributes:
        body_name
        models_dir
        fov
        scale_geometry
        tmp_dir
        use_region
        use_texture
    :param cam_intrinsics: dict containing camera intrinsics
    """

    # Missing arguments
    if 'use_depth' not in args:
        args.use_depth = False
    if 'use_depth_viewer' not in args:
        args.use_depth_viewer = False
    if 'measure_occlusions' not in args:
        args.measure_occlusions = False
    
    tmp_dir = Path(args.tmp_dir)
    tmp_dir.mkdir(exist_ok=True)

    # synchronize_cameras: to be able to print elapsed time
    tracker = pym3t.Tracker('tracker', synchronize_cameras=False)
    renderer_geometry = pym3t.RendererGeometry('renderer geometry')

    # Setup camera(s)
    color_camera = pym3t.DummyColorCamera('cam_color')
    color_camera.color2depth_pose = tq_to_SE3(cam_intrinsics['trans_d_c'], cam_intrinsics['quat_d_c_xyzw'])
    color_camera.intrinsics = pym3t.Intrinsics(**cam_intrinsics['intrinsics_color'])
    if args.use_depth: 
        depth_camera = pym3t.DummyDepthCamera('cam_depth')
        depth_camera.depth2color_pose = inv_SE3(color_camera.color2depth_pose)
        depth_camera.intrinsics = pym3t.Intrinsics(**cam_intrinsics['intrinsics_depth'])

    # Most time is spent on rendering (tested without GPU: ~15 ms for both, 8 for color only)
    color_viewer = pym3t.NormalColorViewer('color_viewer', color_camera, renderer_geometry)
    tracker.AddViewer(color_viewer)
    if args.use_depth and args.use_depth_viewer:
        depth_viewer = pym3t.NormalDepthViewer('depth_viewer_name', depth_camera, renderer_geometry)
        tracker.AddViewer(depth_viewer)
    else:
        depth_viewer = None

    # Setup body model and properties
    obj_model_path = Path(args.models_dir) / f'{args.body_name}.obj'
    if not obj_model_path.exists(): raise ValueError(f'{obj_model_path} is a wrong path')
    print(f'Loading object {obj_model_path}')
    body = pym3t.Body(
        name=args.body_name,
        geometry_path=obj_model_path.as_posix(),
        geometry_unit_in_meter=args.scale_geometry,
        geometry_counterclockwise=1,
        geometry_enable_culling=1,
        geometry2body_pose=np.eye(4)
    )
    renderer_geometry.AddBody(body)

    # Set up link: m3t handles polyarticulated systems but 
    # here we have only one link corresponding to the object with identity transform wrt to the body
    link = pym3t.Link(args.body_name + '_link', body)

    # Region Modality
    if args.use_region:
        region_model_path = tmp_dir / (args.body_name + '_region_model.bin')
        region_model = pym3t.RegionModel(args.body_name + '_region_model', body, region_model_path.as_posix())
        region_modality = pym3t.RegionModality(args.body_name + '_region_modality', body, color_camera, region_model)
        if args.measure_occlusions and args.use_depth:
            region_modality.MeasureOcclusions(depth_camera)
        link.AddModality(region_modality)

    # Depth Modality
    if args.use_depth:
        depth_model_path = tmp_dir / (args.body_name + '_depth_model.bin')
        depth_model = pym3t.DepthModel(args.body_name + '_depth_model', body, depth_model_path.as_posix())
        depth_modality = pym3t.DepthModality(args.body_name + '_depth_modality', body, depth_camera, depth_model)
        if args.measure_occlusions and args.use_depth:
            depth_modality.MeasureOcclusions()
        link.AddModality(depth_modality)

    # Texture Modality
    if args.use_texture:
        # Texture modality does not require a model contrary to region and depth (for sparse view precomputations)
        color_silhouette_renderer = pym3t.FocusedSilhouetteRenderer('color_silhouette_renderer', renderer_geometry, color_camera)
        color_silhouette_renderer.AddReferencedBody(body)
        texture_modality = pym3t.TextureModality(args.body_name + '_texture_modality', body, color_camera, color_silhouette_renderer)
        if args.measure_occlusions and args.use_depth:
            texture_modality.MeasureOcclusions(depth_camera)
        link.AddModality(texture_modality)

    optimizer = pym3t.Optimizer(args.body_name+'_optimizer', link)

    tracker.AddOptimizer(optimizer)

    ok = tracker.SetUp()
    if not ok:
        raise ValueError('tracker SetUp failed')

    if args.use_depth:
        return tracker, optimizer, body, link, color_camera, depth_camera, color_viewer, depth_viewer
    else:
        return tracker, optimizer, body, link, color_camera, color_viewer


def ExecuteTrackingStepSingleObject(tracker: pym3t.Tracker, link: pym3t.Link, body: pym3t.Body, 
                                    iteration: int, tikhonov_trans: float, tikhonov_rot: float,  
                                    n_corr_iteration=5, n_update_iterations=2):
    """
    Reproducing the coarse to fine optimization procedure.
     
    Meant for educational purpose only, tracker.ExecuteTrackingStep should used instead for practical applications.
    - CalculateCorrespondences: 
        - RegionModality: get correspondance lines by projecting contour points and normals to camera image and computing
                          likelihood of background/foreground belonging for each pixel
        - DepthModality: ICP like with point-2-plane error metric
        - TextureModality: detect feature/descriptor in new image and match them with previous keyframe

    n_corr_iterations: number of times new correspondences are established
    n_update_iterations: number of times the pose is updated for each correspondence iteration
    """
    for corr_iteration in range(n_corr_iteration):
        """
        CalculateCorrespondences gathers information from current image for all setup modalities. M3T includes: 
        - RegionModality: get correspondance lines by projecting contour points and normals to camera image and computing
                          likelihood that each pixel belongs to background/foreground
        - DepthModality: ICP like with point-2-plane error metric
        - TextureModality: detect feature/descriptor in new image and match them with previous keyframe
        """

        tracker.CalculateCorrespondences(iteration, corr_iteration)
        for update_iteration in range(n_update_iterations):

            # Compute gradient hessians of the log likelihoods for all modalities
            tracker.CalculateGradientAndHessian(iteration, corr_iteration, update_iteration)

            ###################
            # Agregate gradient and hessian for each modality and store in link
            # Unconventionally, hessian and gradients are implemented as derivatives of the log-likelihood
            # -> minus sign to minimize the negative log-likelihood 
            H = np.zeros((6,6))
            g = np.zeros(6)
            for modality in link.modalities:
                H -= modality.hessian
                g -= modality.gradient

            # Solve normal equation with tikhonov regularization
            # Delta pose is defined using SO(3)xR3 representation (angle axis, translation)
            # expressed in the local body frame (-> right multiplication)
            tikho_vec = np.concatenate([3*[tikhonov_rot], 3*[tikhonov_trans]])
            delta_pose = np.linalg.solve(H + np.diag(tikho_vec), -g)
            
            # Update link pose
            delta_theta, delta_posi = delta_pose[:3], delta_pose[3:]
            delta_R = cv2.Rodrigues(delta_theta)[0]  # SO(3) exponential map
            pose_variation = np.eye(4)
            pose_variation[:3,:3] = delta_R
            pose_variation[:3,3] = delta_posi
            body.body2world_pose = body.body2world_pose @ pose_variation
            ###################

    # Update modalities states 
    # - region modality: update color histograms
    # - texture modality: update keyframe if enough rotation, render silhouette+depth and update tracked features
    tracker.CalculateResults(iteration)