#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2023-12-10
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#
# Utilities for CosyPose object pose estimator tailored for agimus winter school.


import torch
import yaml
from happypose.pose_estimators.cosypose.cosypose.config import EXP_DIR, LOCAL_DATA_DIR
from happypose.pose_estimators.cosypose.cosypose.integrated.detector import Detector
from happypose.pose_estimators.cosypose.cosypose.integrated.pose_estimator import (
    PoseEstimator,
)
from happypose.pose_estimators.cosypose.cosypose.training.detector_models_cfg import (
    check_update_config as check_update_config_detector,
)
from happypose.pose_estimators.cosypose.cosypose.training.detector_models_cfg import (
    create_model_detector,
)
from happypose.pose_estimators.cosypose.cosypose.training.pose_models_cfg import (
    check_update_config as check_update_config_pose,
)
from happypose.pose_estimators.cosypose.cosypose.training.pose_models_cfg import (
    create_model_coarse,
    create_model_refiner,
)
from happypose.toolbox.datasets.bop_object_datasets import BOPObjectDataset
from happypose.toolbox.lib3d.rigid_mesh_database import MeshDataBase
from happypose.toolbox.renderer.panda3d_batch_renderer import Panda3dBatchRenderer


class AWSCosyPose:
    @staticmethod
    def load_detector(
        device="cpu",
        ds_name="ycbv",
        run_id="detector-bop-ycbv-pbr--970850",
    ):
        """Load CosyPose detector."""
        run_dir = EXP_DIR / run_id
        assert run_dir.exists(), "The run_id is invalid, or you forget to download data"
        cfg = check_update_config_detector(
            yaml.load((run_dir / "config.yaml").read_text(), Loader=yaml.UnsafeLoader),
        )
        label_to_category_id = cfg.label_to_category_id
        ckpt = torch.load(run_dir / "checkpoint.pth.tar", map_location=device)[
            "state_dict"
        ]
        model = create_model_detector(cfg, len(label_to_category_id))
        model.load_state_dict(ckpt)
        model = model.to(device).eval()
        model.cfg = cfg
        model.config = cfg
        return Detector(model, ds_name)

    @staticmethod
    def _load_pose_model(run_id, renderer, mesh_db, device):
        """Load either coarse or refiner model (decided based on run_id/config)."""
        run_dir = EXP_DIR / run_id
        cfg = yaml.load((run_dir / "config.yaml").read_text(), Loader=yaml.UnsafeLoader)
        cfg = check_update_config_pose(cfg)

        f_mdl = create_model_refiner if cfg.train_refiner else create_model_coarse
        ckpt = torch.load(run_dir / "checkpoint.pth.tar", map_location=device)[
            "state_dict"
        ]
        model = f_mdl(cfg, renderer=renderer, mesh_db=mesh_db)
        model.load_state_dict(ckpt)
        model = model.to(device).eval()
        model.cfg = cfg
        model.config = cfg
        return model

    @staticmethod
    def load_pose_estimator(
        coarse_run_id="coarse-bop-ycbv-pbr--724183",
        refiner_run_id="refiner-bop-ycbv-pbr--604090",
        object_dataset=None,
        n_workers=1,
        device="cpu",
    ):
        """Load coarse and refiner, put them inside PoseEstimator and return it."""
        if object_dataset is None:
            object_dataset = BOPObjectDataset(
                # LOCAL_DATA_DIR / "examples" / "crackers_example" / "models",
                LOCAL_DATA_DIR / "bop_datasets/ycbv/models_bop-compat",
                label_format="ycbv-{label}",
            )
        renderer = Panda3dBatchRenderer(
            object_dataset,
            n_workers=n_workers,
            preload_cache=False,
        )

        mesh_db = MeshDataBase.from_object_ds(object_dataset)
        mesh_db_batched = mesh_db.batched().to(device)
        kwargs = {"renderer": renderer, "mesh_db": mesh_db_batched, "device": device}
        coarse_model = AWSCosyPose._load_pose_model(coarse_run_id, **kwargs)
        refiner_model = AWSCosyPose._load_pose_model(refiner_run_id, **kwargs)

        return PoseEstimator(
            refiner_model=refiner_model,
            coarse_model=coarse_model,
            detector_model=None,
            bsz_objects=1,
            bsz_images=1,
        )
