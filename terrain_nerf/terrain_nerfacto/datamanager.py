"""
Regional Nerfacto DataManager
"""

import json
import math
import os.path as osp
from dataclasses import dataclass, field
from io import BytesIO
from math import cos, floor, log, pi, sin, tan
from pathlib import Path
from typing import Dict, Literal, Tuple, Type, Union

import numpy as np
import requests
import torch
from PIL import Image

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager, VanillaDataManagerConfig)


@dataclass
class TNerfDataManagerConfig(VanillaDataManagerConfig):
    """TNerf DataManager Config

    Add your custom datamanager config parameters here.
    """

    _target: Type = field(default_factory=lambda: TNerfDataManager)


class TNerfDataManager(VanillaDataManager):
    """TNerf DataManager

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: TNerfDataManagerConfig

    def __init__(
        self,
        config: TNerfDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,  # pylint: disable=unused-argument
    ):
        super().__init__(
            config=config, device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank, **kwargs
        )

        # Generate mapping from ENU to NERF coordinate systems
        self.create_enu_mapping()


    def _find_transform(self, image_path: Path) -> Union[Path, None]:
        while image_path.parent != image_path:
            transform_path = image_path.parent / "transforms.json"
            if transform_path.exists():
                return transform_path
            image_path = image_path.parent
        return None

    def create_enu_mapping(self):
        transforms = self._find_transform(self.train_dataparser_outputs.image_filenames[0])

        if transforms is not None:
            meta = json.load(open(transforms, "r"))
            if "scale" in meta.keys():
                scale = torch.tensor(meta["scale"], dtype=torch.float32)
                if scale.numel() == 3:
                    transform_scale = torch.diag(scale)
                    transform_scale_inv = torch.diag(1.0/scale)
                else:
                    transform_scale = scale * torch.eye(3)
                    transform_scale_inv = (1.0/scale) * torch.eye(3)
            else:
                transform_scale = torch.eye(3)
                transform_scale_inv = torch.eye(3)

        # Retrive dataparser transformers
        dataparser_scale = self.train_dataparser_outputs.dataparser_scale
        dataparser_transform = self.train_dataparser_outputs.dataparser_transform   # 3 x 4

        dataparser_rotation = dataparser_transform[:3, :3].to(self.device)
        dataparser_translation = dataparser_transform[:3, 3].to(self.device)

        dataparser_transform = torch.eye(4, device=self.device)
        dataparser_transform[:3, :3] = dataparser_rotation
        dataparser_transform[:3, 3] = dataparser_translation

        dataparser_transform_inv = torch.eye(4, device=self.device)
        dataparser_transform_inv[:3, :3] = dataparser_rotation.T
        dataparser_transform_inv[:3, 3] = -dataparser_rotation.T @ dataparser_translation

        # print("dataparser_transform", dataparser_transform)
        # print("dataparser_transform_inv", dataparser_transform_inv)
        # print("dataparser_scale", dataparser_scale)
        # print("transform_scale", transform_scale)

        def enu2nerf(poses):
            """
            poses: N x 4 x 4
            """
            # Scale to match the scale of the dataparser
            poses[..., :3, 3] = poses[..., :3, 3] @ transform_scale
            # Transform to the dataparser reference frame
            poses = dataparser_transform @ poses
            poses[..., :3, 3] *= dataparser_scale
            return poses

        def nerf2enu(poses):
            """
            poses: N x 4 x 4
            """
            # Scale to match the scale of the dataparser
            poses[..., :3, 3] /= dataparser_scale
            # Transform to the dataparser reference frame
            poses = dataparser_transform_inv @ poses
            poses[..., :3, 3] = poses[..., :3, 3] @ transform_scale_inv
            return poses

        def enu2nerf_points(points):
            """
            points: N x 3
            """
            # Convert to pose via identity rotation
            points = points @ transform_scale
            points = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)
            points = dataparser_transform @ points[..., None]
            points = points[..., :3, 0]
            points *= dataparser_scale
            return points

        def nerf2enu_points(points):
            """
            points: ... x 3
            """
            # Convert to pose via identity rotation
            points /= dataparser_scale
            points = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)
            points = dataparser_transform_inv @ points[..., None]
            points = points[..., :3, 0]
            points = points @ transform_scale_inv
            return points
        
        self.enu2nerf = enu2nerf
        self.nerf2enu = nerf2enu
        self.enu2nerf_points = enu2nerf_points
        self.nerf2enu_points = nerf2enu_points

        print("--------------------------------")
        print("[Data Manager] Finished ENU Mappings")



        