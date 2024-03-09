"""
Minimal Regional Nerfacto DataManager
"""

from dataclasses import dataclass, field
from typing import Dict, Literal, Tuple, Type, Union

import torch
import json

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)
from pathlib import Path

import numpy as np
import math

from minimal_regional_nerfacto.utils.geodetic_utils import get_elevation


@dataclass
class MRNerfDataManagerConfig(VanillaDataManagerConfig):
    """MRNerf DataManager Config

    Add your custom datamanager config parameters here.
    """

    _target: Type = field(default_factory=lambda: MRNerfDataManager)


class MRNerfDataManager(VanillaDataManager):
    """MRNerf DataManager

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: MRNerfDataManagerConfig

    def __init__(
        self,
        config: MRNerfDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,  # pylint: disable=unused-argument
    ):
        super().__init__(
            config=config, device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank, **kwargs
        )

        self.create_enu_mapping()
        print("--------------------------------")
        print("[MRNeRF Data Manager] Finished ENU Mappings")

        # print(f"[Data Manager] len training dataset was: {len(self.train_dataset)}")

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
                transform_scale = meta["scale"]
            else:
                transform_scale = 1.0
            if "lat" in meta.keys() and "lon" in meta.keys():
                rclat = np.radians(meta["lat"])
                rclon = np.radians(meta["lon"])
                rot_ECEF2ENUV = np.array([
                                [-math.sin(rclon),                  math.cos(rclon),                                0],
                                [-math.sin(rclat)*math.cos(rclon), -math.sin(rclat)*math.sin(rclon),  math.cos(rclat)],
                                [math.cos(rclat)*math.cos(rclon),   math.cos(rclat)*math.sin(rclon),  math.sin(rclat)]
                                ])
            
            else:
                rot_ECEF2ENUV = np.eye(3)
        else:
            raise ValueError("Could not find the current transforms.\n" + \
                             "Check the following (self.train_dataparser_outputs.image_filenames[0]):\n" +
                             f"{self.train_dataparser_outputs.image_filenames[0]}")

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
        
        # Camera poses in transformed reference frame
        camera_poses = self.train_dataparser_outputs.cameras.camera_to_worlds   # N x 3 x 4 
        camera_poses = torch.cat(
                [camera_poses, torch.tensor([[[0, 0, 0, 1]]], dtype=camera_poses.dtype).repeat(len(camera_poses), 1, 1)], 1
            )

        def enu2nerf(poses):
            """
            poses: N x 4 x 4
            """
            # Scale to match the scale of the dataparser
            poses[..., :3, 3] *= transform_scale
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
            poses[..., :3, 3] /= transform_scale
            return poses

        def enu2nerf_points(points):
            """
            points: N x 3
            """
            # Convert to pose via identity rotation
            points *= transform_scale
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
            
            points /= transform_scale
            return points

        self.enu2nerf = enu2nerf
        self.nerf2enu = nerf2enu
        self.enu2nerf_points = enu2nerf_points
        self.nerf2enu_points = nerf2enu_points
        self.center_latlon = [meta["lat"], meta["lon"]]

        self.center_height = float(get_elevation(*self.center_latlon))

        # print("--------------------------------")
        # print("[MRNeRF Data Manager] Finished ENU Mappings")

        