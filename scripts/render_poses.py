from nerfstudio.cameras.cameras import Cameras, CameraType, RayBundle
from nerfstudio.scripts.render import _render_trajectory_video, BaseRender
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union
import numpy as np
import torch
import gzip
import json
import os
import shutil
import struct
import sys
from torch import Tensor
from nerfstudio.utils.eval_utils import eval_setup
import pymap3d as pm
from types import SimpleNamespace
from nerfstudio.viewer_legacy.server.utils import three_js_perspective_camera_focal_length
import requests 
import urllib
import tyro
from typing_extensions import Annotated

from minimal_regional_nerfacto.utils.geodetic_utils import get_elevation

# def get_elevation_usgs(lat, lon):
#     url = 'https://epqs.nationalmap.gov/v1/json?'

#     params = {
#         'x': lon,
#         'y': lat, 
#         'units': 'Meters',
#         'output': 'json'
#     }
  
#     full_url = url + urllib.parse.urlencode(params)
#     print("Querying...", full_url)
#     response = requests.get(full_url)
#     data = json.loads(response.text)
#     print("...Done")
#     if 'value' not in data.keys():
#         print(data['message'])
#         raise ValueError
#     return data['value']

def enu_to_ecef_rotation(lat, lon):
    # Compute rotation matrix from ENU to ECEF
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    
    slat = np.sin(lat_rad)
    clat = np.cos(lat_rad)
    slon = np.sin(lon_rad)
    clon = np.cos(lon_rad)
    
    # ENU to ECEF rotation matrix
    R = torch.tensor([
        [-slon, -slat*clon, clat*clon],
        [clon, -slat*slon, clat*slon],
        [0, clat, slat]
    ])
    
    return R

def get_path_from_json(pipeline, camera_path: Dict[str, Any]) -> Cameras:
    """Takes a camera path dictionary and returns a trajectory as a Camera instance.

    Args:
        camera_path: A dictionary of the camera path information coming from the viewer.

    Returns:
        A Cameras instance with the camera path.
    """

    image_height = camera_path["render_height"]
    image_width = camera_path["render_width"]

    # central lat/lon for provided trajectory
    lla_center_trajectory = torch.tensor(camera_path["lat_lon_alt_center"])
    ecef_ref_trajectory = pm.geodetic2ecef(lla_center_trajectory[0], lla_center_trajectory[1], lla_center_trajectory[2])
    R_enu2ecef = enu_to_ecef_rotation(lla_center_trajectory[0], lla_center_trajectory[1]).float()

    # central lat/lon for NeRF
    ll_center_nerf = pipeline.datamanager.center_latlon
    height_center_nerf = float(get_elevation(ll_center_nerf[0], ll_center_nerf[1]))
    R_ecef2enu = enu_to_ecef_rotation(ll_center_nerf[0], ll_center_nerf[1]).float().T

    if "camera_type" not in camera_path:
        camera_type = CameraType.PERSPECTIVE
    elif camera_path["camera_type"] == "fisheye":
        camera_type = CameraType.FISHEYE
    elif camera_path["camera_type"] == "equirectangular":
        camera_type = CameraType.EQUIRECTANGULAR
    elif camera_path["camera_type"].lower() == "omnidirectional":
        camera_type = CameraType.OMNIDIRECTIONALSTEREO_L
    elif camera_path["camera_type"].lower() == "vr180":
        camera_type = CameraType.VR180_L
    else:
        camera_type = CameraType.PERSPECTIVE

    print("========= [Render Poses] =========")
    print(f"[Render Poses] Camera Type set to {camera_type}")

    c2ws = []
    fxs = []
    fys = []
    for camera in camera_path["camera_path"]:
        # lat/lon
        lla = camera["lat_lon_alt"]
        # rotation (w.r.t. ENU frame)
        rot = torch.tensor(camera["rotation"]).view(3, 3).float()
        
        # Compute ENU w.r.t. center of trajectory
        # traj_east, traj_north, traj_up = pm.geodetic2enu(lla[0], lla[1], lla[2], lla_center_trajectory[0], lla_center_trajectory[1], lla_center_trajectory[2])

        # Compute ENU in NeRF frame
        trans_enu_nerf = torch.tensor(pm.geodetic2enu(lla[0], lla[1], lla[2], ll_center_nerf[0], ll_center_nerf[1], height_center_nerf)).float().view(3, 1)
        # print("Input geodetic Query ", lla[0], lla[1], lla[2])
        # print("Input geodetic center ", ll_center_nerf[0], ll_center_nerf[1], height_center_nerf)
        # print("Output ENU ", trans_enu_nerf)

        # Transform pose to ECEF
        rot_enu_nerf = torch.mm(R_ecef2enu, torch.mm(R_enu2ecef, rot))

        # Compute pose in NeRF ENU frame
        c2w = torch.cat([rot_enu_nerf, trans_enu_nerf], 1)
        # print("Before ENU2NERF ", c2w)
        c2w = torch.cat((c2w, torch.tensor([0,0,0,1]).view(1, 4)), 0).view(1, 4, 4)
        # Transform pose in NeRF coordinate frame
        c2w = pipeline.datamanager.enu2nerf(c2w.cuda())[0, :3, :].cpu()
        # print("After ENU2NERF ", c2w)
        
        c2ws.append(c2w)
        if camera_type in [
            CameraType.EQUIRECTANGULAR,
            CameraType.OMNIDIRECTIONALSTEREO_L,
            CameraType.OMNIDIRECTIONALSTEREO_R,
            CameraType.VR180_L,
            CameraType.VR180_R,
        ]:
            fxs.append(image_width / 2)
            fys.append(image_height)
        else:
            # field of view
            fov = camera["fov"]
            focal_length = three_js_perspective_camera_focal_length(fov, image_height)
            fxs.append(focal_length)
            fys.append(focal_length)

    # Iff ALL cameras in the path have a "time" value, construct Cameras with times
    if all("render_time" in camera for camera in camera_path["camera_path"]):
        times = torch.tensor([camera["render_time"] for camera in camera_path["camera_path"]])
    else:
        times = None

    camera_to_worlds = torch.stack(c2ws, dim=0)
    fx = torch.tensor(fxs)
    fy = torch.tensor(fys)
    return Cameras(
        fx=fx,
        fy=fy,
        cx=image_width / 2,
        cy=image_height / 2,
        camera_to_worlds=camera_to_worlds,
        camera_type=camera_type,
        times=times,
    )

# There is some deviation from USGS elevation values and the rendered ground values (for mem church, <12m)
# Identity rotation points downwards
# Row-swapping z with y-axis is North-facing
# Further row-swapping y with x-axis is west-facing 

@dataclass
class RenderCameraPose(BaseRender):
    camera_path_filename: Path = Path("camera_path_global.json")
    """Filename of the camera path to render."""

    output_path: Path = Path("renders")
    """Filename of the output folder/video."""

    output_format: Literal["images", "video"] = "images"
    """How to save output data."""

    def main(self) -> None:
        """Main function."""
        print("[Render Poses] Setting up pipeline...")
        _, pipeline, _, _ = eval_setup(
            self.load_config,
            eval_num_rays_per_chunk=self.eval_num_rays_per_chunk,
            test_mode="inference",
        )
        print("[Render Poses] ...Setting up pipeline done")

        with open(self.camera_path_filename, "r", encoding="utf-8") as f:
            camera_path = json.load(f)
        
        print("[Render Poses] Computing Cameras...")
        cameras = get_path_from_json(pipeline, camera_path)
        print("[Render Poses] ...Done")

        print("[Render Poses] Printing self")
        print(self.nice_self_json())

        print("[Render Poses] Saving params")
        self.save_params_to_json()
        # Also copy the current JSON
        dst = os.path.join(self.output_path, "camera_path.json")
        shutil.copyfile(self.camera_path_filename, dst)

        # Note: If there is something that is not currently being passed, 
        # it will not be used. :) Just a heads up :)
        _render_trajectory_video(
            pipeline=pipeline, 
            cameras=cameras, 
            output_filename=self.output_path,
            rendered_output_names=self.rendered_output_names,
            output_format = self.output_format,
            depth_near_plane = self.depth_near_plane,
            depth_far_plane = self.depth_far_plane,
            colormap_options = self.colormap_options
            )

    def nice_self_dict(self):
        """Take the class variables and convert to dictionary."""
        dict_str = {}

        # Needed since PosixPath cannot be directly converted to JSON
        for key, val in vars(self).items():
            if isinstance(val, (int, float)):
                dict_str[key] = val
            else:
                dict_str[key] = str(val)
        
        return dict_str
    
    def nice_self_json(self):
        """Take the above dictionary and convert to JSON."""
        return json.dumps(self.nice_self_dict(), indent=4)
    
    def save_params_to_json(self):
        """Save the above function to a file"""
        if self.output_path is None:
            print("Could not save params file, output path is None.")
        else:
            json_to_save = self.nice_self_dict()
            file_out_name = os.path.join(self.output_path, 'render_poses_params.json')

            # Make directories, if needed
            os.makedirs(os.path.dirname(file_out_name), exist_ok=True)
            # Then write the file
            with open(file_out_name, 'w', encoding="utf-8") as f: 
                json.dump(json_to_save, f, indent=4)


Commands = tyro.conf.FlagConversionOff[
    Union[
        Annotated[RenderCameraPose, tyro.conf.subcommand(name="camera-pose")]
    ]
]


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(Commands).main()


if __name__ == "__main__":
    entrypoint()


def get_parser_fn():
    """Get the parser function for the sphinx docs."""
    return tyro.extras.get_parser(Commands)  # noqa
