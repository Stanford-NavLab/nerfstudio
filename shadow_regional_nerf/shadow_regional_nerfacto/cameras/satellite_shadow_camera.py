"""
The goal here is to make a new "camera" model that will take 
a grid of points and a satellite (either as a position or a direction,
depending on the camera) and it will generate the "image" of the 
shadow on the ground. Or, more precisely, it will render the outputs
onto the grid on the ground.

Notice that this is a bit of a 'flipped' camera. Usually, we have a
roughly shared origin of the ray bundle, but with different directions.
Now, we have very different ray origins and roughly shared directions.
"""

from typing import Optional, Union

import torch
from jaxtyping import Float, Shaped
from torch import Tensor

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.scene_box import OrientedBox, SceneBox
from nerfstudio.utils.tensor_dataclass import TensorDataclass

TORCH_DEVICE = Union[torch.device, str]


class SatelliteDirectionCamera(TensorDataclass):
    """
    Represent the satellite as a 'direction'
    """

    width:  Shaped[Tensor, "*num_cameras 1"]
    height: Shaped[Tensor, "*num_cameras 1"]

    def __init__(self,
        # device,
        # width: Optional[Union[Shaped[Tensor, "*batch_ws 1"], int]] = None,
        # height: Optional[Union[Shaped[Tensor, "*batch_hs 1"], int]] = None
    ):
        pass
        # self.device = device
        # self.width  = width
        # self.height = height

    # @property
    # def device(self) -> TORCH_DEVICE:
    #     """Returns the device that the camera is on."""
    #     return self.camera_to_worlds.device

    # @property
    # def image_height(self) -> Shaped[Tensor, "*num_cameras 1"]:
    #     """Returns the height of the images."""
    #     return self.height

    # @property
    # def image_width(self) -> Shaped[Tensor, "*num_cameras 1"]:
    #     """Returns the height of the images."""
    #     return self.width

    def generate_rays(
        self,
        direction: Float[Tensor, "3"],
        origins: Float[Tensor, "*num_rays 3"],
        pixel_area: Float[Tensor, "1"],
        aabb_box: Optional[SceneBox] = None,
        obb_box: Optional[OrientedBox] = None
    ) -> RayBundle:
        """
        Generate rays.

        This function is based on
        `nerfstudio > cameras > cameras.py > generate_rays(...)
        """
        # Check the shapes
        assert direction.shape == (3,)
        assert origins.shape[1] == 3
        assert pixel_area.shape == (1,)

        num_rays = origins.shape[0]

        # Expand the direction and pixel area to match the origins shape
        directions  =  direction.reshape(1, 3).repeat(num_rays, 1)
        pixel_areas = pixel_area.reshape(1, 1).repeat(num_rays, 1)

        # Check the shapes
        assert directions.shape == origins.shape, \
            f"Directions has shape {directions.shape}," + \
            f"but origins has shape {origins.shape}"
        assert pixel_areas.shape == origins.shape, \
            f"Pixel areas has shape {pixel_areas.shape}," + \
            f"but origins has shape {origins.shape}"

        raybundle = RayBundle(
            origins=origins,
            directions=directions,
            pixel_area=pixel_areas,
            camera_indices=None,
            times=None,
            metadata=None,
        )

        if aabb_box is not None or obb_box is not None:
            with torch.no_grad():
                rays_o = raybundle.origins.contiguous()
                rays_d = raybundle.directions.contiguous()

                shape = rays_o.shape

                rays_o = rays_o.reshape((-1, 3))
                rays_d = rays_d.reshape((-1, 3))

                if aabb_box is not None:
                    tensor_aabb = Parameter(aabb_box.aabb.flatten(), requires_grad=False)
                    tensor_aabb = tensor_aabb.to(rays_o.device)
                    t_min, t_max = nerfstudio.utils.math.intersect_aabb(rays_o, rays_d, tensor_aabb)
                elif obb_box is not None:
                    t_min, t_max = nerfstudio.utils.math.intersect_obb(rays_o, rays_d, obb_box)
                else:
                    assert False

                t_min = t_min.reshape([shape[0], shape[1], 1])
                t_max = t_max.reshape([shape[0], shape[1], 1])

                raybundle.nears = t_min
                raybundle.fars = t_max

        return ray_bundle
    
