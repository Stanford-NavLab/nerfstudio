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

# from dataclasses import dataclass
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
    Represent A SINGLE satellite (aka SINGLE camera) as a 'direction'
    """

    # origins             #Float[Tensor, "*num_points 3"]
    # direction           #Float[Tensor, "3"]
    # pixel_area_expanded #Float[Tensor, "1"]

    def __init__(self,
        origins:    Float[Tensor, "*batch_origins 3"],
        directions: Float[Tensor, "3"],
        pixel_area: Float[Tensor, "1"]
    ) -> None:

        # Check the shapes
        assert directions.shape == (3,)
        assert origins.shape[1] == 3
        assert pixel_area.shape == (1,)

        self.origins = origins
        self.directions = directions
        self.pixel_area = pixel_area
        
        self.num_points = self.origins.shape[0]

        self.origins_expanded, self.directions_expanded, \
            self.pixel_area_expanded = self._init_get_expand_rays()

        # Check the shapes
        assert self.directions_expanded.shape == self.origins_expanded.shape, \
            f"Directions has shape {self.directions_expanded.shape}," + \
            f"but Origins has shape {self.origins_expanded.shape}"
        assert self.pixel_area_expanded.shape[0] == self.origins_expanded.shape[0], \
            f"Pixel areas has shape {self.pixel_area_expanded.shape}," + \
            f"but Origins has shape {self.origins_expanded.shape}"

    @property
    def device(self) -> TORCH_DEVICE:
        """Returns the device that the camera is on."""
        return self.origins.device

    def _init_get_expand_rays(self):
        """
        Convert origins of shape "*num_points 3" and directions of shape
        "*num_satellites 3" into
        origins_expanded    = ["num_points" 1 3]
        directions_expanded = ["num_points" 1 3]
        pixel_area_expanded = ["num_points" 1 1]
        """
        origins_expanded = self.origins.reshape(-1, 1, 3)
        directions_expanded = self.directions.reshape(1, 1, 3).repeat(
            self.num_points, 1, 1
        )
        pixel_area_expanded = self.pixel_area.reshape(1, 1, 1).repeat(
            self.num_points, 1, 1
        )

        return origins_expanded, directions_expanded, pixel_area_expanded

    def generate_rays(
        self,
        camera_indices=0,
        keep_shape=True,
        aabb_box: Optional[SceneBox] = None,
        obb_box: Optional[OrientedBox] = None
    ) -> RayBundle:
        """
        Generate rays.

        This function is based on
        `nerfstudio > cameras > cameras.py > generate_rays(...)
        """
        # Make sure we're on the right devices
        origins    = self.origins_expanded.to(self.device)
        directions = self.directions_expanded.to(self.device)
        pixel_area = self.pixel_area_expanded.to(self.device)

        # Handle camera indices
        camera_indices = torch.tensor([camera_indices], device=self.device)

        raybundle = RayBundle(
            origins=origins,
            directions=directions,
            pixel_area=pixel_area,
            camera_indices=camera_indices,
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

        return raybundle
    





# # Daniel: Dataclass was having trouble :(
# # @dataclass(init=False)
# class SatelliteDirectionCamera(TensorDataclass):
#     """
#     Represent the satellite as a 'direction'
#     """

#     origins_expanded    #Float[Tensor, "*num_satellites *num_points 3"]
#     directions_expanded #Float[Tensor, "*num_satellites *num_points 3"]
#     pixel_area_expanded #Float[Tensor, "*num_satellites *num_points 1"]

#     def __init__(self,
#         origins:    Float[Tensor, "*batch_origins 3"],
#         directions: Float[Tensor, "*batch_directions 3"],
#         pixel_area: Float[Tensor, "1"]
#     ) -> None:
#         # This will notify the tensordataclass that we have a field with more than 1 dimension
#         self._field_custom_dimensions = {
#             "origins_expanded": 2,
#             "directions_expanded": 2,
#             "pixel_area_expanded": 2
#         }

#         # Check the shapes
#         assert directions.shape[1] == 3
#         assert origins.shape[1] == 3
#         assert pixel_area.shape == (1,)

#         self.origins = origins
#         self.directions = directions
#         self.pixel_area = pixel_area
        
#         self.num_points = self.origins.shape[0]
#         self.num_satellites = self.directions.shape[0]
#         self.total_rays = self.num_points * self.num_satellites

#         self.origins_expanded, self.directions_expanded, \
#             self.pixel_area_expanded = self._init_get_expand_rays()

#         # Check the shapes
#         assert self.directions_expanded.shape == self.origins_expanded.shape, \
#             f"Directions has shape {self.directions_expanded.shape}," + \
#             f"but Origins has shape {self.origins_expanded.shape}"
#         assert self.pixel_area_expanded.shape[:2] == self.origins_expanded.shape[:2], \
#             f"Pixel areas has shape {self.pixel_area_expanded.shape[:2]}," + \
#             f"but Origins has shape {self.origins_expanded.shape[:2]}"

#         # Add back with dataclass fix
#         # self.__post_init__()

#     @property
#     def device(self) -> TORCH_DEVICE:
#         """Returns the device that the camera is on."""
#         return self.origins.device

#     def _init_get_expand_rays(self):
#         """
#         Convert origins of shape "*num_points 3" and directions of shape
#         "*num_satellites 3" into
#         origins_expanded    = ["num_satellites" "num_points" 3]
#         directions_expanded = ["num_satellites" "num_points" 3]
#         pixel_area_expanded = ["num_satellites" "num_points" 1]

#         We put the number of points first since it could be very large and
#         we may want to batch this dimension (if I am understanding correctly)
#         """
#         origins_expanded    = self.origins.reshape(-1, 1, 3).repeat(
#             self.num_satellites, 1, 1
#         )
#         directions_expanded = self.directions.reshape(1, -1, 3).repeat(
#             1, self.num_points, 1
#         )
#         pixel_area_expanded = self.pixel_area.reshape(1, 1, 1).repeat(
#             self.num_satellites, self.num_points, 1
#         )

#         return origins_expanded, directions_expanded, pixel_area_expanded

#     def get_satellite_at_idx(sat_idx):


#     def generate_rays(
#         self,
#         aabb_box: Optional[SceneBox] = None,
#         obb_box: Optional[OrientedBox] = None
#     ) -> RayBundle:
#         """
#         Generate rays.

#         This function is based on
#         `nerfstudio > cameras > cameras.py > generate_rays(...)
#         """
#         # Make sure we're on the right devices
#         origins    = self.origins_expanded.to(self.device)
#         directions = self.directions_expanded.to(self.device)
#         pixel_area = self.pixel_area_expanded.to(self.device)

#         raybundle = RayBundle(
#             origins=origins,
#             directions=directions,
#             pixel_area=pixel_area,
#             camera_indices=None,
#             times=None,
#             metadata=None,
#         )

#         if aabb_box is not None or obb_box is not None:
#             with torch.no_grad():
#                 rays_o = raybundle.origins.contiguous()
#                 rays_d = raybundle.directions.contiguous()

#                 shape = rays_o.shape

#                 rays_o = rays_o.reshape((-1, 3))
#                 rays_d = rays_d.reshape((-1, 3))

#                 if aabb_box is not None:
#                     tensor_aabb = Parameter(aabb_box.aabb.flatten(), requires_grad=False)
#                     tensor_aabb = tensor_aabb.to(rays_o.device)
#                     t_min, t_max = nerfstudio.utils.math.intersect_aabb(rays_o, rays_d, tensor_aabb)
#                 elif obb_box is not None:
#                     t_min, t_max = nerfstudio.utils.math.intersect_obb(rays_o, rays_d, obb_box)
#                 else:
#                     assert False

#                 t_min = t_min.reshape([shape[0], shape[1], 1])
#                 t_max = t_max.reshape([shape[0], shape[1], 1])

#                 raybundle.nears = t_min
#                 raybundle.fars = t_max

#         return ray_bundle
    
