"""
Regional Nerfacto Field

Currently this subclasses the NerfactoField. Consider subclassing the base Field.
"""

from typing import Dict, Literal, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field  # for custom Field
from nerfstudio.fields.nerfacto_field import \
    NerfactoField  # for subclassing NerfactoField

try:
    import tinycudann as tcnn
except ImportError:
    pass


class RNerfField(NerfactoField):
    """Regional Nerf Field
    """

    def __init__(
        self, 
        grid_resolutions,
        grid_layers,
        grid_sizes,
        **kwargs) -> None:
        super().__init__(**kwargs)

        self.top_cutoff = 1.0

        self.encs = torch.nn.ModuleList(
            [
                RNerfField._get_encoding(
                    grid_resolutions[i][0], grid_resolutions[i][1], grid_layers[i], indim=3, hash_size=grid_sizes[i]
                )
                for i in range(len(grid_layers))
            ]
        )
        tot_out_dims = sum([e.n_output_dims for e in self.encs])

        self.dino_net = tcnn.Network(
            n_input_dims=tot_out_dims,
            n_output_dims=384,
            network_config={
                "otype": "CutlassMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 256,
                "n_hidden_layers": 1,
            },
        )
        
        self.encs2d = torch.nn.ModuleList(
            [
                RNerfField._get_encoding(
                    grid_resolutions[i][0], grid_resolutions[i][1], grid_layers[i], indim=2, hash_size=grid_sizes[i]
                )
                for i in range(len(grid_layers))
            ]
        )
        
        tot_out_dims_2d = sum([e.n_output_dims for e in self.encs2d])
        
        # Create a network that maps elevation z = f(x, y) for surface
        self.heightcap_net = tcnn.Network(
            n_input_dims=tot_out_dims_2d,
            n_output_dims=1,
            network_config={
                "otype": "CutlassMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 256,
                "n_hidden_layers": 1,
            },
        )

        # Create a single trainable variable to store ground height
        self.ground_height = torch.nn.Parameter(torch.tensor(0.0))

    @staticmethod
    def _get_encoding(start_res, end_res, levels, indim=3, hash_size=19):
        growth = np.exp((np.log(end_res) - np.log(start_res)) / (levels - 1))
        enc = tcnn.Encoding(
            n_input_dims=indim,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": levels,
                "n_features_per_level": 8,
                "log2_hashmap_size": hash_size,
                "base_resolution": start_res,
                "per_level_scale": growth,
            },
        )
        return enc

    def set_enu_transform(self, enu2nerf, nerf2enu, enu2nerf_points, nerf2enu_points, osm_image, osm_scale):
        self.enu2nerf = enu2nerf
        self.nerf2enu = nerf2enu
        self.enu2nerf_points = enu2nerf_points
        self.nerf2enu_points = nerf2enu_points
        self.osm_image = osm_image
        self.osm_scale = osm_scale

    def get_density(self, ray_samples: RaySamples, do_heightcap=True) -> Tuple[Tensor, Tensor]:
        """Computes and returns the densities."""
        if self.spatial_distortion is not None:
            unnorm_positions = ray_samples.frustums.get_positions()
            positions = self.spatial_distortion(unnorm_positions)
            positions = (positions + 2.0) / 4.0
        else:
            unnorm_positions = ray_samples.frustums.get_positions()
            positions = SceneBox.get_normalized_positions(unnorm_positions, self.aabb)
        # Make sure the tcnn gets inputs between 0 and 1.
        
        if do_heightcap:
            xs = [e(positions.view(-1, 3)[:, :2]) for e in self.encs2d]
            x = torch.concat(xs, dim=-1)

            heightcaps = self.heightcap_net(x).view(*ray_samples.frustums.shape)
            ground_height = self.ground_height
        else:
            heightcaps = 10000.0
            ground_height = -10000.0

        # Selector to mask out positions with z higher than heightcaps
        # selector_0 = (unnorm_positions[..., 2] <= heightcaps)
        selector_0 = (unnorm_positions[..., 2] <= heightcaps) & (unnorm_positions[..., 2] >= ground_height) # Navlab added
            
        selector = selector_0 & ((positions > 0.0) & (positions < 1.0)).all(dim=-1)

        positions = positions * selector[..., None]
        self._sample_locations = positions
        if not self._sample_locations.requires_grad:
            self._sample_locations.requires_grad = True
        positions_flat = positions.view(-1, 3)
        h = self.mlp_base(positions_flat).view(*ray_samples.frustums.shape, -1)
        density_before_activation, base_mlp_out = torch.split(h, [1, self.geo_feat_dim], dim=-1)
        self._density_before_activation = density_before_activation

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = trunc_exp(density_before_activation.to(positions))
        density = density * selector[..., None]
        
        # heights = unnorm_positions[..., 2][..., None]

        # clusters = (density > 0.1).float()
        
        # max_heights = torch.max(heights * clusters, dim=-2).values
        
        # selectors = (heights[..., 0] <= max_heights)
        
        # density = density * selectors[..., None]
        
        return density, base_mlp_out
    
    def get_dino_outputs(
        self, ray_samples: RaySamples) -> Dict[FieldHeadNames, Tensor]:
        outputs = {}

        positions = ray_samples.frustums.get_positions().detach()
        positions = self.spatial_distortion(positions)
        positions = (positions + 2.0) / 4.0

        xs = [e(positions.view(-1, 3)) for e in self.encs]
        x = torch.concat(xs, dim=-1)

        dino_pass = self.dino_net(x).view(*ray_samples.frustums.shape, -1)
        outputs["dino"] = dino_pass

        return outputs
    
    def get_heightcap_outputs(
        self, ray_samples: RaySamples) -> Dict[FieldHeadNames, Tensor]:
        outputs = {}

        positions = ray_samples.frustums.get_positions().detach()
        positions = self.spatial_distortion(positions)
        positions = (positions + 2.0) / 4.0

        xs = [e(positions.view(-1, 3)[:, :2]) for e in self.encs2d]
        x = torch.concat(xs, dim=-1)

        heightcap_pass = self.heightcap_net(x).view(*ray_samples.frustums.shape, -1)
        outputs["heightcap"] = heightcap_pass
        
        return outputs
    
    def get_ground_height(self):
        return self.ground_height

    def positions_to_heights(self, positions):
        inp_shape = positions.shape
        positions = torch.cat([positions, torch.zeros_like(positions[..., :1])], dim=-1)
        positions = self.spatial_distortion(positions)
        positions = (positions + 2.0) / 4.0

        xs = [e(positions.view(-1, 3)[:, :2]) for e in self.encs2d]
        x = torch.concat(xs, dim=-1)

        # TODO: Reshape to multi-batch
        heights = self.heightcap_net(x).view(*inp_shape[:-1], -1)
        return heights


    def xy_to_heights(self, x, y):
        XY_grid = torch.stack((x, y), dim=-1)
        positions = XY_grid.reshape(-1, 2)
        positions = torch.cat([positions, torch.zeros_like(positions[:, :1])], dim=-1)
        positions = self.spatial_distortion(positions)
        positions = (positions + 2.0) / 4.0

        xs = [e(positions.view(-1, 3)[:, :2]) for e in self.encs2d]
        x = torch.concat(xs, dim=-1)

        # TODO: Reshape to multi-batch
        heights = self.heightcap_net(x)
        return heights