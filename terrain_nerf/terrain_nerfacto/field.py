"""
Terrain Nerfacto Field

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

from terrain_nerfacto.nemo import Nemo
from terrain_nerfacto.utils.siren import Siren


class TNerfField(NerfactoField):
    """Terrain Nerf Field

    Model which jointly learns a NeRF and height field. The height field is extracted through quantile regression on the 
    NeRF densities, and is then used in turn to supervise the training of the densities.

    """

    def __init__(self, 
        grid_resolutions,
        grid_layers,
        grid_sizes,
        **kwargs) -> None:
        super().__init__(**kwargs)
        """
        Parameters
        ----------
        grid_resolutions : list of tuples
            List of tuples of grid resolutions (x, y) for each grid.
        grid_layers : list of int
            List of integers for the number of layers in each grid.
        grid_sizes : list of int
            List of integers for the hash size of each grid.
        
        """
        # 2D Height network
        # self.nemo = Nemo(spatial_distortion=self.spatial_distortion, hash=True)
        self.encs2d = torch.nn.ModuleList(
            [
                tcnn.Encoding(
                    n_input_dims=2,
                    encoding_config={
                        "otype": "HashGrid",
                        "n_levels": 8,
                        "n_features_per_level": 8,
                        "log2_hashmap_size": 19,
                        "base_resolution": 16,
                        "per_level_scale": 1.2599210739135742,
                        "interpolation": "Smoothstep"
                    },
                )
            ]
        )
        
        tot_out_dims_2d = sum([e.n_output_dims for e in self.encs2d])
        
        # Create a network that maps elevation z = f(x, y) for surface
        # Input: ...x2, Output: ...x1
        self.height_net = tcnn.Network(
            n_input_dims=tot_out_dims_2d,
            n_output_dims=1,
            network_config={
                "otype": "CutlassMLP",
                "activation": "ReLU",   
                "output_activation": "None",
                "n_neurons": 256,
                "n_hidden_layers": 3,
            },
        )

        # 2D DINO network
        # self.dino_net = tcnn.Network(
        #     n_input_dims=tot_out_dims_2d,
        #     n_output_dims=384,
        #     network_config={
        #         "otype": "CutlassMLP",
        #         "activation": "ReLU",
        #         "output_activation": "None",
        #         "n_neurons": 256,
        #         "n_hidden_layers": 1,
        #     },
        # )

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

    def set_enu_transform(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_density(self, ray_samples: RaySamples, do_heightcap=True) -> Tuple[Tensor, Tensor]:
        """Computes and returns the densities
        
        """
        if self.spatial_distortion is not None:
            unnorm_positions = ray_samples.frustums.get_positions()
            positions = self.spatial_distortion(unnorm_positions)  # Spatial distortion squeezes into -2 to 2 box
            positions = (positions + 2.0) / 4.0                    # Normalize to 0 to 1
        else:
            unnorm_positions = ray_samples.frustums.get_positions()
            positions = SceneBox.get_normalized_positions(unnorm_positions, self.aabb)  # NOTE: what does this do?
        
        if do_heightcap:
            xs = [e(positions.view(-1, 3)[:, :2]) for e in self.encs2d]
            x = torch.concat(xs, dim=-1)

            heightcaps = self.height_net(x).view(*ray_samples.frustums.shape)
            # positions_2d = positions[..., :2]
            # encoded = self.nemo.encoder(positions_2d.view(-1, 2))
            # heightcaps = self.nemo.decoder(encoded).view(*ray_samples.frustums.shape)
        else:
            heightcaps = 10000.0 

        selector_0 = (unnorm_positions[..., 2] <= -0.25)
        # Selector to mask out positions with z higher than heightcaps
        selector_0 = selector_0 & (unnorm_positions[..., 2] <= heightcaps)
        
        # ------ Standard Nerfstudio masking ------ #
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
        # ------ Standard Nerfstudio masking ------ #
        
        return density, base_mlp_out

    
    def get_heightcap_outputs(
        self, ray_samples: RaySamples) -> Dict[FieldHeadNames, Tensor]:
        outputs = {}

        positions = ray_samples.frustums.get_positions().detach()  # (N_rays, N_samples, 3)
        positions_2d = positions[..., :2]                          # (N_rays, N_samples, 2)    
        outputs["heightcap"] = self.positions_to_heights(positions_2d)
        # outputs["heightcap"] = self.nemo(positions_2d.view(-1, 2)).view(*positions.shape[:-1], -1)
        
        return outputs
    

    def positions_to_heights(self, positions):
        """Get heights from 2D positions
        
        positions : torch.Tensor (..., 2)
            Tensor of 2D positions 
        
        """
        inp_shape = positions.shape
        positions = self.spatial_distortion(positions)  
        positions = (positions + 2.0) / 4.0

        xs = [e(positions.view(-1, 2)) for e in self.encs2d]
        x = torch.concat(xs, dim=-1)

        heights = self.height_net(x).view(*inp_shape[:-1], -1)
        
        return heights
    
    
    # def get_dino_outputs(
    #     self, ray_samples: RaySamples) -> Dict[FieldHeadNames, Tensor]:
    #     outputs = {}

    #     positions = ray_samples.frustums.get_positions().detach()  # (N_rays, N_samples, 3)
    #     positions_2d = positions[..., :2]                          # (N_rays, N_samples, 2)
    #     outputs["dino"] = self.positions_to_dino(positions_2d)

    #     return outputs
    
    
    # def positions_to_dino(self, positions):
    #     """Get DINO features from 2D positions
        
    #     positions : torch.Tensor (..., 2)
    #         Tensor of 2D positions
        
    #     """
    #     inp_shape = positions.shape
    #     positions = self.spatial_distortion(positions)
    #     positions = (positions + 2.0) / 4.0

    #     xs = [e(positions.view(-1, 2)) for e in self.encs2d]
    #     x = torch.concat(xs, dim=-1)

    #     dino = self.dino_net(x).view(*inp_shape[:-1], -1)
        
    #     return dino