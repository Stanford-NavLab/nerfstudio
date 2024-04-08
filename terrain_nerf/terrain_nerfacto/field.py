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

        self.top_cutoff = 1.0   # Assume no density above cameras

        self.height_net_arch = 'MLP'  # 'SIREN' or 'MLP'

        if self.height_net_arch == 'SIREN':
            self.height_net = Siren(in_features=2, out_features=1, hidden_features=256,
                                    hidden_layers=3, outermost_linear=True)
        elif self.height_net_arch == 'MLP':
            # 2D positional encodings
            self.encs2d = torch.nn.ModuleList(
                [
                    TNerfField._get_encoding(
                        grid_resolutions[i][0], grid_resolutions[i][1], grid_layers[i], indim=2, hash_size=grid_sizes[i]
                    )
                    for i in range(len(grid_layers))
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
                    "n_hidden_layers": 1,
                },
            )
            # self.height_net = Siren(in_features=tot_out_dims_2d, out_features=1, hidden_features=256,
            #                         hidden_layers=1, outermost_linear=True)

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

    def set_enu_transform(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_density(self, ray_samples: RaySamples, do_heightcap=True) -> Tuple[Tensor, Tensor]:
        """Computes and returns the densities
        
        """
        if self.spatial_distortion is not None:
            unnorm_positions = ray_samples.frustums.get_positions()
            positions = self.spatial_distortion(unnorm_positions)  # Spatial distortion squeezes into -2 to 2
            positions = (positions + 2.0) / 4.0                    # Normalize to 0 to 1
        else:
            unnorm_positions = ray_samples.frustums.get_positions()
            positions = SceneBox.get_normalized_positions(unnorm_positions, self.aabb)  # NOTE: what does this do?
        
        if do_heightcap:
            if self.height_net_arch == 'SIREN':
                x = positions.view(-1, 3)[:, :2]
                heightcaps, coords = self.height_net(x)
                heightcaps = heightcaps.view(*ray_samples.frustums.shape)
            elif self.height_net_arch == 'MLP':
                xs = [e(positions.view(-1, 3)[:, :2]) for e in self.encs2d]
                x = torch.concat(xs, dim=-1)

                heightcaps = self.height_net(x).view(*ray_samples.frustums.shape)
            ground_height = self.ground_height
        else:
            heightcaps = 10000.0 
            ground_height = -10000.0

        selector_0 = (unnorm_positions[..., 2] <= -0.25)
        # Selector to mask out positions with z higher than heightcaps
        selector_0 = selector_0 & (unnorm_positions[..., 2] <= heightcaps)
        #selector_0 = (unnorm_positions[..., 2] <= heightcaps) & (unnorm_positions[..., 2] >= ground_height) # Navlab added
        #selector_0 = True
        
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
    
    
    # TODO: can we consolidate this with get_density?
    def get_heightcap_outputs(
        self, ray_samples: RaySamples) -> Dict[FieldHeadNames, Tensor]:
        outputs = {}

        positions = ray_samples.frustums.get_positions().detach()
        positions = self.spatial_distortion(positions)
        positions = (positions + 2.0) / 4.0

        if self.height_net_arch == 'SIREN':
            x = positions.view(-1, 3)[:, :2]
            heightcap_pass, coords = self.height_net(x)
            heightcap_pass = heightcap_pass.view(*ray_samples.frustums.shape, -1)
        elif self.height_net_arch == 'MLP':
            xs = [e(positions.view(-1, 3)[:, :2]) for e in self.encs2d]
            x = torch.concat(xs, dim=-1)

            heightcap_pass = self.height_net(x).view(*ray_samples.frustums.shape, -1)

        outputs["heightcap"] = heightcap_pass
        
        return outputs
    

    def get_heights(self, positions):
        positions = self.spatial_distortion(positions) # -2 to 2
        positions = positions / 2.0            # -1 to 1

        if self.height_net_arch == 'SIREN':
            x = positions.view(-1, 3)[:, :2]
            heightcap_pass, coords = self.height_net(x)
            #heightcap_pass = heightcap_pass.view(*ray_samples.frustums.shape, -1)
        elif self.height_net_arch == 'MLP':
            xs = [e(positions.view(-1, 3)[:, :2]) for e in self.encs2d]
            x = torch.concat(xs, dim=-1)

            heightcap_pass = self.height_net(x).view(*ray_samples.frustums.shape, -1)

        return heightcap_pass

    
    def get_ground_height(self):
        return self.ground_height


    # TODO: clean-up / consolidate these methods
    def positions_to_heights(self, positions):
        """Positions are 2D for now"""
        inp_shape = positions.shape
        positions = torch.cat([positions, torch.zeros_like(positions[..., :1])], dim=-1)
        positions = self.spatial_distortion(positions)
        positions = (positions + 2.0) / 4.0

        if self.height_net_arch == 'SIREN':
            x = positions.view(-1, 3)[:, :2]
            heights, coords = self.height_net(x)
            heights = heights.view(*inp_shape[:-1], -1)
        elif self.height_net_arch == 'MLP':
            xs = [e(positions.view(-1, 3)[:, :2]) for e in self.encs2d]
            x = torch.concat(xs, dim=-1)

            heights = self.height_net(x).view(*inp_shape[:-1], -1)
        
        return heights