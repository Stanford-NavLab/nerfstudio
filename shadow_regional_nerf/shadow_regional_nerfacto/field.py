"""
Shadow Regional Nerfacto Field

Currently this subclasses the NerfactoField. Consider subclassing the base Field.
"""

from typing import Dict, Literal, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

# from nerfstudio.cameras.rays import RaySamples
# from nerfstudio.data.scene_box import SceneBox
# from nerfstudio.field_components.activations import trunc_exp
# from nerfstudio.field_components.field_heads import FieldHeadNames
# from nerfstudio.field_components.spatial_distortions import SpatialDistortion
# from nerfstudio.fields.base_field import Field  # for custom Field
# from nerfstudio.fields.nerfacto_field import \
#     NerfactoField  # for subclassing NerfactoField

from minimal_regional_nerfacto.field import MRNerfField


class SRNerfField(MRNerfField):
    """
    Shadow Regional Nerf Field
    """

    def __init__(
        self, 
        # grid_resolutions,
        # grid_layers,
        # grid_sizes,
        **kwargs) -> None:
        print("---------------------------")
        print("[SRNeRF Field] STARTING SUPER INIT FIELD")

        # I get the impression that grid_* are left over and
        # can be removed
        # super().__init__(
        #     grid_resolutions,
        #     grid_layers,
        #     grid_sizes,
        #     **kwargs)

        super().__init__(**kwargs)

        self.top_cutoff = 1.0

        print("---------------------------")
        print("[SRNeRF Field] FINISHED FIELD")

    def set_enu_transform(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
