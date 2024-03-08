"""
Shadow Regional Nerfacto Model File

Currently this subclasses the Nerfacto model. Consider subclassing from the base Model.
"""
# import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type

# import nerfacc
# import numpy as np
import torch
from shadow_regional_nerfacto.field import SRNerfField

from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames

# from nerfstudio.field_components.spatial_distortions import SceneContraction
# from nerfstudio.model_components.losses import (
#     MSELoss, distortion_loss, interlevel_loss, orientation_loss,
#     pred_normal_loss, scale_gradients_by_distance_squared)
# from nerfstudio.models.base_model import Model, ModelConfig  # for custom Model
# from nerfstudio.models.nerfacto import (  # for subclassing Nerfacto model
#     NerfactoModel, NerfactoModelConfig)
# from nerfstudio.viewer.viewer_elements import ViewerText, ViewerButton
# from minimal_regional_nerfacto.utils.geodetic_utils import geodetic_to_enu

from minimal_regional_nerfacto.model import MRNerfModel, MRNerfModelConfig


@dataclass
class SRNerfModelConfig(MRNerfModelConfig):
    """SRNerf Model Configuration.
    """

    _target: Type = field(default_factory=lambda: SRNerfModel)
    hashgrid_layers: Tuple[int] = (12, 12)
    hashgrid_resolutions: Tuple[Tuple[int]] = ((16, 128), (128, 512))
    hashgrid_sizes: Tuple[int] = (19, 19)


class SRNerfModel(MRNerfModel):
    """Shadow Regional NeRF Model."""

    config: SRNerfModelConfig

    def populate_modules(self):
        super().populate_modules()
        
    def get_outputs(self, ray_bundle: RayBundle):
        """
        This is the inference of the NeRF per ray.

        Parameters
        --------------
        ray_bundle
            The rays that we want to accumulate the field of.
        """
        ray_samples: RaySamples
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        field_outputs = self.field.forward(ray_samples, compute_normals=self.config.predict_normals)
        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        with torch.no_grad():
            depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        expected_depth = self.renderer_expected_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "expected_depth": expected_depth,
        }

        if self.config.predict_normals:
            normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
            pred_normals = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
            outputs["normals"] = self.normals_shader(normals)
            outputs["pred_normals"] = self.normals_shader(pred_normals)
        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        if self.training and self.config.predict_normals:
            outputs["rendered_orientation_loss"] = orientation_loss(
                weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
            )

            outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                weights.detach(),
                field_outputs[FieldHeadNames.NORMALS].detach(),
                field_outputs[FieldHeadNames.PRED_NORMALS],
            )

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        # If nerf_from_enu_coords is not None, visualize the nerf_from_enu_coords (i.e., user query point)
        if self.nerf_from_enu_coords is not None:
            xy = ray_samples.frustums.get_positions()[..., :2].detach()
            
            # Convert nerf_from_enu_coords to same shape as xy
            nerf_from_enu_coords = self.nerf_from_enu_coords[:2].unsqueeze(0).expand_as(xy)
            
            # If xy is within epsilon of dino_coords, set mask to 1 else 0
            cylinder_radius = 0.05
            mask = (torch.norm(xy - nerf_from_enu_coords, dim=-1) < cylinder_radius).unsqueeze(-1)
            
            outputs["enu_vis_masked"] = 0.5*torch.sum(weights * mask, dim=-2) + 0.5*rgb
        
        return outputs
