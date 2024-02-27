"""
Minimal Regional Nerfacto Model File

Currently this subclasses the Nerfacto model. Consider subclassing from the base Model.
"""
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type

# Geospatial conversions
import pymap3d as pm

import nerfacc
import numpy as np
import torch
from minimal_regional_nerfacto.field import MRNerfField

from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
# from nerfstudio.model_components.losses import (
#     MSELoss, distortion_loss, interlevel_loss, orientation_loss,
#     pred_normal_loss, scale_gradients_by_distance_squared)
from nerfstudio.models.base_model import Model, ModelConfig  # for custom Model
from nerfstudio.models.nerfacto import (  # for subclassing Nerfacto model
    NerfactoModel, NerfactoModelConfig)
from nerfstudio.viewer.viewer_elements import ViewerText, ViewerButton
from minimal_regional_nerfacto.utils.geodetic_utils import geodetic_to_enu


@dataclass
class MRNerfModelConfig(NerfactoModelConfig):
    """MRNerf Model Configuration.
    """

    _target: Type = field(default_factory=lambda: MRNerfModel)
    hashgrid_layers: Tuple[int] = (12, 12)
    hashgrid_resolutions: Tuple[Tuple[int]] = ((16, 128), (128, 512))
    hashgrid_sizes: Tuple[int] = (19, 19)


class MRNerfModel(NerfactoModel):
    """Minimal Regional NeRF Model."""

    config: MRNerfModelConfig

    def set_enu_transform(self, *args, **kwargs):
        self.field.set_enu_transform(*args, **kwargs)

    def populate_modules(self):
        super().populate_modules()

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))
        
        # Fields
        self.field = MRNerfField(
            grid_resolutions=self.config.hashgrid_resolutions,
            grid_layers=self.config.hashgrid_layers,
            grid_sizes=self.config.hashgrid_sizes,
            aabb=self.scene_box.aabb,
            hidden_dim=self.config.hidden_dim,
            num_levels=self.config.num_levels,
            max_res=self.config.max_res,
            base_res=self.config.base_res,
            features_per_level=self.config.features_per_level,
            log2_hashmap_size=self.config.log2_hashmap_size,
            hidden_dim_color=self.config.hidden_dim_color,
            hidden_dim_transient=self.config.hidden_dim_transient,
            spatial_distortion=scene_contraction,
            num_images=self.num_train_data,
            use_pred_normals=self.config.predict_normals,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            appearance_embedding_dim=self.config.appearance_embed_dim,
            implementation=self.config.implementation,
        )

        self.latlon_reader = ViewerText("Lat, Lon", "", cb_hook=self.latlon_cb)
        self.latlon_str = None
        self.latlon_setter = ViewerButton(name="Set Lat/Lon", cb_hook=self.latlon_set)

        self.nerf_from_enu_coords = None

    def latlon_set(self, element):
        """
        Call back when user _sets_ the latitude and longitude in the viewer.
        This is used for later visualizations.

        Parameters:
        --------------
        element
            Viewer element (e.g., the button)
        """

        if self.latlon_str is not None:
            latlon_separated = self.latlon_str.split(",")

            if len(latlon_separated) >= 2:
                lat = float(latlon_separated[0])
                lon = float(latlon_separated[1])
                rclat = self.field.center_latlon[0]
                rclon = self.field.center_latlon[1]
                rcalt = self.field.center_usgs_height

                # OLD VERSION
                # # Convert latlon into ENU
                # r_earth = 6378137.0 # (m)
                # east, north, up = geodetic_to_enu(lat, lon, r_earth, rclat, rclon, r_earth)
                # # Convert to (1x3) tensor
                # enu = torch.tensor([east, north, up])[None, :].to(self.device)
                # # Convert ENU into NeRF coordinates
                # self.nerf_from_enu_coords = self.field.enu2nerf_points(enu).reshape(-1)

                # NEW VERSION
                # Convert latlon into ENU as a (1x3) torch tensor
                enu = torch.tensor(pm.geodetic2enu(lat, lon, rcalt, 
                                                   rclat, rclon, rcalt)).float().view(1, 3).to(self.device)
                # Convert ENU into NeRF coordinates
                self.nerf_from_enu_coords = self.field.enu2nerf_points(enu).reshape(-1)

                # Print everything
                print("Query Point lat, lon: ", lat, lon)
                print("NeRF Center lat, lon: ", rclat, rclon)

                print("ENU (m)   : ", enu)
                print("NeRF coord: ", self.nerf_from_enu_coords)
                
    
    def latlon_cb(self, element):
        """
        Waits for the set callback

        Parameters:
        --------------
        element
            Viewer element (e.g., the button)
        """
        # Change the button text box value
        self.latlon_str = element.value
        
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
