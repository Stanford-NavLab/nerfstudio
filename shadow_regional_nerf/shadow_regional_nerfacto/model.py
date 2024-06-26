"""
Shadow Regional Nerfacto Model File

Currently this subclasses the Nerfacto model. Consider subclassing from the base Model.
"""
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type

# import nerfacc
# import numpy as np
import torch
from shadow_regional_nerfacto.field import SRNerfField

from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames

from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.viewer.viewer_elements import ViewerText, ViewerButton, ViewerCheckbox


# from nerfstudio.model_components.losses import (
#     MSELoss, distortion_loss, interlevel_loss, orientation_loss,
#     pred_normal_loss, scale_gradients_by_distance_squared)
# from nerfstudio.models.base_model import Model, ModelConfig  # for custom Model
# from nerfstudio.models.nerfacto import (  # for subclassing Nerfacto model
#     NerfactoModel, NerfactoModelConfig)
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
        super(MRNerfModel, self).populate_modules()

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))
        
        # Fields
        self.field = SRNerfField(
            # grid_resolutions=self.config.hashgrid_resolutions,
            # grid_layers=self.config.hashgrid_layers,
            # grid_sizes=self.config.hashgrid_sizes,
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

        # For plotting the lat/lon values with ENU masked
        self.latlon_reader = ViewerText("Lat, Lon", 
                                        "", 
                                        cb_hook=self.latlon_cb,
                                        hint="Lat/Lon to Highlight")
        self.latlon_str = None
        self.latlon_setter = ViewerButton(name="Set Lat/Lon", cb_hook=self.latlon_set)

        # This will get set when Pipeline calls "set_enu_transforms"
        self.nerf_from_enu_coords = None

        # For operations with the satellites
        self.satellite_cos_angular = math.cos(math.radians(5))
        # Not sure if we should set a device at this point
        self.satellite_directions = torch.tensor(
                [[1.0, 0.0, 0.0], 
                 [0.0, 1.0, 0.0],
                 [0.11043, 0.99388, 0.0],
                 [0.936329, 0.0, -0.351123]])

        # For plotting the satellites
        self.output_satellites = False
        self.use_satellites_reader = ViewerCheckbox("Show Satellites", 
                                                    self.output_satellites, 
                                                    cb_hook=self.satellites_cb, 
                                                    hint="Overlay the satellites")

    def satellites_cb(self, element):
        """
        Waits for toggle

        Parameters:
        --------------
        element
            Viewer element (e.g., the button)
        """
        # Change the button text box value
        
        self.output_satellites = element.value


    def get_outputs(self, ray_bundle: RayBundle):
        """
        This is the inference of the NeRF per ray.

        Parameters
        --------------
        ray_bundle
            The rays that we want to accumulate the field of.
        """
        
        #############
        # Standard NeRF

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


        #############
        # Shadow / NLOS Features

        with torch.no_grad():
            outputs["max_density"], _ = torch.max(field_outputs[FieldHeadNames.DENSITY], dim=1)
            outputs["max_density_log10"] = torch.log10(outputs["max_density"])
            outputs["sum_density"] = torch.sum(field_outputs[FieldHeadNames.DENSITY], dim=1)
            outputs["sum_density_log10"] = torch.log10(outputs["sum_density"])

            # print("From Max Density", torch.min(outputs["max_density"]), torch.max(outputs["max_density"]))
            # print(f"From Max Density: ({torch.min(outputs["max_density"])}, {torch.max(outputs["max_density"])})")
            # print(f"From Sum Density: ({torch.min(outputs["sum_density"])}, {torch.max(outputs["sum_density"])})")

        #############
        # Minimal Regional NeRF Part (Manually inherited)

        # If nerf_from_enu_coords is not None, visualize the nerf_from_enu_coords 
        # (i.e., user query point)
        if self.nerf_from_enu_coords is not None:
            xy = ray_samples.frustums.get_positions()[..., :2].detach()
            
            # Convert nerf_from_enu_coords to same shape as xy
            nerf_from_enu_coords = self.nerf_from_enu_coords[:2].unsqueeze(0).expand_as(xy)

            # If xy is within epsilon of the NeRF ENU coords, set mask to 1 else 0
            cylinder_radius = 0.05
            lla_mask = (torch.norm(xy - nerf_from_enu_coords, dim=-1) < cylinder_radius).unsqueeze(-1)
            
            outputs["enu_vis_masked"] = 0.5*torch.sum(weights * lla_mask, dim=-2) + 0.5*rgb

        #############
        # Satellites

        if self.output_satellites:
            with torch.no_grad():
                # The directions can be between -1 and 1. We want to convert this to 
                # 0 to 1. Hence, +1 to get 0 to 2 and /2 to get 0 to 1
                outputs["directions"] = (ray_bundle.directions + 1) / 2

                # We expand the axes to match [# rays, # sats, 3D]
                # So, the rays will be        [# rays,      1,  3]
                # and the satellites will be  [     1, # sats,  3]
                # We then want to do a dot product (element-wise multiply & sum) over 
                # the direction (i.e., 3D, which is the last axis).
                # This leaves [# rays, # sats]
                # Lastly, we check if any satellites will mask the ray and hence
                # reduce the satellite dimension, leaving [# rays, 1] with keepdim
                sat_cos_angular = 0.98
                sat_bool_mask = torch.any(
                    torch.sum(
                        ray_bundle.directions[:, None, :] * self.satellite_directions[None, :, :], 
                        dim=-1) > self.satellite_cos_angular,
                    dim=-1, keepdim=True)

                outputs["satellite_only"] = sat_bool_mask.to(accumulation.dtype)


        
        
        return outputs
