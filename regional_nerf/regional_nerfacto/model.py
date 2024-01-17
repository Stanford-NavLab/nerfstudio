"""
Regional Nerfacto Model File

Currently this subclasses the Nerfacto model. Consider subclassing from the base Model.
"""
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type

import nerfacc
import numpy as np
import torch
from regional_nerfacto.field import RNerfField

from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.model_components.losses import (
    MSELoss, distortion_loss, interlevel_loss, orientation_loss,
    pred_normal_loss, scale_gradients_by_distance_squared)
from nerfstudio.models.base_model import Model, ModelConfig  # for custom Model
from nerfstudio.models.nerfacto import (  # for subclassing Nerfacto model
    NerfactoModel, NerfactoModelConfig)


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


@dataclass
class RNerfModelConfig(NerfactoModelConfig):
    """RNerf Model Configuration.
    """

    _target: Type = field(default_factory=lambda: RNerfModel)
    num_lerf_samples: int = 24
    hashgrid_layers: Tuple[int] = (12, 12)
    hashgrid_resolutions: Tuple[Tuple[int]] = ((16, 128), (128, 512))
    hashgrid_sizes: Tuple[int] = (19, 19)


class RNerfModel(NerfactoModel):
    """Regional NeRF Model."""

    config: RNerfModelConfig

    def set_enu_transform(self, *args, **kwargs):
        self.field.set_enu_transform(*args, **kwargs)

    def populate_modules(self):
        super().populate_modules()

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))
        
        # Fields
        self.field = RNerfField(
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

        self.tall_loss_factor = 0.01
        self.max_height = 1.0
        self.quantile_frac = 0.9

    def get_outputs(self, ray_bundle: RayBundle):
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
        
        lerf_weights, best_ids = torch.topk(weights, self.config.num_lerf_samples, dim=-2, sorted=False)

        def gather_fn(tens):
            return torch.gather(tens, -2, best_ids.expand(*best_ids.shape[:-1], tens.shape[-1]))

        dataclass_fn = lambda dc: dc._apply_fn_to_fields(gather_fn, dataclass_fn)
        lerf_samples: RaySamples = ray_samples._apply_fn_to_fields(gather_fn, dataclass_fn)
        
        lerf_field_outputs = self.field.get_dino_outputs(lerf_samples)

        outputs["dino"] = torch.sum(lerf_weights * lerf_field_outputs["dino"], dim=-2)
        
        heightcap_field_outputs = self.field.get_heightcap_outputs(ray_samples)
        height = ray_samples.frustums.get_positions()[..., 2][..., None]

        height_density, _ = self.field.get_density(ray_samples, do_heightcap=False)
        height_weights = ray_samples.get_weights(height_density.detach())

        # Soft penalty for height exceeding the heightcap
        heightcap_penalty = torch.relu(height - heightcap_field_outputs["heightcap"]) + (1.0 - self.quantile_frac)*heightcap_field_outputs["heightcap"]
        outputs["height_penalty"] = torch.sum(height_weights * heightcap_penalty, dim=-2)

        outputs["heightcap_net_output"] = torch.sum(height_weights * heightcap_field_outputs["heightcap"], dim=-2)

        # TODO: compute Jacobian of heightnet wrt. points while retaining computational graph
        # add to outputs to use in loss function

        # Map rendering
        enu_positions = self.field.nerf2enu_points(ray_samples.frustums.get_positions())[..., :2]

        osm_scale = self.field.osm_scale
        osm_positions = enu_positions / osm_scale
        osm_positions = osm_positions + torch.tensor([self.field.osm_image.shape[1]/2, self.field.osm_image.shape[0]/2], device=osm_positions.device)[None, None, :]
        osm_positions = osm_positions.long()
        osm_positions = torch.clamp(osm_positions, 0, self.field.osm_image.shape[0]-1)

        # Access the RGB value for each position
        rgb_values = self.field.osm_image[:, osm_positions[..., 1], osm_positions[..., 0]]  # Access the RGB value for each position

        rgb_values = rgb_values.permute(1, 2, 0)  # Move the RGB dimension to the front
        # print(field_outputs[FieldHeadNames.RGB].shape, rgb_values.shape)

        outputs["osm_rgb"] = self.renderer_rgb(rgb=rgb_values, weights=weights)

        # Compute heightnet spatial derivatives
        positions = ray_samples.frustums.get_positions().detach().clone()
        xy = positions[..., :2]
        init_shape = xy.shape
        xy = xy.reshape(-1, 2)
        h_xy = torch.zeros_like(xy)
        delta = 1e-4
        height_vals_cur = self.field.positions_to_heights(xy)
        for i in range(2):
            delta_vec = torch.zeros_like(xy)
            delta_vec[:, i] = delta
            height_vals_pos = self.field.positions_to_heights(xy + delta_vec)
            #height_vals_neg = self.field.positions_to_heights(xy - delta_vec)
            h_xy[:, i] = (height_vals_pos - height_vals_cur).reshape(-1) / (delta)
            height_vals_pos.detach()
        h_xy = h_xy.reshape(*init_shape)

            # x = positions[..., 0]
            # y = positions[..., 1]
            # print(x.requires_grad, y.requires_grad)
            # height_vals = self.field.xy_to_heights(x, y)
            # print(height_vals.requires_grad)
            # h_x = gradient(height_vals, x)
            # h_y = gradient(height_vals, y)
            # outputs["heightnet_spatial_derivatives"] = torch.cat([h_x, h_y], dim=-1)
        outputs["heightnet_dx"] = h_xy[..., 0]
        outputs["heightnet_dy"] = h_xy[..., 1] 
            #print("h_xy shape: ", h_xy.shape)
        
        return outputs
    
    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)

        if self.training:
            dino_wt = 1.0-np.exp(-self.step*1e-10)
            unreduced_dino = torch.nn.functional.mse_loss(outputs["dino"], batch["dino"], reduction="none")
            loss_dict["dino_loss"] = dino_wt*unreduced_dino.sum(dim=-1).nanmean()

        # Add height opacity loss by its average
        # TODO: NAVLAB
            loss_dict["height_opacity_loss"] = self.tall_loss_factor * outputs["height_penalty"].sum(dim=-1).nanmean()

            # TODO: add heightnet smoothness (Jacobian) loss
            #loss_dict["height_smoothness_loss"] = 1.0 * outputs["heightnet_spatial_derivatives"].sum(dim=-1).nanmean()
            #smoothness_loss = 1.0 * outputs["heightnet_spatial_derivatives"].sum(dim=-1).nanmean()
            # smoothness_loss = 1.0 * torch.sum(torch.norm(outputs["heightnet_spatial_derivatives"][..., :2], dim=-1))
            # smoothness_loss = 0.001 * torch.nanmean(torch.square(outputs["heightnet_dx"]) + torch.square(outputs["heightnet_dy"]))
            # print("smoothness loss: ", smoothness_loss)
            # loss_dict["height_smoothness_loss"] = smoothness_loss
        
        return loss_dict