"""
Terrain Nerfacto Model File

Currently this subclasses the Nerfacto model. Consider subclassing from the base Model.
"""
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type

import nerfacc
import numpy as np
import torch
from terrain_nerfacto.field import TNerfField

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
class TNerfModelConfig(NerfactoModelConfig):
    """TNerf Model Configuration.
    """

    _target: Type = field(default_factory=lambda: TNerfModel)
    num_dino_samples: int = 24
    hashgrid_layers: Tuple[int] = (12, 12)
    hashgrid_resolutions: Tuple[Tuple[int]] = ((16, 128), (128, 512))
    hashgrid_sizes: Tuple[int] = (19, 19)


class TNerfModel(NerfactoModel):
    """Terrain NeRF Model."""

    config: TNerfModelConfig

    def set_enu_transform(self, *args, **kwargs):
        self.field.set_enu_transform(*args, **kwargs)

    def populate_modules(self):
        super().populate_modules()

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))
        
        # Fields
        self.field = TNerfField(
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

        self.tall_loss_factor = 1.0
        self.max_height = 1.0
        self.quantile_frac = 0.95
        self.ground_forget_fac = 0.1
        
    def get_outputs(self, ray_bundle: RayBundle):
        """Volume rendering
        
        
        """
        ray_samples: RaySamples
        # Importance sampling from proposal distribution
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)

        # ==== added for heightcap ==== #
        z_pred = self.field.get_heightcap_outputs(ray_samples)
        z_sample = ray_samples.frustums.get_positions().detach()[..., 2][..., None]

        # Use full density field when computing height penalty
        height_density, _ = self.field.get_density(ray_samples, do_heightcap=False)
        height_weights = ray_samples.get_weights(height_density.detach())
        # ==== added for heightcap ==== #

        # Push the ray samples through the field
        field_outputs = self.field.forward(ray_samples, compute_normals=self.config.predict_normals)

        # Standard gradient scaling
        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        # Volume rendering
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

        # Predict normal vectors for each ray in camera frame
        if self.config.predict_normals:
            normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
            pred_normals = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
            outputs["normals"] = self.normals_shader(normals)
            outputs["pred_normals"] = self.normals_shader(pred_normals)

        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        # Predicted normals supervision
        if self.training and self.config.predict_normals:
            outputs["rendered_orientation_loss"] = orientation_loss(
                weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
            )

            outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                weights.detach(),
                field_outputs[FieldHeadNames.NORMALS].detach(),
                field_outputs[FieldHeadNames.PRED_NORMALS],
            )

        # Render depth 
        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        # ==== added for DINO field ==== #
        # dino_weights, best_ids = torch.topk(weights, self.config.num_dino_samples, dim=-2, sorted=False)

        # def gather_fn(tens):
        #     return torch.gather(tens, -2, best_ids.expand(*best_ids.shape[:-1], tens.shape[-1]))

        # dataclass_fn = lambda dc: dc._apply_fn_to_fields(gather_fn, dataclass_fn)
        # dino_samples: RaySamples = ray_samples._apply_fn_to_fields(gather_fn, dataclass_fn)
        
        # dino_field_outputs = self.field.get_dino_outputs(dino_samples)

        # outputs["dino"] = torch.sum(dino_weights * dino_field_outputs["dino"], dim=-2)
        # ==== added for DINO field ==== #

        # Use depth to project to xyz point
        # depth = outputs["depth"]

        # ray_origins = ray_bundle.origins
        # ray_directions = ray_bundle.directions

        # # Project to xyz
        # xyz = ray_origins + ray_directions * depth
        # #print("xyz shape: ", xyz.shape)

        # pred_z = self.field.get_heights(xyz)
        # #print("pred_z shape: ", pred_z.shape, "z shape: ", xyz[:, 2].shape)
        # outputs["height_penalty"] = torch.mean((pred_z - xyz[:, 2])**2)
        # #print("height_penalty: ", outputs["height_penalty"])

        
        # Heightcap

        # Soft penalty for height exceeding the heightcap: y = max(0, height - x) + (1 - quantile_frac)*x
        error = z_sample - z_pred
        heightcap_penalty = torch.max((self.quantile_frac - 1) * error, self.quantile_frac * error)
        # TODO: adjust the quantile frac such that most points below have density, but points above don't
        #ground_penalty = torch.square(ground_height - torch.min(heightcap_field_outputs["heightcap"]))

        if self.training:
            outputs["height_penalty"] = torch.sum(height_weights.detach() * heightcap_penalty, dim=-2)
            #outputs["ground_penalty"] = torch.sum(height_weights.detach() * ground_penalty, dim=-2)
        
        # Hard penalty for height exceeding the camera height
        outputs["heightcap_net_output"] = torch.sum(height_weights.detach() * z_pred, dim=-2)
        # outputs["height"] = z_pred

        ## ==== SMOOTHNESS LOSS ==== ##
        # Compute heightnet spatial derivatives
        # - Sample some xy points, for each xy point, consider a small delta in x and y and compute the difference in height

        # TODO: if autograd 2nd derivative is not working, use finite differences (Laplacian kernel)
        # if self.training:
        #     #print("z_pred shape: ", z_pred.shape)
        #     delta = 1e-4
        #     positions = ray_samples.frustums.get_positions().detach().clone()
        #     xy = positions[..., :2]
        #     xy = xy.reshape(-1, 2)

        #     laplacian = 4 * z_pred.view(-1)

        #     for dxy in [torch.tensor([-1, 0], device=xy.device), torch.tensor([1, 0], device=xy.device), 
        #                 torch.tensor([0, -1], device=xy.device), torch.tensor([0, 1], device=xy.device)]:
        #         xy_delta = xy + dxy * delta
        #         z_pred_delta = self.field.positions_to_heights(xy_delta)
        #         laplacian -= z_pred_delta.view(-1)

        #     laplacian /= delta
        #     outputs["height_laplacian"] = laplacian
        
        return outputs
    

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)

        if self.training:
            pass
            #loss_dict["heightcap_loss"] = 1.0 * outputs["height_penalty"]

            # Add height opacity loss by its average
            loss_dict["height_opacity_loss"] = self.tall_loss_factor * outputs["height_penalty"].sum(dim=-1).nanmean()
            #loss_dict["ground_opacity_loss"] = self.tall_loss_factor * outputs["ground_penalty"].sum(dim=-1).nanmean()
            
            # Enforce heightnet smoothness loss
            # Temperature: starts at 0 and increases to 1
            #smoothness_loss = (1.0 - np.exp(-self.step*1e-10)) * torch.nanmean(torch.square(outputs["height_grad_dx"]) + torch.square(outputs["height_grad_dy"]))
            #smoothness_loss = 1e-3 * torch.nanmean(torch.square(outputs["heightnet_dx"]) + torch.square(outputs["heightnet_dy"]))
            #smoothness_loss = 1e-5 * torch.nanmean(torch.square(outputs["height_laplacian"]))
            #smoothness_loss = (1.0 - np.exp(-self.step*1e-10)) * torch.nanmean(torch.square(outputs["height_laplacian"]))
            #loss_dict["height_smoothness_loss"] = smoothness_loss

            # dino_wt = 1.0 - np.exp(-self.step * 1e-10)  
            # unreduced_dino = torch.nn.functional.mse_loss(outputs["dino"], batch["dino"], reduction="none")
            # loss_dict["dino_loss"] = dino_wt * unreduced_dino.sum(dim=-1).nanmean()
        
        return loss_dict