"""
Terrain Nerfacto Config
"""

from __future__ import annotations

from terrain_nerfacto.datamanager import TNerfDataManagerConfig
from terrain_nerfacto.model import TNerfModelConfig
from terrain_nerfacto.pipeline import TNerfPipelineConfig

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager, VanillaDataManagerConfig)
from nerfstudio.data.dataparsers.nerfstudio_dataparser import \
    NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import (AdamOptimizerConfig,
                                          RAdamOptimizerConfig)
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.plugins.types import MethodSpecification

terrain_nerfacto = MethodSpecification(
    config=TrainerConfig(
        method_name="terrain-nerfacto",
        steps_per_eval_batch=500,
        steps_per_save=600,
        max_num_iterations=10000,
        mixed_precision=True,
        pipeline=TNerfPipelineConfig(
            datamanager=TNerfDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(train_split_fraction=0.98),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                # camera_optimizer=CameraOptimizerConfig(
                #     mode="SO3xR3",
                #     optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2),
                #     scheduler=ExponentialDecaySchedulerConfig(lr_final=6e-6, max_steps=200000),
                # ),
            ),
            model=TNerfModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                hashgrid_sizes=(19,),
                hashgrid_layers=(16,),
                hashgrid_resolutions=((16, 512),),
                num_dino_samples=12,
                camera_optimizer=CameraOptimizerConfig(mode="SO3xR3")
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "fields": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=200000),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=6e-6, max_steps=200000),
            }
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Terrain Nerf method.",
)