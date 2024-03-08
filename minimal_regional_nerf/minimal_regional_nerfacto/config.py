"""
Minimal Regional Nerfacto Config
"""

from __future__ import annotations

from minimal_regional_nerfacto.model import MRNerfModelConfig
from minimal_regional_nerfacto.datamanager import MRNerfDataManagerConfig
from minimal_regional_nerfacto.pipeline import MRNerfPipelineConfig

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager, VanillaDataManagerConfig
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    ExponentialDecaySchedulerConfig,
)
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification


minimal_regional_nerfacto = MethodSpecification(
    config=TrainerConfig(
        method_name="minimal-regional-nerfacto",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=MRNerfPipelineConfig(
        datamanager=MRNerfDataManagerConfig(
            dataparser=NerfstudioDataParserConfig(train_split_fraction=0.99),
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=4096
        ),
            model=MRNerfModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                hashgrid_sizes=(19,),
                hashgrid_layers=(16,),
                hashgrid_resolutions=((16, 512),),
                camera_optimizer=CameraOptimizerConfig(mode="off")
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
            # "camera_opt": {
            #     "optimizer": AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2),
            #     "scheduler": ExponentialDecaySchedulerConfig(lr_final=6e-6, max_steps=200000),
            # }
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="[NAVLab] Minimal Regional Nerf method.",
)
