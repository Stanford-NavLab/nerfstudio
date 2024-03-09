"""
Nerfstudio MRNerf Pipeline
"""

import typing
from dataclasses import dataclass, field
from typing import Literal, Optional, Type

import torch.distributed as dist
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from minimal_regional_nerfacto.datamanager import MRNerfDataManagerConfig
from minimal_regional_nerfacto.model import MRNerfModel, MRNerfModelConfig
from nerfstudio.data.datamanagers.base_datamanager import (
    DataManager,
    DataManagerConfig,
)
from nerfstudio.models.base_model import ModelConfig
from nerfstudio.pipelines.base_pipeline import (
    VanillaPipeline,
    VanillaPipelineConfig,
)


@dataclass
class MRNerfPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: MRNerfPipeline)
    """target class to instantiate"""
    datamanager: DataManagerConfig = MRNerfDataManagerConfig()
    """specifies the datamanager config"""
    model: ModelConfig = MRNerfModelConfig()
    """specifies the model config"""


class MRNerfPipeline(VanillaPipeline):
    """MRNerf Pipeline

    Args:
        config: the pipeline config used to instantiate class
    """

    def __init__(
        self,
        config: MRNerfPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        print("---------------------------")
        print("[MRNeRF Pipeline] STARTING SUPER INIT PIPELINE")
        # VanillaPipeline, self
        super(VanillaPipeline, self).__init__()
        self.config = config
        self.test_mode = test_mode

        print("---------------------------")
        print("[MRNeRF Pipeline] Setting up datamanager")
        self.datamanager: DataManager = config.datamanager.setup(
            device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank
        )
        print("----------------------------------------------------")
        print("[MRNeRF Pipeline] Sending data manager to GPU")
        # print(self.datamanager)
        self.datamanager.to(device)
        print("----------------------------------------------------")
        print(f"[MRNeRF Pipeline] Data manager that is on GPU: {device}")
        # print(self.datamanager)

        assert self.datamanager.train_dataset is not None, "Missing input dataset"
        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            metadata=self.datamanager.train_dataset.metadata,
            device=device,
            grad_scaler=grad_scaler,
        )

        ######################
        # Minimal Regional Nerf Specific
        ######################
        print("----------------------------------------------------")
        print("[MRNeRF Pipeline] Setting ENU transforms")
        self.model.set_enu_transform(
            enu2nerf=self.datamanager.enu2nerf, 
            nerf2enu=self.datamanager.nerf2enu, 
            enu2nerf_points=self.datamanager.enu2nerf_points, 
            nerf2enu_points=self.datamanager.nerf2enu_points,  
            center_latlon=self.datamanager.center_latlon,
            center_height=self.datamanager.center_height
            )
        print("----------------------------------------------------")
        print("[MRNeRF Pipeline] Sending model to GPU")
        self.model.to(device)
        print("----------------------------------------------------")
        print(f"[MRNeRF Pipeline] Model is on GPU: {device}")

        # Run the ENU setting callback now that the device is on the GPU
        # In particular, we need both model and datamanager on the same device
        self.model.latlon_set(None)

        # Code borrowed from Nerfacto
        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(
                MRNerfModel, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True)
            )
            dist.barrier(device_ids=[local_rank])
