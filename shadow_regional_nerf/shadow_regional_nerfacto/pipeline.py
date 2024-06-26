"""
Nerfstudio SRNerf Pipeline
"""

import typing
from dataclasses import dataclass, field
from typing import Literal, Optional, Type

import torch.distributed as dist
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from nerfstudio.data.datamanagers.base_datamanager import (
    DataManager,
    DataManagerConfig,
)
from nerfstudio.models.base_model import ModelConfig

from minimal_regional_nerfacto.pipeline import (
    MRNerfPipeline,
    MRNerfPipelineConfig
)

from shadow_regional_nerfacto.datamanager import SRNerfDataManagerConfig
from shadow_regional_nerfacto.model import SRNerfModel, SRNerfModelConfig


@dataclass
class SRNerfPipelineConfig(MRNerfPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: SRNerfPipeline)
    """target class to instantiate"""
    datamanager: DataManagerConfig = SRNerfDataManagerConfig()
    """specifies the datamanager config"""
    model: ModelConfig = SRNerfModelConfig()
    """specifies the model config"""


class SRNerfPipeline(MRNerfPipeline):
    """
    SRNerf Pipeline

    Args:
        config: the pipeline config used to instantiate class
    """

    def __init__(
        self,
        config: SRNerfPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        print("---------------------------")
        print("[SRNeRF Pipeline] STARTING SUPER INIT PIPELINE")
        # super(MRNerfPipeline, self)
        super().__init__(
            config=config,
            device=device,
            test_mode=test_mode,
            world_size=world_size,
            local_rank=local_rank,
            grad_scaler=grad_scaler
        )

        print("---------------------------")
        print(f"[SRNeRF Pipeline] Sending satellites to GPU: {device}")
        self.model.satellite_directions = self.model.satellite_directions.to(device)

        print("---------------------------")
        print("[SRNeRF Pipeline] Complete\n")
