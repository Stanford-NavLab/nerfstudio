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
        print("STARTING SUPER INIT PIPELINE FROM MRNERF")
        super(MRNerfPipeline, self).__init__()
