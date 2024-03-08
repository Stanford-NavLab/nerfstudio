"""
Nerfstudio SRNerf Pipeline
"""

import typing
from dataclasses import dataclass, field
from typing import Literal, Optional, Type

import torch.distributed as dist
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

# from minimal_regional_nerfacto.datamanager import MRNerfDataManagerConfig
# from minimal_regional_nerfacto.model import MRNerfModel, MRNerfModelConfig
from nerfstudio.data.datamanagers.base_datamanager import (
    DataManager,
    DataManagerConfig,
)
from nerfstudio.models.base_model import ModelConfig
# from nerfstudio.pipelines.base_pipeline import (
#     VanillaPipeline,
#     VanillaPipelineConfig,
# )

from minimal_regional_nerfacto.pipeline import (
    MRNerfPipeline,
    MRNerfPipelineConfig
)

from shadow_regional_nerfacto.datamanager import SRNerfDataManagerConfig
from shadow_regional_nerfacto.model import SRNerfModel, SRNerfModelConfig


@dataclass
class SRNerfPipelineConfig(MRNerfPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: MRNerfPipeline)
    """target class to instantiate"""
    datamanager: DataManagerConfig = MRNerfDataManagerConfig()
    """specifies the datamanager config"""
    model: ModelConfig = MRNerfModelConfig()
    """specifies the model config"""


class SRNerfPipeline(MRNerfPipeline):
    """
    SRNerf Pipeline

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
        print("STARTING SUPER INIT PIPELINE FROM MRNERF")
        super(MRNerfPipeline, self).__init__()
