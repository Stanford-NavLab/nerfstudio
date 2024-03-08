"""
Shadow Regional Nerfacto DataManager
"""

from dataclasses import dataclass, field
from typing import Dict, Literal, Tuple, Type, Union

import torch
import json

from nerfstudio.cameras.rays import RayBundle
# from nerfstudio.data.datamanagers.base_datamanager import (
#     VanillaDataManager,
#     VanillaDataManagerConfig,
# )

from minimal_regional_nerfacto.datamanager import (
    MRNerfDataManager,
    MRNerfDataManagerConfig
)

from pathlib import Path

import numpy as np
import math

from minimal_regional_nerfacto.utils.geodetic_utils import get_elevation


@dataclass
class SRNerfDataManagerConfig(MRNerfDataManagerConfig):
    """SRNerf DataManager Config

    Add your custom datamanager config parameters here.
    """

    _target: Type = field(default_factory=lambda: SRNerfDataManager)


class SRNerfDataManager(MRNerfDataManager):
    """SRNerf DataManager

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: MRNerfDataManagerConfig

    def __init__(
        self,
        config: MRNerfDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,  # pylint: disable=unused-argument
    ):
        super().__init__(
            config=config, device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank, **kwargs
        )

        self.create_enu_mapping()
        print("--------------------------------")
        print("[Data Manager] Finished ENU Mappings")
