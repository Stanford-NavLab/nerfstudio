"""
Render the shadows from a satellite using the satellite shadow camera model.
"""

from pathlib import Path
import shutil
from dataclasses import dataclass
from typing import List, Literal, Optional, Union

import tyro
from typing_extensions import Annotated


from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils import colormaps
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.scripts.render import CropData

from render_poses import generate_safe_path, RenderCameraPose
from shadow_regional_nerfacto.cameras.satellite_shadow_camera import SatelliteDirectionCamera



def _render_shadow_images(
    pipeline: Pipeline,
    satellite_shadow_cameras: SatelliteDirectionCamera,
    output_filename: Path,
    rendered_output_names: List[str],
    crop_data: Optional[CropData] = None,
    rendered_resolution_scaling_factor: float = 1.0,
    image_format: Literal["jpeg", "png"] = "png",
    jpeg_quality: int = 100,
    depth_near_plane: Optional[float] = None,
    depth_far_plane: Optional[float] = None,
    colormap_options: colormaps.ColormapOptions = colormaps.ColormapOptions()
):
    pass




@dataclass
class ShadowRenderer(RenderCameraPose):
    """Render a GNSS satellite shadow"""

    image_format: Literal["jpeg", "png"] = "jpeg"
    """File type of the output images"""

    def main(self) -> None:
        """
        Entrypoint should lead here.
        """
        print("[Shadow Renderer] Setting up pipeline...")
        _, pipeline, _, _ = eval_setup(
            self.load_config,
            eval_num_rays_per_chunk=self.eval_num_rays_per_chunk,
            test_mode="inference",
        )
        print("[Render Poses] ...Setting up pipeline done")

        self.safe_output_path = generate_safe_path(self.output_path)

        print(f"[Render Poses] Will save to {self.safe_output_path}")
        print(f"Original path was {self.output_path}")

        # TODO!
        crop_data = None

        print("[Render Poses] Printing self")
        print(self.nice_self_json())

        print("[Render Poses] Saving params")
        self.save_params_to_json()
        # Also copy the current JSON
        dst = Path(self.safe_output_path, "camera_path.json")

        _render_shadow_images(
            pipeline=pipeline,
            satellite_shadow_cameras=None,
            output_filename=self.safe_output_path,
            rendered_output_names=self.rendered_output_names,
            crop_data=crop_data,
            image_format = self.image_format,
            depth_near_plane = self.depth_near_plane,
            depth_far_plane = self.depth_far_plane,
            colormap_options = self.colormap_options
        )







Commands = tyro.conf.FlagConversionOff[
    Union[
        Annotated[ShadowRenderer, tyro.conf.subcommand(name="shadow")]
    ]
]


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(Commands).main()


if __name__ == "__main__":
    entrypoint()
