"""
Render the shadows from a satellite using the satellite shadow camera model.
"""

from pathlib import Path
import shutil
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Union
from contextlib import ExitStack
import json

import torch
from torch import Tensor
from jaxtyping import Float

import mediapy as media
import tyro
from typing_extensions import Annotated

from rich import box, style
from rich.panel import Panel
from rich.progress import (BarColumn, Progress, TaskProgressColumn, TextColumn,
                           TimeElapsedColumn, TimeRemainingColumn)
from rich.table import Table

from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.model_components import renderers
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils import colormaps
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE, ItersPerSecColumn
from nerfstudio.scripts.render import CropData, get_crop_from_json

from render_poses import generate_safe_path, RenderCameraPose
from shadow_regional_nerfacto.cameras.satellite_shadow_camera import SatelliteDirectionCamera



def _render_shadow_images(
    pipeline: Pipeline,
    satellite_shadow_cameras: List[SatelliteDirectionCamera],
    output_filename: Path,
    rendered_output_names: List[str],
    crop_data: Optional[CropData] = None,
    image_format: Literal["jpeg", "png"] = "png",
    jpeg_quality: int = 100,
    depth_near_plane: Optional[float] = None,
    depth_far_plane: Optional[float] = None,
    colormap_options: colormaps.ColormapOptions = colormaps.ColormapOptions()
):
    # Pretty terminal interface
    CONSOLE.print("[bold green]Creating shadow images")
    progress = Progress(
        TextColumn(":movie_camera: Rendering :movie_camera:"),
        BarColumn(),
        TaskProgressColumn(
            text_format="[progress.percentage]{task.completed}/{task.total:>.0f}({task.percentage:>3.1f}%)",
            show_speed=True,
        ),
        ItersPerSecColumn(suffix="fps"),
        TimeRemainingColumn(elapsed_when_finished=False, compact=False),
        TimeElapsedColumn(),
    )

    # Make the directory to save the images
    output_image_dir = output_filename.parent / output_filename.stem
    output_image_dir.mkdir(parents=True, exist_ok=True)

    # No idea if this is necessary
    with ExitStack() as stack:
        import numpy as np

        with progress:
            num_sats = len(satellite_shadow_cameras)

            for camera_idx in progress.track(range(num_sats), description=""):
                # Make sure that the camera is on device
                camera = satellite_shadow_cameras[camera_idx]
                camera.send_to_device(pipeline.device)
                # print("Camera on device: ", camera.device)
                # print("Pipeline on device: ", pipeline.device)

                obb_box = None
                if crop_data is not None:
                    obb_box = crop_data.obb

                max_dist, max_idx = -1, -1
                true_max_dist, true_max_idx = -1, -1

                if crop_data is not None:
                    with renderers.background_color_override_context(
                        crop_data.background_color.to(pipeline.device)
                    ), torch.no_grad():
                        outputs = pipeline.model.get_outputs_for_camera(
                            camera, obb_box=obb_box
                        )
                else:
                    with torch.no_grad():
                        outputs = pipeline.model.get_outputs_for_camera(
                            camera, obb_box=obb_box
                        )

                render_image = []
                for rendered_output_name in rendered_output_names:
                    if rendered_output_name not in outputs:
                        CONSOLE.rule("Error", style="red")
                        CONSOLE.print(f"Could not find {rendered_output_name} in the model outputs", justify="center")
                        CONSOLE.print(
                            f"Please set --rendered_output_name to one of: {outputs.keys()}", justify="center"
                        )
                        sys.exit(1)

                    output_image = outputs[rendered_output_name]
                    output_image = (
                            colormaps.apply_colormap(
                                image=output_image,
                                colormap_options=colormap_options,
                            )
                            .cpu()
                            .numpy()
                        )
                    render_image.append(output_image)

                render_image = np.concatenate(render_image, axis=1)

                if image_format == "png":
                    media.write_image(output_image_dir / f"{camera_idx:05d}.png", render_image, fmt="png")
                if image_format == "jpeg":
                    media.write_image(
                        output_image_dir / f"{camera_idx:05d}.jpg", render_image, fmt="jpeg", quality=jpeg_quality
                    )

    table = Table(
        title=None,
        show_header=False,
        box=box.MINIMAL,
        title_style=style.Style(bold=True),
    )
    table.add_row("Images", str(output_image_dir))
    CONSOLE.print(Panel(table, title="[bold][green]:tada: Render Complete :tada:[/bold]", expand=False))


@dataclass
class SatAndGrid:
    """
    Convenience Data Class to store satellite information and grid points.
    """

    satellite_directions: Float[Tensor, "*num_satellites 3"]
    full_grid: Float[Tensor, "*num_points 3"]
    pixel_area: Float[Tensor, "1"]

    def __init__(self, satellite_directions, grid_bounds, grid_size, z_plane, pixel_area):
        """
        Initialize the Satellites and Grid together from user parameters.

        Generally x -> East, y -> North, and z -> Up
        """

        # Satellites
        if isinstance(satellite_directions, Tensor):
            self.satellite_directions = satellite_directions
        else:
            self.satellite_directions = torch.tensor(satellite_directions)
        assert len(self.satellite_directions.shape) == 2
        assert self.satellite_directions.shape[1] == 3

        # Grid
        assert len(grid_size) == 2
        assert len(grid_bounds) == 4
        self.grid_x_num, self.grid_y_num = grid_size
        self.grid_x_min, self.grid_x_max, self.grid_y_min, self.grid_y_max = grid_bounds
        
        assert isinstance(z_plane, float)
        self.z_plane = z_plane

        self.full_grid = self._init_make_grid()
        assert self.full_grid.shape == (self.grid_y_num, self.grid_x_num, 3), \
            f"Full Grid is of size {self.full_grid.shape}, but expected " + \
            f"grid size {grid_size}."

        # Auxiliary paramaters
        assert isinstance(pixel_area, float)
        self.pixel_area = torch.tensor([pixel_area])

    def check_bounds(self):
        """
        Check that the bounds make sense
        """
        assert self.grid_x_min < self.grid_x_max
        assert self.grid_y_min < self.grid_y_max

    def _init_make_grid(self):
        """
        Use meshgrid to make an appropriate grid.
        """
        # Order in linspace is super important!
        # For the camera, xy is the _top left_ corner
        # So   x goes min -> max, 
        # but, y goes max -> min
        # It is also important we use "xy" indexing to match ENU
        grid_x, grid_y = torch.meshgrid(
            torch.linspace(self.grid_x_min, self.grid_x_max, steps=self.grid_x_num),
            torch.linspace(self.grid_y_max, self.grid_y_min, steps=self.grid_y_num),
            indexing='xy'
        )
        grid_z = torch.tensor([self.z_plane]).reshape(1, 1).repeat(
            *grid_x.shape
        )
        return torch.stack(
            (grid_x, grid_y, grid_z), dim=-1
        ) 


    def get_shadow_cameras(self) -> List:
        """
        Package all the satellites and grids into camera objects.

        (One "camera" per satellite)
        """
        return [
            SatelliteDirectionCamera(
                self.full_grid, direction, self.pixel_area
            ) for direction in self.satellite_directions
        ]


def get_sat_and_grid_from_json(shadow_cam_params: Dict[str, Any]) -> SatAndGrid:
    """
    Takes a dictionary (from JSON) of satellite and grid parameters
    """

    # Satellites
    satellite_enu_directions = shadow_cam_params["satellite_enu_directions"]

    # Grid
    grid_x_num = shadow_cam_params["grid"]["grid_num_east"]
    grid_y_num = shadow_cam_params["grid"]["grid_num_north"]

    # IMPORTANT: In the future these will be Lat, Lon not NeRF Coords
    # Hence, we will get a grid_lat_bounds and grid_lon_bounds
    # that we will need to transfer to NeRF coordinates
    grid_x_bounds = shadow_cam_params["grid"]["grid_bounds_east"]
    grid_y_bounds = shadow_cam_params["grid"]["grid_bounds_north"]

    # Similarly, this will altitude that we need to convert to NeRF Coords
    z_plane = shadow_cam_params["grid"]["grid_up_plane"]

    # Auxiliary paramaters
    pixel_area = shadow_cam_params["grid"]["grid_area"]

    return SatAndGrid(
        satellite_directions=satellite_enu_directions,
        grid_bounds=(*grid_x_bounds, *grid_y_bounds),
        grid_size=(grid_x_num, grid_y_num),
        z_plane=z_plane,
        pixel_area=pixel_area
    )



@dataclass
class ShadowRenderer(RenderCameraPose):
    """Render a GNSS satellite shadow"""
    shadow_cam_params_filename: Path = None
    """Filename to the satellite directions and grid points to render"""

    image_format: Literal["jpeg", "png"] = "png"
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
        print("[Shadow Renderer] ...Setting up pipeline done")

        print("[Shadow Renderer] Computing Satellites & Grid...")
        with open(self.shadow_cam_params_filename, "r", encoding="utf-8") as f:
            shadow_cam_params = json.load(f)

        sat_and_grid = get_sat_and_grid_from_json(shadow_cam_params)
        shadow_cams = sat_and_grid.get_shadow_cameras()

        print("[Shadow Renderer] ... Done")

        print("[Shadow Renderer] Computing Crop...")
        crop_data = get_crop_from_json(shadow_cam_params)
        print("[Shadow Renderer] ...Done")

        self.safe_output_path = generate_safe_path(self.output_path)

        print(f"[Shadow Renderer] Will save to {self.safe_output_path}")
        print(f"Original path was {self.output_path}")

        print("[Shadow Renderer] Printing self")
        print(self.nice_self_json())

        print("[Shadow Renderer] Saving params")
        self.save_params_to_json()
        # Also copy the current JSON
        dst = Path(self.safe_output_path, "shadow_cam_params.json")
        shutil.copyfile(self.shadow_cam_params_filename, dst)

        _render_shadow_images(
            pipeline=pipeline,
            satellite_shadow_cameras=shadow_cams,
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
