"""
Render the shadows from a satellite using the satellite shadow camera model.
"""

from pathlib import Path
import shutil
import sys
from dataclasses import dataclass
from typing import List, Literal, Optional, Union
from contextlib import ExitStack

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
from nerfstudio.scripts.render import CropData

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


def make_default_crop():
    """
    Default crop
    """
    center = ( 0.0,  0.0, -0.5)
    rot    = ( 0.0,  0.0,  0.0)
    scale  = ( 4.0,  4.0,  1.0) 
    # Scale is the width, height, depth of the box.
    # So, if you want -3 to 3, you need to set 6

    return CropData(
            background_color=torch.Tensor([1, 1, 1]),
            obb=OrientedBox.from_params(center, rot, scale),
        )

@dataclass
class SatAndGrid:
    """
    Convenience Data Class to store satellite information and grid points.
    """

    satellite_directions: Float[Tensor, "*num_satellites 3"]
    grid_points: Float[Tensor, "*num_points 3"]
    pixel_area: Float[Tensor, "1"]

    def __init__(self, satellite_directions, grid_bounds, grid_size, alt, pixel_area):
        """
        Initialize the Satellites and Grid together from user parameters.
        """

        # Satellites
        self.satellite_directions = torch.tensor(satellite_directions)

        # Grid
        self.grid_x_len, self.grid_y_len = grid_size
        self.grid_x_min, self.grid_x_max, self.grid_y_min, self.grid_y_max = grid_bounds
        self.nerf_alt = alt

        self.full_grid = self._init_make_grid()

        # Auxiliary paramaters
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
            torch.linspace(self.grid_x_min, self.grid_x_max, steps=self.grid_x_len),
            torch.linspace(self.grid_y_max, self.grid_y_min, steps=self.grid_y_len),
            indexing='xy'
        )
        grid_z = torch.tensor([self.nerf_alt]).reshape(1, 1).repeat(
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


@dataclass
class ShadowRenderer(RenderCameraPose):
    """Render a GNSS satellite shadow"""
    satellite_grids_filename: Path = None
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
        # sat_and_grid = get_sat_and_grid_from_json(satellite_grids_filename)
        # shadow_cams = sat_and_grid.get_shadow_cameras()
        print("[Shadow Renderer] ... Done")

        self.safe_output_path = generate_safe_path(self.output_path)

        print(f"[Shadow Renderer] Will save to {self.safe_output_path}")
        print(f"Original path was {self.output_path}")

        # Get from file in the future
        crop_data = make_default_crop()
        # crop_data = None

        print("[Shadow Renderer] Printing self")
        print(self.nice_self_json())

        print("[Shadow Renderer] Saving params")
        self.save_params_to_json()
        # Also copy the current JSON
        dst = Path(self.safe_output_path, "camera_path.json")

        grid_x_len = 1007   # Image Width
        grid_y_len = 1003    # Image Height
        nerf_alt   = -0.54
        grid_x, grid_y = torch.meshgrid(
            torch.linspace(-1,  1, steps=grid_x_len),
            torch.linspace( 1, -1, steps=grid_y_len),
            indexing='xy'
        )
        grid_z = torch.tensor([nerf_alt]).reshape(1, 1).repeat(
            *grid_x.shape
        )
        origins = torch.stack(
            (grid_x, grid_y, grid_z), dim=-1
        )
        # directions = torch.tensor(
        #     # [-6.82395927e-04,  8.59793005e-02,  9.96296690e-01]
        #     [0.000001, 0.0000001, 0.9999999999]
        # ) 

        directions = torch.tensor(
           [[-6.82395927e-04,  8.59793005e-02,  9.96296690e-01],
            [ 3.52654100e-01, -3.55976992e-01,  8.65399022e-01],
            [ 6.68119077e-01,  5.68803053e-01,  4.79666536e-01],
            [-7.70372852e-01,  2.59597404e-01,  5.82352863e-01],
            [-9.37999502e-01,  1.76337054e-01,  2.98432869e-01],
            [ 2.03725551e-01,  1.25803163e-01,  9.70911666e-01],
            [-3.50202213e-02, -9.41419647e-01,  3.35414120e-01],
            [ 5.06430401e-01,  5.81563778e-01,  6.36641046e-01],
            [-7.18379069e-01,  4.29425366e-01,  5.47289109e-01],
            [ 8.38455667e-01, -2.88599108e-01,  4.62279838e-01],
            [ 3.62825361e-01, -5.51396912e-01,  7.51211823e-01],
            [ 7.80072757e-01, -4.84298367e-01,  3.96158536e-01],
            [-6.45188601e-01, -1.63485530e-02,  7.63848411e-01]]
        )
        pixel_area = torch.tensor([0.001])

        print("Origins has shape: ", origins.shape)
        print("Directions has shape: ", directions.shape)
        print("Pixel Area has shape: ", pixel_area.shape)

        # shadow_cam = SatelliteDirectionCamera(
        #     origins, directions, pixel_area
        # )
        # print(f"Shadow Cam: {vars(shadow_cam)}")

        # shadow_cams = [
        #     SatelliteDirectionCamera(
        #         origins, direction, pixel_area
        #     ) for direction in directions
        # ]

        grid_bounds = [-1, 1, -1, 1]
        grid_dims = [1007, 1003]
        sat_and_grid = SatAndGrid(directions, grid_bounds, grid_dims, -0.54, 0.001)
        shadow_cams = sat_and_grid.get_shadow_cameras()

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
