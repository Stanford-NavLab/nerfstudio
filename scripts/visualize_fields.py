"""
Visualize height field or DINO features from a trained model.

"""

import argparse
from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch

from nerfstudio.utils.eval_utils import eval_setup

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# %% Functions

def grid_2d(N, bounds):
    """Grid of 2D (N x N) points"""
    xs = torch.linspace(bounds[0], bounds[1], N, device=device)
    ys = torch.linspace(bounds[2], bounds[3], N, device=device)
    XY_grid = torch.meshgrid(xs, ys, indexing='xy')
    XY_grid = torch.stack(XY_grid, dim=-1)
    positions = XY_grid.reshape(-1, 2)
    return positions


def vis_height_field(N=512, bounds=[-1., 1., -1., 1.]):
    """
    Visualize the height field of the terrain model.
    """
    positions = grid_2d(N, bounds)

    xy = positions[:, :2].detach().cpu().numpy()
    x = xy[:,0] 
    y = xy[:,1] 
    z = pipeline.model.field.positions_to_heights(positions).detach().cpu().numpy().flatten()

    print("Ground height: ", pipeline.model.field.ground_height)
    print("Min z: ", z.min())
    print("Max z: ", z.max())

    fig = go.Figure(data=[go.Surface(x=x.reshape(N, N), y=y.reshape(N, N), z=z.reshape(N, N), colorscale='Viridis')])
    fig.update_layout(title='Elevation Model', width=1600, height=900)
    fig.update_layout(scene_aspectmode='data')
    fig.show()


def vis_dino_features(N=512, bounds=[-1., 1., -1., 1.]):
    """
    Visualize the DINO features of the terrain model.
    """
    positions = grid_2d(N, bounds)
    dino_features = pipeline.model.field.positions_to_dino(positions).detach().cpu().numpy()

    dino_img = dino_features.reshape(N, N, -1)
    # Get the first 3 channels
    dino_img = dino_img[:, :, :3]

    fig = px.imshow(dino_img)
    fig.update_layout(title='DINO Features', width=800, height=800)
    fig.update_layout(scene_aspectmode='data')
    fig.show()


# %% Main

if __name__ == '__main__':

    # TODO: argparser: config.yml path
    #config, pipeline, checkpoint_path, _ = eval_setup(Path('outputs/RedRocks/terrain-nerfacto/2024-03-15_175147/config.yml'))  # RedRocks MLP sine
    #config, pipeline, checkpoint_path, _ = eval_setup(Path('outputs/RedRocks/terrain-nerfacto/2024-03-15_194521/config.yml'))  # RedRocks MLP relu
    #config, pipeline, checkpoint_path, _ = eval_setup(Path('outputs/GES_KT22/terrain-nerfacto/2024-05-09_184038/config.yml')) # KT22 MLP relu
    #config, pipeline, checkpoint_path, _ = eval_setup(Path('outputs/GES_KT22/terrain-nerfacto/2024-03-21_132915/config.yml'))  # GES Moon MLP relu
    #config, pipeline, checkpoint_path, _ = eval_setup(Path('outputs/RedRocks/terrain-nerfacto/2024-04-11_160510/config.yml'))  # RedRocks w/DINO
    #config, pipeline, checkpoint_path, _ = eval_setup(Path('outputs/CraterMountain/terrain-nerfacto/2024-04-18_151549/config.yml'))  # Crater Mountain
    #config, pipeline, checkpoint_path, _ = eval_setup(Path('outputs/AirSimMountains/terrain-nerfacto/2024-05-09_185714/config.yml'))  # AirSim LandscapeMountains
    #config, pipeline, checkpoint_path, _ = eval_setup(Path('outputs/moon_spiral/terrain-nerfacto/2024-05-21_165443/config.yml')) # AirSim Moon


    parser = argparse.ArgumentParser(description='Trained model path.')
    parser.add_argument('config_path', type=str, help='Path to config.yml file.')

    args = parser.parse_args()
    config_path = args.config_path

    config, pipeline, checkpoint_path, _ = eval_setup(Path(config_path))

    vis_height_field(N=512, bounds=0.75*np.array([-1., 1., -1., 1.]))

