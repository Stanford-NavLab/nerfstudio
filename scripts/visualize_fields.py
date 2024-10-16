"""
Visualize height field or DINO features from a trained model.

Usage:
    python scripts/visualize_fields.py path/to/config.yml
e.g. 
    python scripts/visualize_fields.py outputs/moon_spiral_2/terrain-nerf/2024-10-15_220925/config.yml


NOTE: currently nerfstudio expects this to be called from the nerfstudio directory (for the load config/pipeline
call, since data is a relative path).

"""

import argparse
from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

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


def vis_height_field(N=512, bounds=[-1., 1., -1., 1.], gradients=False):
    """
    Visualize the height field of the terrain model.
    """
    positions = grid_2d(N, bounds)
    if gradients:
        positions.requires_grad = True

    xy = positions[:, :2].detach().cpu().numpy()
    x = xy[:,0] 
    y = xy[:,1] 
    heights = pipeline.model.field.positions_to_heights(positions)
    z = heights.detach().cpu().numpy().flatten()

    print("Min z: ", z.min())
    print("Max z: ", z.max())

    fig = go.Figure(data=[go.Surface(x=x.reshape(N, N), y=y.reshape(N, N), z=z.reshape(N, N), colorscale='Viridis', showscale=False)])
    fig.update_layout(title='Elevation Model', width=1600, height=900)
    fig.update_layout(scene_aspectmode='data')
    fig.show()

    if gradients:
        grad = torch.autograd.grad(heights.sum(), positions, create_graph=True)[0]
        x_grad = grad[:,0].reshape(N, N).detach().cpu().numpy()
        y_grad = grad[:,1].reshape(N, N).detach().cpu().numpy()

        grad_fig = make_subplots(rows=1, cols=2, subplot_titles=('X Gradient', 'Y Gradient'), horizontal_spacing=0.15)
        grad_fig.add_trace(go.Heatmap(z=x_grad, colorbar=dict(len=1.05, x=0.44, y=0.5)), row=1, col=1)
        grad_fig.add_trace(go.Heatmap(z=y_grad, colorbar=dict(len=1.05, x=1.01, y=0.5)), row=1, col=2)
        grad_fig.update_layout(width=1300, height=600, scene_aspectmode='data')
        grad_fig.show()
    
    return fig, grad_fig


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

    parser = argparse.ArgumentParser(description='Trained model path.')
    parser.add_argument('config_path', type=str, help='Path to config.yml file.')

    args = parser.parse_args()
    config_path = args.config_path

    config, pipeline, checkpoint_path, _ = eval_setup(Path(config_path))

    fig, grad_fig = vis_height_field(N=512, bounds=[-1., 1., -1., 1.], gradients=True)

    # Extract scene name from config_path
    scene_name = config_path.split('/')[1]

    # Get path up to config.yml
    save_path = '/'.join(config_path.split('/')[:-1])

    # print(pipeline.model.field.encoder.encoding_config)

    torch.save(pipeline.model.field.encoder.state_dict(), f'{save_path}/{scene_name}_encs.pth')
    torch.save(pipeline.model.field.height_net.state_dict(), f'{save_path}/{scene_name}_mlp.pth')
    #torch.save(pipeline.model.field.nemo.state_dict(), f'{save_path}/{scene_name}.pth')
    print("Saved weights to ", save_path)
    fig.write_html(save_path + '/height_field.html')
    grad_fig.write_html(save_path + '/gradients.html')

    # Create a txt file called description.txt
    with open(save_path + '/description.txt', 'w') as f:
        pass
