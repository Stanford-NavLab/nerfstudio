"""
Generate NeRF features for a given scene

Run from nerfstudio directory

Usage:
    python scripts/nerf_features.py path/to/config.yml
e.g. 
    python scripts/nerf_features.py outputs/moon_spiral_2/terrain-nerf/2024-10-15_220925/config.yml

"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.utils.eval_utils import eval_setup

torch.manual_seed(42)


#%% ------------------------------------ Functions ------------------------------------ %%

def nerf_render_rays(origins, directions, device):
    """Accumulated NeRF outputs for N rays

    Parameters
    ----------
    origins : np.array (N, 3)
        Ray origins
    directions : np.array (N, 3)
        Ray direction vectors

    """
    # Leave as default
    pixel_area = torch.ones_like(origins[..., :1])
    camera_indices = torch.zeros_like(origins[..., :1])

    ray_bundle = RayBundle(origins=origins, directions=directions, 
                           pixel_area=pixel_area, camera_indices=camera_indices)
    
    # Sets the near and far properties
    ray_bundle = pipeline.model.collider(ray_bundle).to(device)
    
    #return pipeline.model(ray_bundle)
    return pipeline.model.get_outputs_for_camera_ray_bundle(ray_bundle)


#%% ------------------------------------ Main ------------------------------------ %%

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Trained model path.')
    parser.add_argument('config_path', type=str, help='Path to config.yml file.')
    args = parser.parse_args()
    config_path = args.config_path
    save_path = '/'.join(config_path.split('/')[:-1])

    config, pipeline, checkpoint_path, _ = eval_setup(Path(config_path))

    print(f"Using checkpoint_path: {checkpoint_path}")

    with torch.no_grad():
        # Grid of xy points in NeRF coordinates ([-1, 1] x [-1, 1])
        z = 1.0

        bounds = 1.0
        N_res = 512

        x = torch.linspace(-bounds, bounds, N_res)
        y = torch.linspace(-bounds, bounds, N_res)

        xx, yy = torch.meshgrid(x, y, indexing='ij')
        xyz = torch.stack([xx, yy, z*torch.ones_like(xx)], dim=-1)
        
        origins = xyz.clone().detach().to(pipeline.device)
        print("Origins shape: ", origins.shape)

        N_rays = len(origins)

        # All rays pointing down
        directions = torch.tensor([[0.0, 0.0, -1.0]], device=pipeline.device).repeat(N_rays, 1)

        print("Generating features for NeRF coordinates")
    
        rgbd = nerf_render_rays(origins, directions, device=pipeline.device)

        rgb = rgbd['rgb'].detach().cpu().numpy()
        print(rgb.shape)
        rgb = rgb.reshape((N_res, N_res, 3))
        
        # Plot the RGB image
        fig = plt.figure()
        plt.imshow(rgb)
        fig.savefig(save_path + '/nerf_img.png')

        # Plot the depth
        depth = -rgbd['depth'].detach().cpu().numpy()
        depth = depth.reshape((N_res, N_res))
        # Limit depth to -2.0 to -1.0
        depth = np.clip(depth, -2.0, -1.0)
        fig = plt.figure()
        plt.imshow(depth)
        fig.savefig(save_path + '/nerf_depth.png')

        fig = go.Figure(data=[go.Surface(x=xx, y=yy, z=depth, colorscale='Viridis', cmin=-1.6, cmax=-1.4)])
        fig.update_layout(title='Elevation Model', width=1500, height=800)
        fig.update_layout(scene_aspectmode='data')
        fig.show()
        fig.write_html(save_path + '/nerf_depth.html')

        np.save(save_path + '/nerf_depth.npy', depth)


