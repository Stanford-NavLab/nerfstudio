"""
Given given config.yml, save weights for trained nerf model 

"""
# %% Setup

from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import torch

from nerfstudio.utils.eval_utils import eval_setup

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# TODO: argparser: config.yml path
#config, pipeline, checkpoint_path, _ = eval_setup(Path('/home/navlab-exxact/NeRF/nerfstudio/outputs/MoonTestScenario/regional-nerfacto/2023-10-13_135547/config.yml'))
#config, pipeline, checkpoint_path, _ = eval_setup(Path('/home/navlab-exxact/NeRF/nerfstudio/outputs/GESMoonRender/regional-nerfacto/2023-10-25_144723/config.yml'))
#config, pipeline, checkpoint_path, _ = eval_setup(Path('/home/navlab-exxact/NeRF/nerfstudio/outputs/GESMoonRender/regional-nerfacto/2023-11-06_145111/config.yml'))
#config, pipeline, checkpoint_path, _ = eval_setup(Path('/home/navlab-exxact/NeRF/nerfstudio/outputs/GESSanJose/terrain-nerfacto/2024-03-15_161957/config.yml'))

#config, pipeline, checkpoint_path, _ = eval_setup(Path('outputs/RedRocks/terrain-nerfacto/2024-03-15_175147/config.yml'))  # RedRocks MLP sine
config, pipeline, checkpoint_path, _ = eval_setup(Path('outputs/RedRocks/terrain-nerfacto/2024-03-15_194521/config.yml'))  # RedRocks MLP relu
#config, pipeline, checkpoint_path, _ = eval_setup(Path('outputs/GES_KT22/terrain-nerfacto/2024-03-15_203528/config.yml')) # KT22 MLP relu
#config, pipeline, checkpoint_path, _ = eval_setup(Path('outputs/GES_KT22/terrain-nerfacto/2024-03-21_132915/config.yml'))  # GES Moon MLP relu
config, pipeline, checkpoint_path, _ = eval_setup(Path('outputs/RedRocks/terrain-nerfacto/2024-04-11_160510/config.yml'))  # RedRocks w/DINO


# TODO: arg: model component options

# TODO: arg: save path

# torch.save(pipeline.model.field.encs2d[0].state_dict(), 'outputs/redrocks_encs_relu.pth')
# torch.save(pipeline.model.field.height_net.state_dict(), 'outputs/redrocks_height_net_relu.pth')
# print("saved model")

# print("Ground height: ", pipeline.model.field.ground_height)

# %% --------------------- Sample DINO feature field --------------------- %% #

N = 512
bound = 0.75
XY_grid = torch.meshgrid(
    torch.linspace(-bound, bound, N, device=device),
    torch.linspace(-bound, bound, N, device=device),
    indexing='xy'
)
XY_grid = torch.stack(XY_grid, dim=-1)
positions = XY_grid.reshape(-1, 2)

xy = positions[:, :2].detach().cpu().numpy()
x = xy[:,0] 
y = xy[:,1] 
dino_features = pipeline.model.field.positions_to_dino(positions).detach().cpu().numpy()

dino_img = dino_features.reshape(N, N, -1)
# Get the first 3 channels
dino_img = dino_img[:, :, :3]

fig = px.imshow(dino_img)
fig.update_layout(title='DINO Features', width=800, height=800)
fig.update_layout(scene_aspectmode='data')
fig.show()

print("dino features shape: ", dino_features.shape)
raise


# %% --------------------- Sample heights --------------------- %% #

# Random Nx3 values
N = 512
bound = 0.75
XY_grid = torch.meshgrid(
    torch.linspace(-bound, bound, N, device=device),
    torch.linspace(-bound, bound, N, device=device),
    indexing='xy'
)
# x_grid = XY_grid[0].cpu().numpy()
# y_grid = XY_grid[1].cpu().numpy()
XY_grid = torch.stack(XY_grid, dim=-1)
positions = XY_grid.reshape(-1, 2)
#positions = torch.cat([positions, torch.zeros_like(positions[:, :1])], dim=-1)

xy = positions[:, :2].detach().cpu().numpy()
x = xy[:,0] 
y = xy[:,1] 
z = pipeline.model.field.positions_to_heights(positions).detach().cpu().numpy().flatten()

print("Ground height: ", pipeline.model.field.ground_height)
print("Min z: ", z.min())

# keep = z < -1.5
# x = x[keep]
# y = y[keep]
# z = z[keep]

# fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=2, color=z, colorscale='Viridis'))])
fig = go.Figure(data=[go.Surface(x=x.reshape(N, N), y=y.reshape(N, N), z=z.reshape(N, N))])
fig.update_layout(title='Elevation Model', width=1500, height=800)
fig.update_layout(scene_aspectmode='data')
fig.show()
# fig.write_html("kt22.html")


# %% ------------------ Test computing spatial derivatives ------------------ %% #

# def gradient(y, x, grad_outputs=None):
#     if grad_outputs is None:
#         grad_outputs = torch.ones_like(y)
#     grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
#     return grad

# x = torch.tensor([0.1, 0.2], dtype=torch.float32, requires_grad=True)
# y = torch.tensor([0.0, 0.1], dtype=torch.float32, requires_grad=True)
# positions = torch.stack([x, y], dim=1)
# #positions = torch.cat([positions, torch.zeros_like(positions[:, :1])], dim=-1)
# #print("positions: ", positions)
# # test_pos = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32, requires_grad=True)

# heights = pipeline.model.field.positions_to_heights(positions)
# #print("heights: ", heights)
# print(gradient(heights, x))

# %% --------------------- Plot spatial derivatives --------------------- %% #

# N = 256
# x, y = torch.meshgrid(
#     torch.linspace(-2, 2, N, requires_grad=True),
#     torch.linspace(-2, 2, N, requires_grad=True),
#     indexing='ij'
# )
# print(x.shape, y.shape)
# XY_grid = torch.stack((x, y), dim=-1)
# print(XY_grid.shape)
# positions = XY_grid.reshape(-1, 2)
# print(positions.shape)
# positions = torch.cat([positions, torch.zeros_like(positions[:, :1])], dim=-1)

# heights = pipeline.model.field.positions_to_heights(positions)
# print(heights.shape)

# fig = px.imshow(heights.reshape(N, N).detach().cpu().numpy())

# # fig = px.imshow(gradient(heights, x).detach().cpu().numpy())
# fig.show()

# %% --------------------- Compute derivatives along path --------------------- %% #

# x0 = torch.tensor([0.0, 0.0], dtype=torch.float32, requires_grad=True)
# xf = torch.tensor([1.0, 1.0], dtype=torch.float32, requires_grad=True)

# t = torch.linspace(0, 1, 100)
# positions = x0 + t[:, None] * (xf - x0)
# heights = pipeline.model.field.positions_to_heights(positions)
# print(positions.shape)
# print(gradient(heights, positions).shape)
