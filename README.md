# NeRFstudio fork for Stanford NAVLab

This is a fork of the [NeRFstudio](https://github.com/nerfstudio-project/nerfstudio/) project. It is used in the construction of Neural City Maps, a project by the Stanford NAVLab.

This branch (adam/terrain) is used for the Neural Elevation Models (NEMo) project, for terrain mapping and path planning. The `terrain_nerf` folder contains the code for the `terrain-nerfacto` method, which implements NEMo (a combined radiance field and height field).

## Installation

Follow the instructions for Nerfstudio installation [here](https://docs.nerf.studio/quickstart/installation.html) up to "Dependencies" (create conda environment and install PyTorch and dependencies). Afterwards, clone this repo and install the nerfstudio package from source:
```
git clone https://github.com/Stanford-NavLab/nerfstudio.git
cd nerfstudio
pip install --upgrade pip setuptools
pip install -e .
```

Next, install the `terrain_nerf` package:
```
cd terrain_nerf
pip install -e .
ns-install-cli
```

## Data 

We use simulated imagery from [Google Earth Studio](https://www.google.com/earth/studio/) as well as real-world aerial drone imagery. Drone imagery is available at https://dronemapper.com/sample_data/ (we use the [Red Rocks, Oblique dataset](https://s3.amazonaws.com/DroneMapper_US/example/DroneMapper-RedRocks-Oblique.zip)).

### Google Earth Studio (GES)

GES allows for rendering imagery of any location on Earth (and the Moon and Mars) using Google Earth. An account is needed.

1. Generate GES `.esp` project file from lat/lon/alt using `scripts/generate_ges_traj.py`
2. Load the `.esp` file into GES (create blank project, then import)
3. Set settings, then render.
4. Use `scripts/ges2transforms.py` to generate `transforms.json`.

### Preparation

Data preparation is identical to that of Nerfstudio. 
1. Create a `/data` folder within the repo.
2. For each scene, create a folder within `/data` (e.g., `/Scene01`).
3. Inside the scene folder, place imagery and a `transforms.json` file containing camera poses and parameters. If needed, use COLMAP or Nerfstudio's `ns-process-data` to estimate camera poses.

## Training


