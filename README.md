# Nerfstudio fork for Stanford NAV Lab

This is a fork of the [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio/) repository. It is used in the construction of Neural City Maps and [Neural Elevation Models](https://github.com/adamdai/neural_elevation_models), projects by the Stanford NAV Lab.

## Installation

Follow the instructions for Nerfstudio installation [here](https://docs.nerf.studio/quickstart/installation.html) up to "Dependencies" (create conda environment and install PyTorch and dependencies). Afterwards, clone the repo, switch to this branch, and install the nerfstudio package from source:
```
git clone https://github.com/Stanford-NavLab/nerfstudio.git
cd nerfstudio
git checkout adam/terrain
pip install --upgrade pip setuptools
pip install -e .
```

Then, install the `terrain_nerf` package:
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
     - e.g., `python scripts/generate_ges_traj.py ges_traj.esp 37.333976 -121.8875317 200 --template Templates/Transamerica.json`
2. Load the `.esp` file into GES (create blank project, then import)
     - Set total time to 10 seconds (with 30 FPS for a total of 300 frames), check "Scale existing keyframes"
3. After setting settings, then render.
     - Drag Google Earth logo to bottom right corner
4. Use `scripts/ges2transforms.py` to generate `transforms.json`.
     - e.g., `python scripts/ges2transforms.py ../nerfstudio_ws/GESSanJose/ san_jose.json 37.333976 -121.8875317`

### Preparation

Data preparation is identical to that of Nerfstudio. 
1. Create a `/data` folder within the repo.
2. For each scene, create a folder within `/data` (e.g., `/Scene01`).
3. Inside the scene folder, place imagery and a `transforms.json` file containing camera poses and parameters. If needed, use COLMAP or Nerfstudio's `ns-process-data` to estimate camera poses.


## Training

As per Nerfstudio training procedure, run the following command:
```
ns-train terrain-nerfacto --data data/Scene01
```
and monitor training through Viser and/or Weights and Biases.

To save height field weights, use `scripts/save_nemo_weights.py`.
