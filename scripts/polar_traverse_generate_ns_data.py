#%%
import glob
import json
import os

import imageio
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

"""
The raw dataset is organized into folders with the following structure:
```
View1/
    Traverse1/
        00m/
            loc0_camL_001ms.png
            loc0_camL_005ms.png
            ...
            loc1_camR_001ms.png
            ...
        01m/
            loc2_camL_001ms.png
            loc2_camL_005ms.png
            ...
            loc3_camR_001ms.png
            ...
        02m/
        ...
        10m/
        poses/
            approx_pose.txt
            refined_pose.txt
    Traverse2/
        00m/
            loc22_camL_001ms.png
            loc22_camL_005ms.png
            ...
            loc23_camR_001ms.png
            ...
        01m/
        ...
        10m/
        poses/
    ...
    Traverse6/
View2/
...
View4/
```

Each View uses different lighting. The images are named according to exposure time

"""


base = './View1/Traverse'
shutter = '050ms'

save_img = False

image_width = 2048
image_height = 2048
camera_matrix = np.array([[1.45271e+03, 0., 0.99953e+03],
           [0., 1.45288e+03, 1.03540e+03],
           [0., 0., 1.]])
distortion_coefficients = [-0.016834, -0.027914, -0.000321, -0.000487, -0.001499]

#%%

frames = []
images = []
for i in range(1, 7):
    df = pd.read_csv(base + f'{i}/poses/refined_pose.txt', delimiter = "\t")
    data = np.array(df)

    ID = data[:, 0].astype(np.uint8)
    trans = data[:, 1:4]
    rot = data[:, 4:]

    for id in ID:
        img_fp = glob.glob(base + f'{i}/*' + f'/loc{id}_*'+shutter + '.png')

        save_fp = f"images/frame_{id}.png"

        assert len(img_fp) == 1

        if save_img:
            img = imageio.imread(img_fp[0])
            imageio.imwrite(save_fp, img)

        index = np.where(ID == id)
        t = trans[index].squeeze()
        r = rot[index].squeeze()

        # r = R.from_quat(r)
        r = R.from_quat([r[1], r[2], r[3], r[0]])
        rmat = r.as_matrix()

        # OpenCV format
        transform = np.eye(4)
        transform[:3, :3] = rmat
        transform[:3, -1] = t

        transform[:3, 1] = -transform[:3, 1]
        transform[:3, 2] = -transform[:3, 2]

        # transform[:3, -1] = np.array([-t[0], -t[1], t[2]])

        # transform[1:3, -1] = -transform[1:3, -1]

        # transform[0] = -transform[0]
        # transform[1] = -transform[1]
        # transform = np.linalg.inv(transform)

        frame = {
            "file_path": save_fp,
            "transform_matrix": transform.tolist()
        }

        frames.append(frame)

#%%

data = {
    "w": image_width,
    "h": image_height,
    "fl_x": camera_matrix[0, 0],
    "fl_y": camera_matrix[1, 1],
    "cx": camera_matrix[0, -1],
    "cy": camera_matrix[1, -1],
    "k1": distortion_coefficients[0],
    "k2": distortion_coefficients[1],
    "p1": distortion_coefficients[2],
    "p2": distortion_coefficients[3],
    "camera_model": "OPENCV",
    "ply_file_path": "sparse_pc.ply",
    "frames": frames
}

with open('transforms.json', 'w') as fp:
    json.dump(data, fp, indent=4)

# %%
