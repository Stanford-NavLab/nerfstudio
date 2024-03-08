import json
import os

import numpy as np


def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument("datadir", type=str, help='path to your meta')
    parser.add_argument("filename", type=str, default='transforms.json', help='file name')
    return parser
    

if __name__ == '__main__':

    parser = config_parser()
    args = parser.parse_args()

    with open(os.path.join(args.datadir, args.filename), 'r') as f:
        data = json.load(f)

    # Extract camera poses
    poses = np.array([[data['frames'][i]['transform_matrix']] 
                            for i in range(len(data['frames']))])
    poses = np.squeeze(poses)
    positions = poses[:, :3, 3]

    # PCA on positions
    mean = np.mean(positions, axis=0)
    positions = positions - mean
    cov = np.dot(positions.T, positions) / positions.shape[0]
    U, S, V = np.linalg.svd(cov)

    # X and Y directions
    X = U[:, 0]
    Y = U[:, 1]
    Z = U[:, 2]

    T = np.eye(4)
    T[:3, 0] = X
    T[:3, 1] = Y
    T[:3, 2] = Z
    poses_rotated = [T.T @ pose for pose in poses]
    poses_rotated = np.array(poses_rotated)

    # Write the new poses back into the json file
    for i in range(len(data['frames'])):
        data['frames'][i]['transform_matrix'] = poses_rotated[i].tolist()

    with open(os.path.join(args.datadir, 'transforms_rotated.json'), 'w') as f:
        json.dump(data, f)
    