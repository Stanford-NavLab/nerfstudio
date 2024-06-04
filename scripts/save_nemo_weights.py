"""
Given given config.yml, save weights for trained nerf model 

"""

import argparse
from pathlib import Path

import torch

from nerfstudio.utils.eval_utils import eval_setup

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Trained model path.')
    parser.add_argument('config_path', type=str, help='Path to config.yml file.')

    args = parser.parse_args()
    config_path = args.config_path

    config, pipeline, checkpoint_path, _ = eval_setup(Path(config_path))

    # Extract scene name from config_path
    scene_name = config_path.split('/')[1]

    # Get path up to config.yml
    save_path = '/'.join(config_path.split('/')[:-1])

    torch.save(pipeline.model.field.encs2d[0].state_dict(), f'{save_path}/{scene_name}_encs.pth')
    torch.save(pipeline.model.field.height_net.state_dict(), f'{save_path}/{scene_name}_mlp.pth')
    print("Saved weights to ", save_path)
