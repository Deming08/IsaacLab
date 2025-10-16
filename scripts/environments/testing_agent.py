# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to an environment with random action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")

parser.add_argument(
    "--save_img",
    action="store_true",
    default=False,
    help="Save the data from camera RGB image.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# Force enable cameras for this script by modifying the parsed arguments
args_cli.enable_cameras = True

##########
# Import pinocchio before AppLauncher to force the use of the version installed by IsaacLab and not the one installed by Isaac Sim
# pinocchio is required by the Pink IK controllers and the GR1T2 retargeter
import pinocchio  # noqa: F401
##########

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

##########
#import isaaclab_tasks.manager_based.manipulation.playground_g1  # noqa: F401
##########

#import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


# PLACEHOLDER: Extension template (do not remove this comment)

import carb
carb_settings_iface = carb.settings.get_settings()
carb_settings_iface.set_bool("/gr00t/use_joint_space", False)

carb_settings_iface.set_string("/unitree_g1_env/hand_type", "trihand")  # ["trihand", "inspire"]

"""Data collection setup"""
import cv2
import os
import numpy as np
import pandas as pd

def main():
    dataset_path = "datasets/collection_test/G1_testing_dataset/"
    output_video_dir = f"{dataset_path}videos/chunk-000/observation.images.camera"
    output_data_dir = f"{dataset_path}data/chunk-000"
    os.makedirs(output_video_dir, exist_ok=True)
    os.makedirs(output_data_dir, exist_ok=True)

    episode_index = 0
    frame_count = 0
    global_index = 100
    task_index = 0
    fps = 30 
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = None
    frames = []
    obs_list = []
    action_list = []
    episode_len = 5 # second

    """Random actions agent with Isaac Lab environment."""
    # create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    obs = env.reset()
    
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # sample actions from -1 to 1
            actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
            # apply actions
            obs, _, _, _, _ = env.step(actions)

            # ===================|
            if args_cli.save_img:
                rgb_image = obs["policy"]["rgb_image"]  # shape: (1, 480, 640, 3)
                rgb_image = rgb_image.squeeze(0)  # remove batch dim (480, 640, 3)
                rgb_image = rgb_image.cpu().numpy()  # from cuda to cpu
                rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR) # RGB to CV2 BGR format

                obs_policy = obs["policy"]
                observation_state = np.concatenate([
                    obs_policy["hand_joint_state"].cpu().numpy().flatten(),
                    obs_policy["object_pos"].cpu().numpy().flatten(),
                ]).astype(float).tolist()
                
                obs_list.append(observation_state)
                action_list.append(actions.cpu().numpy().flatten().tolist())
                frames.append(rgb_image)
                
                # preview
                """video_writer.write(rgb_image)"""
                cv2.imwrite("output/frame_preview.png", rgb_image)

                # start new episode or continue current
                if frame_count % (fps * episode_len) == 0 and frame_count > 0: 

                    timestamps = [i / fps for i in range(len(frames))]
                    data = {
                        #"observation.images.camera": [f"videos/chunk-000/observation.images.camera/episode_{episode_index:06d}.mp4"] * len(frames),
                        "observation.state": obs_list,
                        "action": action_list,
                        "timestamp": timestamps,
                        "frame_index": list(range(len(frames))),
                        "episode_index": [episode_index] * len(frames),
                        "index": list(range(global_index, global_index + len(frames))),
                        "task_index": [task_index] * len(frames),
                    }
                    #print(len(data['observation.images.camera']),len(data['observation.state']),len(data['action']),len(data['timestamp']),len(data['frame_index']),len(data['episode_index']),len(data['index']),len(data['task_index']))
                    df = pd.DataFrame(data)
                    df.to_parquet(os.path.join(output_data_dir, f"episode_{episode_index:06d}.parquet"))
                    print(f"Save data to {output_data_dir}/episode_{episode_index:06d}")
                    # create new video
                    output_path = os.path.join(output_video_dir, f"episode_{episode_index:06d}.mp4")
                    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (rgb_image.shape[1], rgb_image.shape[0]))
                    for frame in frames:
                        video_writer.write(frame)
                    episode_index += 1
                    global_index += len(frames)
                    frames = []
                    obs_list = []
                    action_list =[]
                    if video_writer is not None:
                        video_writer.release()

                frame_count += 1

    # Save last episode
    if frames:
        timestamps = [i / fps for i in range(len(frames))]
        data = {
            "observation.images.camera": [f"videos/chunk-000/observation.images.camera/episode_{episode_index:06d}.mp4"] * len(frames),
            "observation.state": obs_list,
            "action": action_list,
            "timestamp": timestamps,
            "frame_index": list(range(len(frames))),
            "episode_index": [episode_index] * len(frames),
            "index": list(range(global_index, global_index + len(frames))),
            "task_index": [0] * len(frames),
        }
        df = pd.DataFrame(data)
        df.to_parquet(os.path.join(output_data_dir, f"episode_{episode_index:06d}.parquet"))
        output_path = os.path.join(output_video_dir, f"episode_{episode_index:06d}.mp4")
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (rgb_image.shape[1], rgb_image.shape[0]))
        for frame in frames:
            video_writer.write(frame)
        video_writer.release()
    # close the simulator
    env.close()



if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
