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
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-PickPlace-G1-Abs-v0",
    choices=["Isaac-BlockStack-G1-Abs-v0", "Isaac-PickPlace-G1-Abs-v0"],
    help="Name of the task. Options: 'Isaac-BlockStack-G1-Abs-v0', 'Isaac-PickPlace-G1-Abs-v0'."
)
parser.add_argument(
    "--save_data",
    action="store_true",
    default=False,
    help="Save video and compose data to parquet.",
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
import isaaclab_tasks.manager_based.manipulation.pick_place_g1  # noqa: F401
##########
from isaaclab_tasks.utils import parse_env_cfg


# PLACEHOLDER: Extension template (do not remove this comment)
"""Data collection setup"""
import cv2
import os
import numpy as np
import pandas as pd
import time

""" Customized modules """
from utils.trajectory_player import TrajectoryPlayer

""" Constants """
STEPS_PER_MOVEMENT_SEGMENT = 100  # 4 segments for movement
STEPS_PER_GRASP_SEGMENT = 50  # 2 segments for grasp
EPISODE_FRAMES_LEN = STEPS_PER_MOVEMENT_SEGMENT * 4 + STEPS_PER_GRASP_SEGMENT * 2 # frames (steps)
STABILIZATION_STEPS = 30 # Step 30 times for stabilization after env.reset()
FPS = 30  # In pickplace_g1_env_cfg.py, sim.dt * decimation = 1/60 * 2 = 1/30

def main():

    # parquet data setup
    if args_cli.save_data:
        dataset_path = "datasets/gr00t_collection/G1_testing_dataset/"
        output_video_dir = f"{dataset_path}videos/chunk-000/observation.images.camera"
        output_data_dir = f"{dataset_path}data/chunk-000"
        os.makedirs(output_video_dir, exist_ok=True)
        os.makedirs(output_data_dir, exist_ok=True)
    
    video_writer = None
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    frames, obs_list, action_list = [], [], []

    episode_index = 0
    frame_count = 0
    global_index = 100
    task_index = 0    

    # Unitree G1 joint indices in whole body 43 joint.
    LEFT_ARM_INDICES = [11, 15, 19, 21, 23, 25, 27]
    RIGHT_ARM_INDICES = [12, 16, 20, 22, 24, 26, 28]
    LEFT_HAND_INDICES = [31, 37, 41, 30, 36, 29, 35]
    RIGHT_HAND_INDICES = [34, 40, 42, 32, 38, 33, 39] 

    joint_id = LEFT_ARM_INDICES + RIGHT_ARM_INDICES + LEFT_HAND_INDICES + RIGHT_HAND_INDICES

    target_indices = {
        "left_arm": [0, 2, 4, 6, 8, 10, 12],  # 11, 15, 19, 21, 23, 25, 27
        "right_arm": [1, 3, 5, 7, 9, 11, 13],  # 12, 16, 20, 22, 24, 26, 28
        "left_hand": [16, 22, 26, 15, 21, 14, 20],  # 31, 37, 41, 30, 36, 29, 35
        "right_hand": [19, 25, 27, 17, 23, 18, 24]  # 34, 40, 42, 32, 38, 33, 39
    }
    target_idx = np.concatenate([
                    target_indices["left_arm"],
                    target_indices["right_arm"],
                    target_indices["left_hand"],
                    target_indices["right_hand"],
                ]).tolist()
    
    """Random actions agent with Isaac Lab environment."""
    # create environment configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    
    # reset environment
    obs, _ = env.reset()
    # Pass initial observation to TrajectoryPlayer to set default poses
    trajectory_player = TrajectoryPlayer(env.unwrapped, initial_obs=obs, steps_per_movement_segment=100, steps_per_grasp_segment=50)    # 30 fps
    # Get the idle action based on the initial reset pose
    idle_action_np = trajectory_player.get_idle_action_np()
    idle_actions_tensor = torch.tensor(idle_action_np, dtype=torch.float, device=args_cli.device).repeat(env.unwrapped.num_envs, 1)

    iteration = 1
    should_generate_and_play_trajectory = True
    # simulate environment
    while simulation_app.is_running() and iteration < 1000:  # Limit to 1000 iterations for data collection
        with torch.inference_mode():
        
            if should_generate_and_play_trajectory:
                print("Reset and generate new grasp trajectory...")
                obs, _ = env.reset()
                # Step environment with idle action to stabilize after reset
                for _ in range(STABILIZATION_STEPS):
                    obs, _, _, _, _ = env.step(idle_actions_tensor)
                
                # 1. Generate the full trajectory by passing the current observation
                trajectory_player.generate_auto_grasp_pick_place_trajectory(obs=obs)
                # 2. Prepare the playback trajectory
                trajectory_player.prepare_playback_trajectory()
                # 3. Set to False to play this trajectory
                should_generate_and_play_trajectory = False

            actions = None # Initialize actions for the current step
            if trajectory_player.is_playing_back:
                playback_action_tuple = trajectory_player.get_formatted_action_for_playback()
                if playback_action_tuple is not None:
                    action_array_28D_np = playback_action_tuple[0]
                    actions = torch.tensor(action_array_28D_np, dtype=torch.float, device=args_cli.device).repeat(env.unwrapped.num_envs, 1) # type: ignore
                else: # Playback finished
                    print(f"{iteration} trajectory playback finished, and next iteration will start.\n")
                    should_generate_and_play_trajectory = True
                    iteration += 1 # Increment iteration only when a full trajectory cycle finishes
                    actions = torch.tensor(idle_action_np, dtype=torch.float, device=args_cli.device).repeat(env.unwrapped.num_envs, 1)
 
            # apply actions
            obs, _, _, _, _ = env.step(actions)

            robot_joint_state = obs["policy"]["robot_joint_pos"].cpu().numpy().flatten().astype(np.float64)
            processed_action = obs["policy"]["processed_actions"].cpu().numpy().flatten().astype(np.float64)

            rgb_image = obs["policy"]["rgb_image"]  # shape: (1, 480, 640, 3)
            rgb_image = rgb_image.squeeze(0).cpu().numpy()  # remove batch dim (480, 640, 3);from cuda to cpu
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR) # RGB to CV2 BGR format

            data_state = robot_joint_state[joint_id]
            data_action = np.zeros(28)
            # Swap value to match joint action orders
            for tar_i, src_i in enumerate(target_idx):
                data_action[tar_i] = processed_action[src_i]

            # print("State:",data_state)
            # print("Action:",data_action)


            # ===================|
            if args_cli.save_data:

                frames.append(rgb_image)
                obs_list.append(data_state)
                action_list.append(data_action)
                
                
                # preview
                #cv2.imwrite("output/frame_preview.png", rgb_image)

                # start new episode or continue current
                if frame_count % EPISODE_FRAMES_LEN == 0 and frame_count > 0: 

                    timestamps = [i / FPS for i in range(len(frames))]
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
                    video_writer = cv2.VideoWriter(output_path, fourcc, FPS, (rgb_image.shape[1], rgb_image.shape[0]))
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
    if frames and args_cli.save_data:
        timestamps = [i / FPS for i in range(len(frames))]
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
        video_writer = cv2.VideoWriter(output_path, fourcc, FPS, (rgb_image.shape[1], rgb_image.shape[0]))
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
