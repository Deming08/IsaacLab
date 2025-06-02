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
parser.add_argument("--task", type=str, default="Isaac-BlockStack-G1-Abs-v0", help="Name of the task.")

parser.add_argument("--port", type=int, help="Port number for the server.", default=5555)
parser.add_argument(
    "--host", type=str, help="Host address for the server.", default="localhost"
)

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

"""gr00t integration"""
from gr00t.eval.robot import RobotInferenceClient, RobotInferenceServer
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy

import time

import carb
carb_settings_iface = carb.settings.get_settings()
carb_settings_iface.set_bool("/gr00t/use_joint_space", True)


def main():
    """GR00T actions agent with Isaac Lab environment."""

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
    obs, _ = env.reset()

    """gr00t inference client"""
    policy_client = RobotInferenceClient(host=args_cli.host, port=args_cli.port)

    print("Available modality config available:")
    modality_configs = policy_client.get_modality_config()
    print(modality_configs.keys())
    
    # Unitree G1 joint indices in whole body 43 joint.
    LEFT_ARM_INDICES = [11, 15, 19, 21, 23, 25, 27]
    RIGHT_ARM_INDICES = [12, 16, 20, 22, 24, 26, 28]
    LEFT_HAND_INDICES = [31, 37, 41, 30, 36, 29, 35]
    RIGHT_HAND_INDICES = [34, 40, 42, 32, 38, 33, 39] 
    
    source_indices = {
        "left_arm": list(range(0, 7)),
        "right_arm": list(range(7, 14)),
        "left_hand": list(range(14, 21)),
        "right_hand": list(range(21, 28))
    }
    target_indices = {
        "left_arm": [0, 2, 4, 6, 8, 10, 12],  # 11, 15, 19, 21, 23, 25, 27
        "right_arm": [1, 3, 5, 7, 9, 11, 13],  # 12, 16, 20, 22, 24, 26, 28
        "left_hand": [16, 22, 26, 15, 21, 14, 20],  # 31, 37, 41, 30, 36, 29, 35
        "right_hand": [19, 25, 27, 17, 23, 18, 24]  # 34, 40, 42, 32, 38, 33, 39
    }


    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            
            """fake_obs = {
                "video.cam_right_high": np.random.randint(0, 256, (1, 480, 640, 3), dtype=np.uint8),
                "state.left_arm": np.random.rand(1, 7),
                "state.right_arm": np.random.rand(1, 7),
                "state.left_hand": np.random.rand(1, 7),
                "state.right_hand": np.random.rand(1, 7),
                "annotation.human.action.task_description": ["do your thing!"],
            }"""

            robot_joint_pos = obs["policy"]["robot_joint_pos"].cpu().numpy().astype(np.float64)
            rgb_image = obs["policy"]["rgb_image"].cpu().numpy().astype(np.uint8)  # shape: (1, 480, 640, 3)
            #rgb_image2 = cv2.cvtColor(rgb_image1.squeeze(0), cv2.COLOR_RGB2BGR)
            #rgb_image = np.expand_dims(rgb_image2, axis=0)
            #print(rgb_image1.shape,rgb_image2.shape,rgb_image.shape)
            
            gr00t_obs = {
                "video.cam_right_high": rgb_image, # shape: (1, 480, 640, 3)
                "state.left_arm": robot_joint_pos[:, LEFT_ARM_INDICES],  # (1, 7)
                "state.right_arm": robot_joint_pos[:, RIGHT_ARM_INDICES],  # (1, 7)
                "state.left_hand": robot_joint_pos[:, LEFT_HAND_INDICES],  # (1, 7)
                "state.right_hand": robot_joint_pos[:, RIGHT_HAND_INDICES],  # (1, 7)
                "annotation.human.action.task_description": ["stack three block"],
            }
            
      
            time_start = time.time()
            gr00t_action = policy_client.get_action(gr00t_obs)

            print(f"Total time taken to get action from server: {time.time() - time_start} seconds")
            """for key, value in gr00t_action.items():
                print(f"Action: {key}: {value.shape}")"""
            
            
            gr00t_action_cat = np.concatenate([
                            gr00t_action["action.left_arm"], 
                            gr00t_action["action.right_arm"], 
                            gr00t_action["action.left_hand"], 
                            gr00t_action["action.right_hand"]
                            ], axis=1)
            

            env_action_np = np.zeros((16, 28))
            # Swap value to match joint action orders
            for key in ["left_arm", "right_arm", "left_hand", "right_hand"]:
                src_idx = source_indices[key]
                tgt_idx = target_indices[key]
                env_action_np[:, tgt_idx] = gr00t_action_cat[:, src_idx]

            env_action = torch.tensor(env_action_np, device=env.unwrapped.device).unsqueeze(1)  # (16, 1, 28)
            
            actions = env_action[0] # use first result


            # apply actions
            obs, _, _, _, _ = env.step(actions)
            
            print("INPUT:",actions)
            print("OUTPUT:",obs["policy"]["processed_actions"])

            # ===================|
            if args_cli.save_img:
                rgb_image = obs["policy"]["rgb_image"]  # shape: (1, 480, 640, 3)
                rgb_image = rgb_image.squeeze(0)  # remove batch dim (480, 640, 3)
                rgb_image = rgb_image.cpu().numpy()  # from cuda to cpu
                rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR) # RGB to CV2 BGR format
                cv2.imwrite("output/frame_preview.png", rgb_image)

    # close the simulator
    env.close()



if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
