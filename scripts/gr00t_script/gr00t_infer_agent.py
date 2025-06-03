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
parser.add_argument("--task", type=str, default="Isaac-PickPlace-G1-Abs-v0", help="Name of the task.")
parser.add_argument("--port", type=int, help="Port number for the server.", default=5555)
parser.add_argument("--host", type=str, help="Host address for the server.", default="localhost")
parser.add_argument("--save_img", action="store_true", default=False, help="Save the data from camera RGB image.")

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
import time

"""gr00t integration"""
from gr00t.eval.robot import RobotInferenceClient
from utils.joint_mapper import JointMapper # Import the new JointMapper
from isaaclab_tasks.manager_based.manipulation.pick_place_g1.mdp import get_right_eef_pos, get_right_eef_quat

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

    # Initialize the JointMapper - needed early if we use its properties for settling action
    # Ensure robot_articulation is fetched before JointMapper initialization
    try:
        robot_articulation = env.unwrapped.scene.articulations["robot"]
    except KeyError:
        print("ERROR: Could not find robot articulation with key 'robot'. Exiting.")
        return

    # Option 1: Short loop for visual settling and UI responsiveness
    print("\n--- Allowing a few simulation ticks for visual settling after reset ---")
    for _ in range(100): # Adjust the number of ticks as needed
        if simulation_app.is_running():
            simulation_app.update()
        else:
            break # Exit if sim stopped
    print("--- Visual settling complete ---")
    
    # Perform a no-op step to refresh observations after the passive settling
    # Construct a specific joint target action for settling
    
    # Define your desired initial joint positions
    # (Only specify the ones you want to actively set, others will be 0.0
    #  or you can fetch them from env.unwrapped.scene.robot.init_state.joint_pos if needed)
    desired_settling_joint_positions = {
        # right-arm
        "right_shoulder_pitch_joint": 0.65,
        "right_shoulder_roll_joint": 0.0, # Explicitly 0.0 as per your example
        "right_shoulder_yaw_joint": 0.0,
        "right_elbow_joint": -0.65,
        "right_wrist_yaw_joint": -0.5,
        "right_wrist_roll_joint": 0.0,
        "right_wrist_pitch_joint": 0.0,
        # left-arm (all zeros as per your example)
        "left_shoulder_pitch_joint": 0.0,
        "left_shoulder_roll_joint": 0.0,
        "left_shoulder_yaw_joint": 0.0,
        "left_elbow_joint": 0.0, # Explicitly 0.0
        "left_wrist_yaw_joint": 0.0,
        "left_wrist_roll_joint": 0.0,
        "left_wrist_pitch_joint": 0.0,
        # Hand joints will default to 0.0 unless specified
    }

    # Get the order of joints expected by the JointPositionActionCfg
    # This comes from the env_cfg, which JointMapper also uses.
    # We need JointMapper initialized to get this.
    # Temporarily create a JointMapper instance if not already done, or ensure it's initialized before this.
    # Assuming joint_mapper will be initialized later, for now, let's get it directly from env_cfg
    action_joint_names_ordered = env_cfg.actions.pink_ik_cfg.joint_names # This is JointPositionActionCfg's joint_names
    settling_action_values = np.zeros(len(action_joint_names_ordered), dtype=np.float32)
    for i, name in enumerate(action_joint_names_ordered):
        settling_action_values[i] = desired_settling_joint_positions.get(name, 0.0) # Default to 0.0 if not specified

    settling_action_tensor = torch.tensor(settling_action_values, dtype=torch.float32, device=args_cli.device).unsqueeze(0)
    obs, _, _, _, _ = env.step(settling_action_tensor) # This updates obs

    print("\n--- Initial Observation After Reset & Visual Settling ---")
    # Get EEF poses using the helper functions (env.unwrapped provides direct access to the underlying env)
    initial_right_eef_pos_w = get_right_eef_pos(env.unwrapped)[0].cpu().numpy()
    initial_right_eef_quat_wxyz_w = get_right_eef_quat(env.unwrapped)[0].cpu().numpy()
    print(f"Initial Right EEF Position (world): {initial_right_eef_pos_w}")
    print(f"Initial Right EEF Quaternion (wxyz, world): {initial_right_eef_quat_wxyz_w}")

    if "cube_pos" in obs["policy"]:
        initial_cube_pos_w = obs['policy']['cube_pos'][0].cpu().numpy()
        print(f"Initial Cube Position (world): {initial_cube_pos_w}")
        print(f"Initial Relative Position (Cube - RightEEF): {initial_cube_pos_w - initial_right_eef_pos_w}")
    if "cube_rot" in obs["policy"]:
        print(f"Initial Cube Rotation (wxyz, world): {obs['policy']['cube_rot'][0].cpu().numpy()}")
    print("-------------------------------------\n")
    
    # Print specific joint angles AFTER the settling loop and obs refresh
    robot_joint_names = env.unwrapped.scene.articulations["robot"].joint_names
    robot_joint_pos_after_settle = obs["policy"]["robot_joint_pos"][0].cpu().numpy()
    
    try:
        rsp_idx = robot_joint_names.index("right_shoulder_pitch_joint")
        re_idx = robot_joint_names.index("right_elbow_joint")
        print(f"POST-SETTLE Right Shoulder Pitch Joint Angle: {robot_joint_pos_after_settle[rsp_idx]:.4f} (Expected from cfg: 0.65)")
        print(f"POST-SETTLE Right Elbow Joint Angle: {robot_joint_pos_after_settle[re_idx]:.4f} (Expected from cfg: -0.65)")
    except ValueError as e:
        print(f"Error finding joint names for printing: {e}")
    print("-------------------------------------\n")


    """gr00t inference client"""
    policy_client = RobotInferenceClient(host=args_cli.host, port=args_cli.port)

    modality_configs = policy_client.get_modality_config()
    print(f"Retrieved modality keys: {list(modality_configs.keys())}")
    
    # Initialize the JointMapper (robot_articulation should already be defined)
    joint_mapper = JointMapper(env_cfg=env_cfg, robot_articulation=robot_articulation)

    # simulate environment
    loop_counter = 0
    while simulation_app.is_running():
        loop_counter += 1
        # run everything in inference mode
        with torch.inference_mode():
            # --- 1. Process observations ---
            robot_joint_pos = obs["policy"]["robot_joint_pos"].cpu().numpy().astype(np.float64)
            isaac_robot_joint_pos_flat = robot_joint_pos[0]  # (num_envs, num_all_robot_joints) -> (num_all_robot_joints,)
            rgb_image = obs["policy"]["rgb_image"].cpu().numpy().astype(np.uint8)  # (1, 480, 640, 3)

            # --- 2. Prepare GR00T observation ---
            gr00t_state_obs = joint_mapper.map_isaac_obs_to_gr00t_state(isaac_robot_joint_pos_flat)
            gr00t_obs = {
                "video.camera": rgb_image,
                "annotation.human.action.task_description": ["pick and place a cube"],
                **gr00t_state_obs
            }
            
            # --- 3. Query GR00T policy server ---
            # time_start = time.time()
            gr00t_action = policy_client.get_action(gr00t_obs)
            # print(f"Total time taken to get action from server: {time.time() - time_start} seconds")

            # --- 4. Map GR00T action to Isaac action gr00t_action is a dict, e.g., {"action.left_arm": (prediction_horizon, 7), ...} ---
            env_action_values_single_step = joint_mapper.map_gr00t_action_to_isaac_action(gr00t_action)
            actions = torch.tensor(env_action_values_single_step, dtype=torch.float32, device=env.unwrapped.device).unsqueeze(0)

            # --- 5. Step environment ---
            obs, _, _, _, _ = env.step(actions)
            # print("INPUT:", actions)
            # print("OUTPUT:", obs["policy"]["processed_actions"])

            # --- 6. Optionally save image ---
            if args_cli.save_img:
                # rgb_image = obs["policy"]["rgb_image"].squeeze(0).cpu().numpy()  # (1, 480, 640, 3) -> (480, 640, 3)
                # rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                # cv2.imwrite("output/frame_preview.png", rgb_image)
                output_dir = "output"
                os.makedirs(output_dir, exist_ok=True)
                # Assuming rgb_image was fetched for GR00T obs preparation earlier in the loop
                # If obs["policy"]["rgb_image"] is needed here, ensure it's current
                img_to_save = obs["policy"]["rgb_image"].squeeze(0).cpu().numpy()  # (1, H, W, C) -> (H, W, C)
                img_bgr = cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(output_dir, f"frame_{loop_counter:05d}.png"), img_bgr)
            
            # simulation_app.update()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
