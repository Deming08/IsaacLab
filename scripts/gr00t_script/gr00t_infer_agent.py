# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to an environment with random action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Gr00t agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Cabinet-Pour-G1-Abs-v0",
    choices=["Isaac-Cabinet-Pour-G1-Abs-v0", "Isaac-Stack-Cube-G1-Abs-v0", "Isaac-PickPlace-G1-Abs-v0"],
    help="Name of the task. Options: 'Isaac-Cabinet-Pour-G1-Abs-v0', 'Isaac-Stack-Cube-G1-Abs-v0', 'Isaac-PickPlace-G1-Abs-v0'."
)
parser.add_argument("--port", type=int, help="Port number for the server.", default=5555)
parser.add_argument("--host", type=str, help="Host address for the server.", default="localhost")
parser.add_argument("--save_video", action="store_true", default=False, help="Save the data from camera RGB image.")

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
from typing import cast
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.devices import Se3Keyboard

# PLACEHOLDER: Extension template (do not remove this comment)
"""Data collection setup"""
import cv2
import os
import numpy as np
import time

"""gr00t integration"""
from gr00t.eval.robot import RobotInferenceClient
from utils.joint_mapper import JointMapper

import carb
carb_settings_iface = carb.settings.get_settings()
carb_settings_iface.set_bool("/gr00t/use_joint_space", True)

# Determine TASK_DESCRIPTION based on the selected task
if args_cli.task == "Isaac-PickPlace-G1-Abs-v0":
    TASK_DESCRIPTION = ["pick and sort a red or blue can"]
elif args_cli.task == "Isaac-Stack-Cube-G1-Abs-v0":
    TASK_DESCRIPTION = ["stack the cubes in the order of red, green and blue."]
elif args_cli.task == "Isaac-Cabinet-Pour-G1-Abs-v0":
    TASK_DESCRIPTION = ["Perform the default behavior."]

STABILIZATION_STEPS = 30

def run_stabilization(env, idle_actions_tensor):
    """
    Runs stabilization steps by holding a default joint pose and returns the final observation.
    """
    print(f"\n[INFO] Stabilizing robot to default joint positions for {STABILIZATION_STEPS} steps...")
    for _ in range(STABILIZATION_STEPS):
        obs, _, _, _, _ = env.step(idle_actions_tensor)
    print("[INFO] Stabilization complete.")
    
    return obs


def main():
    """GR00T actions agent with Isaac Lab environment."""

    # Set numpy print options to display floats with 3 decimal places
    np.set_printoptions(precision=3, suppress=True, floatmode='fixed')

    # create environment with configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    env = cast(ManagerBasedRLEnv, gym.make(args_cli.task, cfg=env_cfg).unwrapped)
    
    # Calculate FPS for video saving from env_cfg
    if args_cli.save_video:
        video_fps = 1.0 / (env_cfg.sim.dt * env_cfg.decimation)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    obs, _ = env.reset()

    # Initialize teleop interface (just for env_reset, not for robot eef control)
    do_env_reset = False
    def reset_env_and_player():
        nonlocal do_env_reset
        do_env_reset = True
 
    teleop_interface = Se3Keyboard()
    teleop_interface.add_callback("R", reset_env_and_player)

    print("\n==== Teleoperation Interface Controls ====")
    print("  R: Reset the environment.")
    print("  M: Switch task scene and reset environment.(NOT YET IMPLEMENTED)")
    print("========================================\n")
    teleop_interface.reset()
    
    """gr00t inference client"""
    policy_client = RobotInferenceClient(host=args_cli.host, port=args_cli.port)
    modality_configs = policy_client.get_modality_config()
    print(f"Retrieved modality keys: {list(modality_configs.keys())}")
    
    # Initialize the JointMapper
    robot_articulation = env.scene.articulations["robot"]
    joint_mapper = JointMapper(env_cfg=env_cfg, robot_articulation=robot_articulation)

    # Create the default joint action tensor for stabilization ( env_cfg.actions.pink_ik_cfg is JointPositionActionCfg due to /gr00t/use_joint_space = True )
    action_joint_names_list = env_cfg.actions.pink_ik_cfg.joint_names
    default_joint_positions_dict = env_cfg.scene.robot.init_state.joint_pos
    default_joint_action_np = np.zeros(len(action_joint_names_list), dtype=np.float32)
    for i, joint_name in enumerate(action_joint_names_list):
        default_joint_action_np[i] = default_joint_positions_dict.get(joint_name, 0.0)
    default_idle_actions_tensor = torch.tensor(default_joint_action_np, dtype=torch.float32, device=env.device).unsqueeze(0)

    # Initial stabilization run
    obs = run_stabilization(env, default_idle_actions_tensor)
    episode_start_sim_time = env.sim.current_time # Initialize after first stabilization

    # simulate environment
    episode_counter, step_counter = 0, 0
    video_writer = None
    image_list = []
    output_dir = "output/cabinet_pour_n1.5_500k"
    MAX_EPS_NUM = 1000

    while simulation_app.is_running():
        
        # run everything in inference mode
        with torch.inference_mode():
            if do_env_reset:
                obs, _ = env.reset()
                teleop_interface.reset()
                print("[INFO] The environment was reset due to detection of [R] being pressed.")
                do_env_reset = False
                obs = run_stabilization(env, default_idle_actions_tensor) # Run stabilization again
                episode_start_sim_time = env.sim.current_time # Reset episode start time
                episode_counter += 1
                step_counter = 0      # Reset step_counter for the new episode
                image_list = []
        
            # --- 1. Process observations ---
            # obs tensors have a batch dim of 1, even with unwrapped env
            robot_joint_pos = obs["policy"]["robot_joint_pos"].cpu().numpy().astype(np.float64)
            isaac_robot_joint_pos_flat = robot_joint_pos[0]  # Index to get 1D array (num_joints,)
            
            # The rgb_image from the *current* obs is what GR00T needs
            rgb_image = obs["policy"]["rgb_image"].cpu().numpy().astype(np.uint8)  # Shape (1, H, W, C)


            # --- 2. Prepare GR00T observation ---
            gr00t_state_obs = joint_mapper.map_isaac_obs_to_gr00t_state(isaac_robot_joint_pos_flat)
            gr00t_obs = {
                "video.camera": rgb_image, # Pass as is; shape (1, H, W, C) is a valid 4D input for GR00T
                "annotation.human.task_description": TASK_DESCRIPTION,
                **gr00t_state_obs
            }
            
            # --- 3. Query GR00T policy server ---
            time_start = time.time()
            gr00t_action = policy_client.get_action(gr00t_obs)
            get_action_time = time.time() - time_start

            # --- 4. Map GR00T action to Isaac action gr00t_action is a dict, e.g., {"action.left_arm": (prediction_horizon, 7), ...} ---
            env_action_values_fully_step = joint_mapper.map_gr00t_action_to_isaac_action(gr00t_action)
            actions_seqs = torch.tensor(env_action_values_fully_step, dtype=torch.float32, device=env.device).unsqueeze(1) # (16, 1, 28)
            
            # --- 5. Step environment ---
            for action in actions_seqs: # Step every predicted action
                obs, _, terminated, truncated, _ = env.step(action)    # (obs, reward, terminated, truncated, info)
                rgb_image = obs["policy"]["rgb_image"].cpu().numpy().astype(np.uint8)
                image_list.append(rgb_image[0])
                step_counter += 1
                # Interrupt action sequence step
                if terminated or truncated: break

  
            current_sim_time = env.sim.current_time
            relative_episode_time = current_sim_time - episode_start_sim_time

            # print(f"Ep {episode_counter} | Step {step_counter} | SimTime {relative_episode_time:.2f}s: Inference: {get_action_time:.3f}s, "
            #     f"Right EE Pos/Quat: {right_eef_pos}, {right_eef_quat}, Object Pos: {target_object_pos}")
            print(f"Ep {episode_counter} | Step {step_counter} | SimTime {relative_episode_time:.2f}s: Inference: {get_action_time:.3f}s")
            
            # --- 6. Check for termination and reset if necessary ---
            if terminated or truncated:
                print(f"Episode {episode_counter} finished after {step_counter} steps (Terminated: {terminated}, Truncated: {truncated}).")
                if args_cli.save_video and terminated: # or other condition (currently only records success/terminated)

                    os.makedirs(output_dir, exist_ok=True)
                    video_path = os.path.join(output_dir, f"episode_{episode_counter:03d}.mp4")
                    fourcc = cv2.VideoWriter.fourcc(*'mp4v') # Codec for .mp4
                    # rgb_image shape is (1, H, W, C), so we index dimensions 1 and 2
                    frame_height, frame_width = rgb_image.shape[1], rgb_image.shape[2]
                    video_writer = cv2.VideoWriter(video_path, fourcc, video_fps, (frame_width, frame_height))

                    cv2.imwrite(os.path.join(output_dir, f"frame_ep{episode_counter:03d}_step{step_counter:05d}.png"), rgb_image[0])
                    for img in image_list:
                        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        video_writer.write(img_bgr)
                    video_writer.release()
                    video_writer = None

                obs, _ = env.reset()  # Reset the environment
                obs = run_stabilization(env, default_idle_actions_tensor) # Run stabilization again
                episode_start_sim_time = env.sim.current_time # Reset episode start time
                episode_counter += 1
                step_counter = 0      # Reset step_counter for the new episode
                image_list = []

            if episode_counter>=MAX_EPS_NUM: break

    # close the simulator
    if video_writer is not None: # Release writer if simulation ends mid-episode
        video_writer.release()
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
