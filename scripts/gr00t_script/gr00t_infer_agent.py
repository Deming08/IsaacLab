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
    default="Isaac-PickPlace-G1-Abs-v0",
    choices=["Isaac-BlockStack-G1-Abs-v0", "Isaac-PickPlace-G1-Abs-v0"],
    help="Name of the task. Options: 'Isaac-BlockStack-G1-Abs-v0', 'Isaac-PickPlace-G1-Abs-v0'."
)
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
from utils.joint_mapper import JointMapper

import carb
carb_settings_iface = carb.settings.get_settings()
carb_settings_iface.set_bool("/gr00t/use_joint_space", True)

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

    # create environment configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    
    # Calculate FPS for video saving from env_cfg
    if args_cli.save_img:
        video_fps = 1.0 / (env_cfg.sim.dt * env_cfg.decimation)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    obs, _ = env.reset()

    """gr00t inference client"""
    policy_client = RobotInferenceClient(host=args_cli.host, port=args_cli.port)
    modality_configs = policy_client.get_modality_config()
    print(f"Retrieved modality keys: {list(modality_configs.keys())}")
    
    # Initialize the JointMapper
    robot_articulation = env.unwrapped.scene.articulations["robot"]
    joint_mapper = JointMapper(env_cfg=env_cfg, robot_articulation=robot_articulation)

    # Create the default joint action tensor for stabilization ( env_cfg.actions.pink_ik_cfg is JointPositionActionCfg due to /gr00t/use_joint_space = True )
    action_joint_names_list = env_cfg.actions.pink_ik_cfg.joint_names
    default_joint_positions_dict = env_cfg.scene.robot.init_state.joint_pos
    default_joint_action_np = np.zeros(len(action_joint_names_list), dtype=np.float32)
    for i, joint_name in enumerate(action_joint_names_list):
        default_joint_action_np[i] = default_joint_positions_dict.get(joint_name, 0.0)
    default_idle_actions_tensor = torch.tensor(default_joint_action_np, dtype=torch.float32, device=env.unwrapped.device).unsqueeze(0)

    # Initial stabilization run
    obs = run_stabilization(env, default_idle_actions_tensor)
    episode_start_sim_time = env.unwrapped.sim.current_time # Initialize after first stabilization

    # simulate environment
    episode_counter, step_counter = 0, 0
    video_writer = None
    while simulation_app.is_running():
        
        # run everything in inference mode
        with torch.inference_mode():
            # --- 1. Process observations ---
            robot_joint_pos = obs["policy"]["robot_joint_pos"].cpu().numpy().astype(np.float64)
            isaac_robot_joint_pos_flat = robot_joint_pos[0]  # (num_envs, num_all_robot_joints) -> (num_all_robot_joints,)
            
            # The rgb_image from the *current* obs is what GR00T needs
            rgb_image = obs["policy"]["rgb_image"].cpu().numpy().astype(np.uint8)  # (1, 480, 640, 3)

            # Initialize video writer at the start of a new episode (step_counter == 0 after reset and stabilization)
            if args_cli.save_img and step_counter == 0 and episode_counter >= 0: # episode_counter check ensures it's after first stabilization
                output_dir = "output"
                os.makedirs(output_dir, exist_ok=True)
                video_path = os.path.join(output_dir, f"episode_{episode_counter:03d}.mp4")
                fourcc = cv2.VideoWriter.fourcc(*'mp4v') # Codec for .mp4
                # Assuming rgb_image shape is (1, H, W, C), squeeze it to (H, W, C)
                frame_height, frame_width = rgb_image.shape[1], rgb_image.shape[2]
                video_writer = cv2.VideoWriter(video_path, fourcc, video_fps, (frame_width, frame_height))

            # --- 2. Prepare GR00T observation ---
            gr00t_state_obs = joint_mapper.map_isaac_obs_to_gr00t_state(isaac_robot_joint_pos_flat)
            gr00t_obs = {
                "video.camera": rgb_image,
                "annotation.human.action.task_description": ["pick and sort a red or blue can"],
                **gr00t_state_obs
            }
            
            # --- 3. Query GR00T policy server ---
            time_start = time.time()
            gr00t_action = policy_client.get_action(gr00t_obs)
            get_action_time = time.time() - time_start

            # --- 4. Map GR00T action to Isaac action gr00t_action is a dict, e.g., {"action.left_arm": (prediction_horizon, 7), ...} ---
            env_action_values_fully_step = joint_mapper.map_gr00t_action_to_isaac_action(gr00t_action)
            actions_seqs = torch.tensor(env_action_values_fully_step, dtype=torch.float32, device=env.unwrapped.device).unsqueeze(1) # (16, 1, 28)
            
            # --- 5. Step environment ---
            for action in actions_seqs: # Step every predicted action
                obs, _, terminated, truncated, _ = env.step(action)    # (obs, reward, terminated, truncated, info)
                step_counter += 1
                # Interrupt action sequence step
                if terminated or truncated: break

            # Log data from the new observation
            right_eef_pos = obs["policy"]["right_eef_pos"][0].cpu().numpy()
            right_eef_quat = obs["policy"]["right_eef_quat"][0].cpu().numpy()
            target_object_obs = obs["policy"]["target_object_pose"][0].cpu().numpy()
            target_object_pos = target_object_obs[:3]
            
            current_sim_time = env.unwrapped.sim.current_time
            relative_episode_time = current_sim_time - episode_start_sim_time

            print(f"Ep {episode_counter} | Step {step_counter} | SimTime {relative_episode_time:.2f}s: Inference: {get_action_time:.3f}s, "
                f"Right EE Pos/Quat: {right_eef_pos}, {right_eef_quat}, Object Pos: {target_object_pos}")
            
            # --- 6. Optionally save image ---
            if args_cli.save_img:
                output_dir = "output"
                os.makedirs(output_dir, exist_ok=True)
                # Squeeze and convert to BGR for saving frame and video
                img_bgr = cv2.cvtColor(rgb_image.squeeze(0), cv2.COLOR_RGB2BGR) # Shape (H, W, C)
                cv2.imwrite(os.path.join(output_dir, f"frame_ep{episode_counter:03d}_step{step_counter:05d}.png"), img_bgr)
                
                # Write frame to video
                if video_writer is not None:
                    video_writer.write(img_bgr)

            # --- 7. Check for termination and reset if necessary ---
            if terminated or truncated:
                print(f"Episode {episode_counter} finished after {step_counter} steps (Terminated: {terminated}, Truncated: {truncated}).")
                if video_writer is not None:
                    video_writer.release()
                    video_writer = None
                obs, _ = env.reset()  # Reset the environment
                obs = run_stabilization(env, default_idle_actions_tensor) # Run stabilization again
                episode_start_sim_time = env.unwrapped.sim.current_time # Reset episode start time
                episode_counter += 1
                step_counter = 0      # Reset step_counter for the new episode

    # close the simulator
    if video_writer is not None: # Release writer if simulation ends mid-episode
        video_writer.release()
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
