# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to an environment with random action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Data collection for Isaac Lab environments.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--save_data", action="store_true", default=True, help="Save video and compose data to parquet.",)
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Stack-Cube-G1-Abs-v0", # Changed default task
    choices=["Isaac-Stack-Cube-G1-Abs-v0", "Isaac-BlockStack-G1-Abs-v0", "Isaac-PickPlace-G1-Abs-v0"], # Updated choices
    help="Name of the task. Options: 'Isaac-BlockStack-G1-Abs-v0', 'Isaac-PickPlace-G1-Abs-v0'."
)
parser.add_argument(
    "--initial_view",
    type=str,
    default="perspective",
    choices=["perspective", "camera", "front", "right"],
    help="Initial camera view for the viewport: 'perspective', 'camera' (robot's head), 'front', or 'right'."
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
from isaaclab_tasks.utils import parse_env_cfg
import omni.usd
from pxr import Gf, Sdf
from omni.kit.viewport.utility import get_viewport_from_window_name
from omni.kit.viewport.utility.camera_state import ViewportCameraState

# PLACEHOLDER: Extension template (do not remove this comment)
"""Data collection setup"""
import cv2
import numpy as np
import time

import warnings
from qpsolvers.warnings import SparseConversionWarning
# Suppress specific warnings from qpsolvers
warnings.filterwarnings("ignore", category=SparseConversionWarning, module="qpsolvers.conversions.ensure_sparse_matrices")

""" Customized modules """
from utils.trajectory_player import TrajectoryPlayer
from utils.data_collector_util import DataCollector
# Conditionally import task_done based on the task, or import directly if script is specific
if "Stack-Cube-G1" in args_cli.task or "BlockStack-G1" in args_cli.task:
    import isaaclab_tasks.manager_based.manipulation.stack_g1 # noqa: F401
    from isaaclab_tasks.manager_based.manipulation.stack_g1.mdp.terminations import task_done
elif "PickPlace-G1" in args_cli.task:
    import isaaclab_tasks.manager_based.manipulation.pick_place_g1 # noqa: F401
    from isaaclab_tasks.manager_based.manipulation.pick_place_g1.mdp.terminations import task_done

""" Constants """
# EPISODE_FRAMES_LEN = STEPS_PER_MOVEMENT_SEGMENT * 4 + STEPS_PER_GRASP_SEGMENT * 2 # frames (steps)
STEPS_PER_MOVEMENT_SEGMENT = 100  # 4 segments for movement
STEPS_PER_GRASP_SEGMENT = 50  # 2 segments for grasp
STABILIZATION_STEPS = 30 # Step 30 times for stabilization after env.reset()
FPS = 30  # In pickplace_g1_env_cfg.py, sim.dt * decimation = 1/60 * 2 = 1/30
MAX_EPISOIDES = 3000  # Limit to 1000 iterations for data collection

# parquet data setup
DATASET_PATH = "datasets/gr00t_collection/G1_dataset/"
DEFAULT_OUTPUT_VIDEO_DIR = f"{DATASET_PATH}videos/chunk-000/observation.images.camera"
DEFAULT_OUTPUT_DATA_DIR = f"{DATASET_PATH}data/chunk-000"

FAILED_DATASET_PATH = "datasets/gr00t_collection/G1_dataset_failed/"
FAILED_OUTPUT_VIDEO_DIR = f"{FAILED_DATASET_PATH}videos/chunk-000/observation.images.camera"
FAILED_OUTPUT_DATA_DIR = f"{FAILED_DATASET_PATH}data/chunk-000"
        

def set_initial_viewport_camera(viewport, env, view_type="camera"):
    """
    Sets the active viewport camera based on the specified view_type.

    Args:
        viewport: The viewport window object.
        view_type (str): The type of fixed view to set ("front", "right", etc.).
                         Defaults to "front".
    """
    if view_type == "camera":
        if "camera" in env.unwrapped.scene.sensors:
            camera_path = env.unwrapped.scene.sensors["camera"]._view.prim_paths[0]
            viewport.set_active_camera(camera_path)
            print(f"[INFO] Set active viewport camera to: robot's head camera at {camera_path}")
        else:
            print("[WARNING] Robot's head camera not found in scene. Defaulting to perspective view.")
            viewport.set_active_camera("/OmniverseKit_Persp")
    elif view_type in ["front", "right"]:
        stage = omni.usd.get_context().get_stage()
        camera_prim_path = "/World/CustomCamera"
        
        # Define eye and target views based on view_type
        # Target is approximately the center of the table/cube area.
        target = Gf.Vec3d(0.0, 0.0, 0.9)
        # eye is chosen to observe the manipulation area (robot, table, cubes)
        if view_type == "right":
            eye = Gf.Vec3d(0.3, -1.5, 0.85) # Looking from negative Y (robot's right)
        else: # view_type == "front":
            eye = Gf.Vec3d(1.5, 0.0, 0.85) # Looking from X

        # Create the camera prim - These values are typical for a camera in Isaac Lab (e.g., from h1_locomotion.py)
        camera_prim = stage.DefinePrim(camera_prim_path, "Camera")
        camera_prim.GetAttribute("focalLength").Set(10.5)
        # Ensure the center of interest attribute exists and is valid
        coi_prop = camera_prim.GetProperty("omni:kit:centerOfInterest")
        if not coi_prop or not coi_prop.IsValid():
            camera_prim.CreateAttribute(
                "omni:kit:centerOfInterest", Sdf.ValueTypeNames.Vector3d, True, Sdf.VariabilityUniform
            ).Set(Gf.Vec3d(0, 0, -10)) # Default value from h1_locomotion.py

        # Set the camera state using eye, target, and an explicit up vector (Z-up for Isaac Sim)
        camera_state = ViewportCameraState(camera_prim_path, viewport)
        camera_state.set_position_world(eye, True)
        camera_state.set_target_world(target, True)
        viewport.set_active_camera(camera_prim_path)
        print(f"[INFO] Set active viewport camera to: {view_type} view at {camera_prim_path}")
    else:
        viewport.set_active_camera("/OmniverseKit_Persp")



def main():
    # Record start time for summary
    start_time = time.time()

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
    
    
    # create environment configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    
    # Disable the termination term for data collection.
    env_cfg.terminations = None

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # Set the initial camera view based on the argument
    set_initial_viewport_camera(get_viewport_from_window_name("Viewport"), env, view_type=args_cli.initial_view)

    # # print info (this is vectorized environment)
    # print(f"[INFO]: Gym observation space: {env.observation_space}")
    # print(f"[INFO]: Gym action space: {env.action_space}")
    
    # reset environment
    obs, _ = env.reset()
    # Pass initial observation to TrajectoryPlayer to set default poses
    trajectory_player = TrajectoryPlayer(env.unwrapped, initial_obs=obs, steps_per_movement_segment=100, steps_per_grasp_segment=50)    # 30 fps
    # Get the idle action based on the initial reset pose
    idle_action_np = trajectory_player.get_idle_action_np()
    idle_actions_tensor = torch.tensor(idle_action_np, dtype=torch.float, device=args_cli.device).repeat(env.unwrapped.num_envs, 1)

    # Create the data collector
    data_collector_success = DataCollector(output_video_dir=DEFAULT_OUTPUT_VIDEO_DIR, output_data_dir=DEFAULT_OUTPUT_DATA_DIR, fps=FPS)
    data_collector_failed = DataCollector(output_video_dir=FAILED_OUTPUT_VIDEO_DIR, output_data_dir=FAILED_OUTPUT_DATA_DIR, fps=FPS)

    # Buffers for the current episode's data
    current_frames, current_obs_list, current_action_list = [], [], []
    successful_episodes_collected_count = 0
    current_attempt_number = 0 # Starts at 0, increments to 1 for the first attempt
    should_generate_and_play_trajectory = True
    
    # simulate environment
    while simulation_app.is_running() and successful_episodes_collected_count < MAX_EPISOIDES:
        with torch.inference_mode():
        
            if should_generate_and_play_trajectory:
                obs, _ = env.reset()
                # Step environment with idle action to stabilize after reset
                for _ in range(STABILIZATION_STEPS):
                    obs, _, _, _, _ = env.step(idle_actions_tensor)
                
                print(f"\n===== Start the attemp {current_attempt_number} =====")
                current_attempt_number += 1
                # 0. Clear external buffers for the new attempt
                current_frames, current_obs_list, current_action_list = [], [], []

                # 1. Generate the full trajectory by passing the current observation
                if "Stack-Cube-G1" in args_cli.task or "BlockStack-G1" in args_cli.task:
                    trajectory_player.generate_auto_stack_cubes_trajectory(obs=obs)
                elif "PickPlace-G1" in args_cli.task:
                    trajectory_player.generate_auto_grasp_pick_place_trajectory(obs=obs)
                # 2. Prepare the playback trajectory
                trajectory_player.prepare_playback_trajectory()
                # 3. Set to False to play this trajectory
                should_generate_and_play_trajectory = False

            actions = idle_actions_tensor.clone()  # Initialize actions as idle actions if not set
            if trajectory_player.is_playing_back:
                playback_action_tuple = trajectory_player.get_formatted_action_for_playback()
                if playback_action_tuple is not None:
                    action_array_28D_np = playback_action_tuple[0]
                    actions = torch.tensor(action_array_28D_np, dtype=torch.float, device=args_cli.device).repeat(env.unwrapped.num_envs, 1)
                else: # Playback finished
                    current_attempt_was_successful = task_done(env.unwrapped).cpu().numpy()[0]

                    if current_attempt_was_successful:
                        if args_cli.save_data: # Only save if --save_data is enabled
                            data_collector_success.save_episode(current_frames, current_obs_list, current_action_list)
                        successful_episodes_collected_count += 1
                    else: # not current_attempt_was_successful
                        if args_cli.save_data:
                            data_collector_failed.save_episode(current_frames, current_obs_list, current_action_list)
                        
                    # Use idle action after trajectory, that is actions = idle_actions_tensor.clone() 
                    should_generate_and_play_trajectory = True                    
                    print(f"{successful_episodes_collected_count}/{current_attempt_number} ({successful_episodes_collected_count/current_attempt_number * 100:.2f}%): Attempt {current_attempt_number} result: {'Successful' if current_attempt_was_successful else 'Failed'}")
                
            # apply actions
            obs, _, _, _, _ = env.step(actions)

            # Data extraction for saving
            robot_joint_state = obs["policy"]["robot_joint_pos"].cpu().numpy().flatten().astype(np.float64)
            processed_action = obs["policy"]["processed_actions"].cpu().numpy().flatten().astype(np.float64)
            rgb_image_np = obs["policy"]["rgb_image"].squeeze(0).cpu().numpy()  # shape: (1, 480, 640, 3) -> (480, 640, 3);from cuda to cpu
            rgb_image_bgr = cv2.cvtColor(rgb_image_np, cv2.COLOR_RGB2BGR) # RGB to CV2 BGR format

            data_state = robot_joint_state[joint_id]
            data_action = np.zeros(28)
            # Swap value to match joint action orders
            for tar_i, src_i in enumerate(target_idx):
                data_action[tar_i] = processed_action[src_i]

            # print("State:",data_state)
            # print("Action:",data_action)

            # Append data if saving and currently in an active trajectory attempt
            if args_cli.save_data and not should_generate_and_play_trajectory:
                current_frames.append(rgb_image_bgr)
                current_obs_list.append(data_state)
                current_action_list.append(data_action)

    # Calculate total time taken
    end_time = time.time()
    total_time_taken = end_time - start_time

    # Print summary
    print("\n\n" + "=" * 50)
    print("Data Collection Summary")
    print("=" * 50)
    print(f"Task: {args_cli.task}")
    print(f"Total successful episodes collected: {successful_episodes_collected_count}")
    print(f"Total failed episodes collected: {data_collector_failed.episode_index}")
    print(f"Total attempts made: {current_attempt_number}")
    if current_attempt_number > 0:
        success_rate = (successful_episodes_collected_count / current_attempt_number) * 100
        print(f"Success rate: {success_rate:.2f}%")
    else:
        print("Success rate: N/A (no attempts made)")
    print(f"Maximum episodes to collect: {MAX_EPISOIDES}")
    print(f"Data saved to: {DATASET_PATH}")
    total_time_taken_hours = total_time_taken // 3600
    total_time_taken_minutes = (total_time_taken % 3600) // 60
    total_time_taken_seconds = total_time_taken % 60
    print(f"Total time taken: {total_time_taken_hours:.0f}:{total_time_taken_minutes:.0f}:{total_time_taken_seconds:.0f}")
    print("=" * 50 + "\n")

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
