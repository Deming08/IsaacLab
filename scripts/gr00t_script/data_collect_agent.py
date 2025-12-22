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
    default="Isaac-Cabinet-Pour-G1-Abs-v0",
    choices=["Isaac-Cabinet-Pour-G1-Abs-v0", "Isaac-Stack-Cube-G1-Abs-v0", "Isaac-PickPlace-G1-Abs-v0"],
    help="Name of the task. Options: 'Isaac-Cabinet-Pour-G1-Abs-v0', 'Isaac-Stack-Cube-G1-Abs-v0', 'Isaac-PickPlace-G1-Abs-v0'."
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
from typing import cast
import gymnasium as gym
import torch
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.manager_based.manipulation.playground_g1.task_scenes.cabinet_pour.mdp import observations as playground_obs
import omni.usd
from pxr import Gf, Sdf
from omni.kit.viewport.utility import get_viewport_from_window_name
from omni.kit.viewport.utility.camera_state import ViewportCameraState
# Suppress specific warnings from qpsolvers
import warnings
from qpsolvers.warnings import SparseConversionWarning
warnings.filterwarnings("ignore", category=SparseConversionWarning, module="qpsolvers.conversions.ensure_sparse_matrices")

# PLACEHOLDER: Extension template (do not remove this comment)
import carb
carb_settings_iface = carb.settings.get_settings()

if "G1" in args_cli.task:
    G1_HAND_TYPE = "inspire"   # ["trihand", "inspire"]
    carb_settings_iface.set_string("/unitree_g1_env/hand_type", G1_HAND_TYPE)
    ROBOT_TYPE = "g1_"+ G1_HAND_TYPE
elif "OpenArm" in args_cli.task:
    ROBOT_TYPE = "openarm_leaphand"
    raise NotImplementedError("Temporarily unsupported for OpenArm.") #! temporary
else:
    raise NotImplementedError("Currently only for G1 or OpenArm.")
carb_settings_iface.set_string("/data_collect/robot_type", ROBOT_TYPE)

"""Data collection setup"""
import cv2
import numpy as np
import time

""" Customized modules """
from utils.trajectory_player import TrajectoryPlayer
from utils.trajectory_generators import (
    GraspPickPlaceTrajectoryGenerator,
    StackCubesTrajectoryGenerator,
    KitchenTasksTrajectoryGenerator,
)

from utils.data_collector_util import DataCollector
# Conditionally import task_done based on the task, or import directly if script is specific
if "Stack-Cube-G1" in args_cli.task or "BlockStack-G1" in args_cli.task:
    from isaaclab_tasks.manager_based.manipulation.playground_g1.task_scenes.cube_stack.mdp.terminations import task_done
elif "PickPlace-G1" in args_cli.task:   # 
    from isaaclab_tasks.manager_based.manipulation.playground_g1.task_scenes.can_sorting.mdp.terminations import task_done
elif "Cabinet-Pour-G1" in args_cli.task:
    from isaaclab_tasks.manager_based.manipulation.playground_g1.task_scenes.cabinet_pour.mdp.terminations import task_done


""" Constants """
# EPISODE_FRAMES_LEN = STEPS_PER_MOVEMENT_SEGMENT * 4 + STEPS_PER_GRASP_SEGMENT * 2 # frames (steps)
STEPS_PER_MOVEMENT_SEGMENT = 75  # 4 segments for movement
STEPS_PER_SHORTSHIFT_SEGMENT = 30  # Short-distance movement
STEPS_PER_GRASP_SEGMENT = 15  # Hand grasp
STABILIZATION_STEPS = 30 # Step 30 times for stabilization after env.reset()
FPS = 30  # In pickplace_g1_env_cfg.py, sim.dt * decimation = 1/120 * 4 = 1/30
MAX_EPISOIDES = 2000  # Limit to 1000 iterations for data collection
CABINET_POUR_STATES = ["OPEN_DRAWER", "PICK_AND_PLACE_MUG", "POUR_BOTTLE"]  # For 'Isaac-Cabinet-Pour-G1-Abs-v0'
START_STATE_INDEX = 0   # Flexible for starting from an index of the CABINET_POUR_STATES

# parquet data setup
DATASET_PATH = "datasets/gr00t_collection/G1_dataset/"
DEFAULT_OUTPUT_VIDEO_DIR = f"{DATASET_PATH}videos/chunk-000/observation.images.camera"
DEFAULT_OUTPUT_DATA_DIR = f"{DATASET_PATH}data/chunk-000"

FAILED_DATASET_PATH = "datasets/gr00t_collection/G1_dataset_failed/"
FAILED_OUTPUT_VIDEO_DIR = f"{FAILED_DATASET_PATH}videos/chunk-000/observation.images.camera"
FAILED_OUTPUT_DATA_DIR = f"{FAILED_DATASET_PATH}data/chunk-000"
        
# Unitree G1 joint indices in whole body 43 joint.
LEFT_ARM_INDICES = [11, 15, 19, 21, 23, 25, 27]
RIGHT_ARM_INDICES = [12, 16, 20, 22, 24, 26, 28]

# Action observation index mapping to the structure of gr00t
ACTION_INDICE = {
    "left_arm": [0, 2, 4, 6, 8, 10, 12],  # 11, 15, 19, 21, 23, 25, 27
    "right_arm": [1, 3, 5, 7, 9, 11, 13],  # 12, 16, 20, 22, 24, 26, 28
}

if G1_HAND_TYPE=="trihand":
    LEFT_HAND_INDICES = [31, 37, 41, 30, 36, 29, 35]
    RIGHT_HAND_INDICES = [34, 40, 42, 32, 38, 33, 39] 
    ACTION_INDICE["left_hand"] = [16, 22, 26, 15, 21, 14, 20]  # 31, 37, 41, 30, 36, 29, 35
    ACTION_INDICE["right_hand"] = [19, 25, 27, 17, 23, 18, 24]  # 34, 40, 42, 32, 38, 33, 39

elif G1_HAND_TYPE == "inspire":
    LEFT_HAND_INDICES = [29, 30, 31, 32, 33, 39, 40, 41, 42, 43, 49, 51]
    RIGHT_HAND_INDICES = [34, 35, 36, 37, 38, 44, 45, 46, 47, 48, 50, 52] 
    ACTION_INDICE["left_hand"] = [14, 15, 16, 17, 18, 28]
    ACTION_INDICE["right_hand"] = [19, 20, 21, 22, 23, 33]


JOINT_STATE_ID = LEFT_ARM_INDICES + RIGHT_ARM_INDICES + LEFT_HAND_INDICES + RIGHT_HAND_INDICES

TARGET_IDX = np.concatenate([
                ACTION_INDICE["left_arm"],
                ACTION_INDICE["right_arm"],
                ACTION_INDICE["left_hand"],
                ACTION_INDICE["right_hand"],
            ]).tolist()

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

    # create environment configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    
    # Disable the termination term for data collection.
    env_cfg.terminations = None

    # create environment
    env = cast(ManagerBasedRLEnv, gym.make(args_cli.task, cfg=env_cfg).unwrapped)

    # Set the initial camera view based on the argument
    set_initial_viewport_camera(get_viewport_from_window_name("Viewport"), env, view_type=args_cli.initial_view)
    
    # reset environment
    obs, _ = env.reset()
    # Pass initial observation to TrajectoryPlayer to set default poses
    trajectory_player = TrajectoryPlayer(env, initial_obs=obs, steps_per_movement_segment=STEPS_PER_MOVEMENT_SEGMENT, steps_per_grasp_segment=STEPS_PER_GRASP_SEGMENT, steps_per_shortshift_segment=STEPS_PER_SHORTSHIFT_SEGMENT)    # 30 fps
    # Get the idle action based on the initial reset pose
    idle_action_np = trajectory_player.get_idle_action_np()
    idle_actions_tensor = torch.tensor(idle_action_np, dtype=torch.float, device=args_cli.device).repeat(env.unwrapped.num_envs, 1) # type: ignore

    # Create the data collector
    data_collector_success = DataCollector(output_video_dir=DEFAULT_OUTPUT_VIDEO_DIR, output_data_dir=DEFAULT_OUTPUT_DATA_DIR, fps=FPS)
    data_collector_failed = DataCollector(output_video_dir=FAILED_OUTPUT_VIDEO_DIR, output_data_dir=FAILED_OUTPUT_DATA_DIR, fps=FPS)

    # State machine for cabinet-pour tasks
    current_state_index = START_STATE_INDEX
    last_commanded_poses = None
    # Instantiate generators
    kitchen_generator = KitchenTasksTrajectoryGenerator(obs)

    # Buffers for the current episode's data
    current_frames, current_obs_list, current_action_list = [], [], []
    successful_episodes_collected_count = 0
    current_attempt_number = 0 # Starts at 0, increments to 1 for the first attempt
    should_reset_env = True
    should_generate_and_play_trajectory = True
    
    # simulate environment
    while simulation_app.is_running() and successful_episodes_collected_count < MAX_EPISOIDES:
        with torch.inference_mode():
        
            if should_generate_and_play_trajectory:
                # Reset the environment if sub-task is failed or the process (three sub-tasks) is finished.
                if should_reset_env:
                    obs, _ = env.reset()
                    # Step environment with idle action to stabilize after reset
                    for _ in range(STABILIZATION_STEPS):
                        obs, _, _, _, _ = env.step(idle_actions_tensor)
                    
                    print(f"\n===== Start the attemp {current_attempt_number} =====")
                    current_attempt_number += 1
                    # 0. Clear external buffers for the new attempt
                    current_frames, current_obs_list, current_action_list = [], [], []
                    last_commanded_poses = None

                # 1. Generate the full trajectory by passing the current observation
                waypoints = []
                initial_poses = {
                    "right_pos": trajectory_player.initial_right_arm_pos_w,
                    "right_quat": trajectory_player.initial_right_arm_quat_wxyz_w,
                    "left_pos": trajectory_player.initial_left_arm_pos_w,
                    "left_quat": trajectory_player.initial_left_arm_quat_wxyz_w,
                }
                if "Cabinet-Pour-G1" in args_cli.task:
                    if current_state_index < len(CABINET_POUR_STATES):
                        current_state = CABINET_POUR_STATES[current_state_index]
                        print(f"\n--- Generating trajectory for state: {current_state} ---")
                        if current_state == "OPEN_DRAWER":
                            waypoints, last_commanded_poses = kitchen_generator.generate_open_drawer_trajectory(obs=obs, initial_poses=last_commanded_poses)
                        elif current_state == "PICK_AND_PLACE_MUG":
                            waypoints, last_commanded_poses = kitchen_generator.generate_pick_and_place_mug_trajectory(obs=obs, initial_poses=last_commanded_poses, home_poses=initial_poses)
                        elif current_state == "POUR_BOTTLE":
                            waypoints, last_commanded_poses = kitchen_generator.generate_pour_bottle_trajectory(obs=obs, initial_poses=last_commanded_poses, home_poses=initial_poses)
                elif "Stack-Cube-G1" in args_cli.task or "BlockStack-G1" in args_cli.task:
                    generator = StackCubesTrajectoryGenerator(obs=obs, initial_poses=initial_poses)
                    waypoints = generator.generate()
                elif "PickPlace-G1" in args_cli.task:
                    generator = GraspPickPlaceTrajectoryGenerator(obs=obs, initial_poses=initial_poses)
                    waypoints = generator.generate()
                
                # Load the generated waypoints into the player
                trajectory_player.set_waypoints(waypoints)
                    
                # 2. Prepare the interpolated trajectory for playback
                is_continuation = not should_reset_env
                trajectory_player.prepare_playback_trajectory(is_continuation=is_continuation)
                # 3. Set to False to play this trajectory
                should_generate_and_play_trajectory = False

            actions = idle_actions_tensor.clone()  # Initialize actions as idle actions if not set
            if trajectory_player.is_playing_back:
                playback_action_tuple = trajectory_player.get_formatted_action_for_playback(obs=obs)
                if playback_action_tuple is not None:
                    action_array_28D_np = playback_action_tuple[0]
                    actions = torch.tensor(action_array_28D_np, dtype=torch.float, device=args_cli.device).repeat(env.num_envs, 1) # type: ignore
                else: # Playback a sub-task finished
                    current_attempt_was_successful = False  # Initialize to False
                    # Check if the current sub-task was successful
                    if "Cabinet-Pour-G1" in args_cli.task:
                        if current_state == "OPEN_DRAWER":
                            subtask_terms_cfg = getattr(env.observation_manager.cfg, "subtask_terms")
                            drawer_opened_cfg = getattr(subtask_terms_cfg, "drawer_opened").params
                            current_attempt_was_successful = playground_obs.drawer_opened(env, **drawer_opened_cfg, debug=True).cpu().numpy()[0]
                        elif current_state == "PICK_AND_PLACE_MUG":
                            subtask_terms_cfg = getattr(env.observation_manager.cfg, "subtask_terms")
                            mug_placed_cfg = getattr(subtask_terms_cfg, "mug_placed").params
                            current_attempt_was_successful = playground_obs.object_placed(env, **mug_placed_cfg, debug=True).cpu().numpy()[0]
                        elif current_state == "POUR_BOTTLE":
                            current_attempt_was_successful = task_done(env, debug=True).cpu().numpy()[0] # type: ignore
                    else: # For other tasks
                        current_attempt_was_successful = task_done(env).cpu().numpy()[0]

                    # Handle state transitions and saving based on success/failure
                    if "Cabinet-Pour-G1" in args_cli.task:
                        is_final_subtask = (current_state == "POUR_BOTTLE")
                        
                        if current_attempt_was_successful:
                            print(f"--- Sub-task '{current_state}' SUCCEEDED. ---")
                            if is_final_subtask:
                                # Entire episode is successful
                                print(f"--- Episode SUCCEEDED. ---")
                                if args_cli.save_data:
                                    data_collector_success.save_episode(current_frames, current_obs_list, current_action_list)
                                successful_episodes_collected_count += 1
                                print(f"{successful_episodes_collected_count}/{current_attempt_number} ({successful_episodes_collected_count / current_attempt_number * 100:.2f}%): Attempt {current_attempt_number} result: Successful")
                                
                                # Reset for next episode
                                should_reset_env = True
                                current_state_index = START_STATE_INDEX
                            else:
                                # Move to the next sub-task without resetting
                                current_state_index += 1
                                should_reset_env = False
                        else: # current_attempt_was_successful is False
                            # Sub-task failed, so the whole episode failed
                            print(f"--- Sub-task '{current_state}' FAILED. ---")
                            if args_cli.save_data:
                                data_collector_failed.save_episode(current_frames, current_obs_list, current_action_list)
                            print(f"{successful_episodes_collected_count}/{current_attempt_number} ({successful_episodes_collected_count / current_attempt_number * 100:.2f}%): Attempt {current_attempt_number} result: Failed")

                            # Reset for next episode
                            should_reset_env = True
                            current_state_index = START_STATE_INDEX
                        
                        should_generate_and_play_trajectory = True
                        continue

                    else: # Logic for non-Cabinet-Pour tasks
                        if current_attempt_was_successful:
                            if args_cli.save_data:
                                data_collector_success.save_episode(current_frames, current_obs_list, current_action_list)
                            successful_episodes_collected_count += 1
                        else: # not current_attempt_was_successful
                            if args_cli.save_data:
                                data_collector_failed.save_episode(current_frames, current_obs_list, current_action_list)
                        
                        print(f"{successful_episodes_collected_count}/{current_attempt_number} ({successful_episodes_collected_count / current_attempt_number * 100:.2f}%): Attempt {current_attempt_number} result: {'Successful' if current_attempt_was_successful else 'Failed'}")
                        should_generate_and_play_trajectory = True
                        should_reset_env = True
            # apply actions
            obs, _, _, _, _ = env.step(actions)
            
            # Data extraction for saving
            robot_joint_state = obs["policy"]["robot_joint_pos"].cpu().numpy().flatten().astype(np.float64) # type: ignore
            processed_action = obs["policy"]["processed_actions"].cpu().numpy().flatten().astype(np.float64) # type: ignore
            rgb_image_np = obs["policy"]["rgb_image"].squeeze(0).cpu().numpy()  # shape: (1, 480, 640, 3) -> (480, 640, 3);from cuda to cpu # type: ignore
            rgb_image_bgr = cv2.cvtColor(rgb_image_np, cv2.COLOR_RGB2BGR) # RGB to CV2 BGR format

            data_state = robot_joint_state[JOINT_STATE_ID]
            data_action = np.zeros(28) if G1_HAND_TYPE=="trihand" else np.zeros(26)
            # Swap value to match joint action orders
            for tar_i, src_i in enumerate(TARGET_IDX):
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
