# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run data collection for OpenArm-DexHand tasks."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Data collection for OpenArm-DexHand environments.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--save_data", action="store_true", default=True, help="Save video and compose data to parquet.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Can-Sorting-OpenArm-DexHand-v0",
    choices=["Isaac-Can-Sorting-OpenArm-DexHand-v0", "Isaac-Cube-Stack-OpenArm-DexHand-v0", "Isaac-Cabinet-Pour-OpenArm-DexHand-v0"],
    help="Name of the task."
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
# Import pinocchio before AppLauncher to force the use of the version installed by IsaacLab
import pinocchio  # noqa: F401
##########

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb
carb_settings_iface = carb.settings.get_settings()

if "G1" in args_cli.task:
    G1_HAND_TYPE = "inspire"   # ["trihand", "inspire"]
    carb_settings_iface.set_string("/unitree_g1_env/hand_type", G1_HAND_TYPE)
    ROBOT_TYPE = "g1_"+ G1_HAND_TYPE
elif "OpenArm" in args_cli.task:
    ROBOT_TYPE = "openarm_leaphand"
else:
    raise NotImplementedError("Currently only for G1 or OpenArm.")
carb_settings_iface.set_string("/data_collect/robot_type", ROBOT_TYPE)

"""Rest everything follows."""
from typing import cast
import gymnasium as gym
import torch
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.manager_based.manipulation.playground_openarm.dexhand_bimanual.task_scenes.cabinet_pour.mdp import observations as playground_obs
# Suppress specific warnings from qpsolvers
import warnings
from qpsolvers.warnings import SparseConversionWarning
warnings.filterwarnings("ignore", category=SparseConversionWarning, module="qpsolvers.conversions.ensure_sparse_matrices")

# PLACEHOLDER: Extension template (do not remove this comment)
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

# Conditionally import task_done based on the task
if "Cube-Stack-OpenArm" in args_cli.task:
    from isaaclab_tasks.manager_based.manipulation.playground_openarm.dexhand_bimanual.task_scenes.cube_stack.mdp.terminations import task_done
elif "Can-Sorting-OpenArm" in args_cli.task:
    from isaaclab_tasks.manager_based.manipulation.playground_openarm.dexhand_bimanual.task_scenes.can_sorting.mdp.terminations import task_done
elif "Cabinet-Pour-OpenArm" in args_cli.task:
    from isaaclab_tasks.manager_based.manipulation.playground_openarm.dexhand_bimanual.task_scenes.cabinet_pour.mdp.terminations import task_done

""" Constants """
STEPS_PER_MOVEMENT_SEGMENT = 75  # 4 segments for movement
STEPS_PER_SHORTSHIFT_SEGMENT = 30  # Short-distance movement
STEPS_PER_GRASP_SEGMENT = 15  # Hand grasp
STABILIZATION_STEPS = 30  # Step 30 times for stabilization after env.reset()
FPS = 30  # sim.dt * decimation = 1/120 * 4 = 1/30
MAX_EPISODES = 2000  # Limit to 2000 iterations for data collection
CABINET_POUR_STATES = ["OPEN_DRAWER", "PICK_AND_PLACE_MUG", "POUR_BOTTLE"]  # For 'Isaac-Cabinet-Pour-OpenArm-DexHand-v0'
START_STATE_INDEX = 0  # Flexible for starting from an index of the CABINET_POUR_STATES

# parquet data setup
DATASET_PATH = "datasets/gr00t_collection/OpenArm_dataset/"
DEFAULT_OUTPUT_VIDEO_DIR = f"{DATASET_PATH}videos/chunk-000/observation.images.camera"
DEFAULT_OUTPUT_DATA_DIR = f"{DATASET_PATH}data/chunk-000"

FAILED_DATASET_PATH = "datasets/gr00t_collection/OpenArm_dataset_failed/"
FAILED_OUTPUT_VIDEO_DIR = f"{FAILED_DATASET_PATH}videos/chunk-000/observation.images.camera"
FAILED_OUTPUT_DATA_DIR = f"{FAILED_DATASET_PATH}data/chunk-000"

# OpenArm joint indices (based on openarm_robot_cfg.py)
# Arm joints: 14 (7 left + 7 right)
LEFT_ARM_INDICES = [0, 1, 2, 3, 4, 5, 6]
RIGHT_ARM_INDICES = [7, 8, 9, 10, 11, 12, 13]
# Hand joints: 16 per hand (DexHand)
# NOTE: The current robot config only exposes the RIGHT hand (indices 14-29).
# The left hand is missing from the observation, so we leave it empty to avoid IndexError.
LEFT_HAND_INDICES = [] # list(range(14, 30))
RIGHT_HAND_INDICES = list(range(14, 30)) # list(range(30, 46))

# Action observation index mapping for OpenArm (28D action space)
ACTION_INDICE = {
    "left_arm": [0, 1, 2, 3, 4, 5, 6],
    "right_arm": [7, 8, 9, 10, 11, 12, 13],
    "left_hand": [14, 15, 16, 17, 18, 19, 20],  # Assume 7 joints mapped from 16
    "right_hand": [21, 22, 23, 24, 25, 26, 27],
}

JOINT_STATE_ID = LEFT_ARM_INDICES + RIGHT_ARM_INDICES + LEFT_HAND_INDICES + RIGHT_HAND_INDICES

TARGET_IDX = [i for sublist in ACTION_INDICE.values() for i in sublist]  # Should be 0-27


def quaternion_multiply(q_world_target, q_offset):
    """
    This applies the local rotation `q_offset` on a target quaternion `q_world_target` expressed in world frame.
    q_result = q_world_target ⊗ q_offset (active rotation)
    """
    w1,x1,y1,z1 = q_world_target[0]
    w2,x2,y2,z2 = q_offset
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return torch.tensor([[w,x,y,z]], device=q_world_target.device)

def adjust_openarm_quat(actions: torch.Tensor):
    """
    Apply quaternion adjustments for OpenArm to align with world frame. 
    ! Currently OpenArm-LeapHand eef needs to rotate [left(Y:-90),right(X:180,Y:90)] to align with the world frame. (Short-term fix)
    """
    if "OpenArm" in args_cli.task:
        # eef frame quaternion in world frame
        LEFT_Q_IN_WORLD = [0.707, 0, -0.707, 0]
        RIGHT_Q_IN_WORLD = [0, 0.707, 0, 0.707]
        left_hand_quat = quaternion_multiply(actions[:,3:7], LEFT_Q_IN_WORLD)
        right_hand_quat = quaternion_multiply(actions[:,10:14], RIGHT_Q_IN_WORLD)  
        actions[:,3:7] = left_hand_quat
        actions[:,10:14] = right_hand_quat
        
    return actions


def main():
    # Record start time for summary
    start_time = time.time()

    # create environment configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)

    # Disable the termination term for data collection.
    env_cfg.terminations = None

    # create environment
    env = cast(ManagerBasedRLEnv, gym.make(args_cli.task, cfg=env_cfg).unwrapped)

    # reset environment
    obs, _ = env.reset()
    # Pass initial observation to TrajectoryPlayer to set default poses
    trajectory_player = TrajectoryPlayer(env, initial_obs=obs, steps_per_movement_segment=STEPS_PER_MOVEMENT_SEGMENT, steps_per_grasp_segment=STEPS_PER_GRASP_SEGMENT, steps_per_shortshift_segment=STEPS_PER_SHORTSHIFT_SEGMENT)
    # Get the idle action based on the initial reset pose
    idle_action_np = trajectory_player.get_idle_action_np()
    idle_actions_tensor = adjust_openarm_quat(torch.tensor(idle_action_np, dtype=torch.float, device=args_cli.device).repeat(env.unwrapped.num_envs, 1))

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
    current_attempt_number = 0  # Starts at 0, increments to 1 for the first attempt
    should_reset_env = True
    should_generate_and_play_trajectory = True

    # simulate environment
    while simulation_app.is_running() and successful_episodes_collected_count < MAX_EPISODES:
        with torch.inference_mode():

            if should_generate_and_play_trajectory:
                # Reset the environment if sub-task is failed or the process (three sub-tasks) is finished.
                if should_reset_env:
                    obs, _ = env.reset()
                    # Step environment with idle action to stabilize after reset
                    for _ in range(STABILIZATION_STEPS):
                        obs, _, _, _, _ = env.step(idle_actions_tensor)

                    print(f"\n===== Start the attempt {current_attempt_number} =====")
                    current_attempt_number += 1
                    # 0. Clear external buffers for the new attempt
                    current_frames, current_obs_list, current_action_list = [], [], []
                    last_commanded_poses = None

                # 1. Generate the full trajectory by passing the current observation
                waypoints = []
                initial_poses = {
                    "right_eef_pos": trajectory_player.initial_right_arm_pos_w,
                    "right_eef_quat": trajectory_player.initial_right_arm_quat_wxyz_w,
                    "left_eef_pos": trajectory_player.initial_left_arm_pos_w,
                    "left_eef_quat": trajectory_player.initial_left_arm_quat_wxyz_w,
                    "right_hand_closed": False,
                    "left_hand_closed": False,
                }
                if "Cabinet-Pour-OpenArm" in args_cli.task:
                    if current_state_index < len(CABINET_POUR_STATES):
                        current_state = CABINET_POUR_STATES[current_state_index]
                        print(f"\n--- Generating trajectory for state: {current_state} ---")
                        if current_state == "OPEN_DRAWER":
                            waypoints, last_commanded_poses = kitchen_generator.generate_open_drawer_trajectory(obs=obs, initial_poses=last_commanded_poses)
                        elif current_state == "PICK_AND_PLACE_MUG":
                            waypoints, last_commanded_poses = kitchen_generator.generate_pick_and_place_mug_trajectory(obs=obs, initial_poses=last_commanded_poses, home_poses=initial_poses)
                        elif current_state == "POUR_BOTTLE":
                            waypoints, last_commanded_poses = kitchen_generator.generate_pour_bottle_trajectory(obs=obs, initial_poses=last_commanded_poses, home_poses=initial_poses)
                elif "Cube-Stack-OpenArm" in args_cli.task:
                    generator = StackCubesTrajectoryGenerator(obs=obs, initial_poses=initial_poses)
                    waypoints = generator.generate()
                elif "Can-Sorting-OpenArm" in args_cli.task:
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
                    actions = torch.tensor(action_array_28D_np, dtype=torch.float, device=args_cli.device).repeat(env.num_envs, 1)
                else:  # Playback a sub-task finished
                    current_attempt_was_successful = False  # Initialize to False
                    # Check if the current sub-task was successful
                    if "Cabinet-Pour-OpenArm" in args_cli.task:
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
                    else:  # For other tasks
                        current_attempt_was_successful = task_done(env).cpu().numpy()[0]

                    # Handle state transitions and saving based on success/failure
                    if "Cabinet-Pour-OpenArm" in args_cli.task:
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
                        else:  # current_attempt_was_successful is False
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

                    else:  # Logic for non-Cabinet-Pour tasks
                        if current_attempt_was_successful:
                            if args_cli.save_data:
                                data_collector_success.save_episode(current_frames, current_obs_list, current_action_list)
                            successful_episodes_collected_count += 1
                        else:  # not current_attempt_was_successful
                            if args_cli.save_data:
                                data_collector_failed.save_episode(current_frames, current_obs_list, current_action_list)

                        print(f"{successful_episodes_collected_count}/{current_attempt_number} ({successful_episodes_collected_count / current_attempt_number * 100:.2f}%): Attempt {current_attempt_number} result: {'Successful' if current_attempt_was_successful else 'Failed'}")
                        should_generate_and_play_trajectory = True
                        should_reset_env = True
                        

            # apply actions
            actions = adjust_openarm_quat(actions)
            obs, _, _, _, _ = env.step(actions)

            # Data extraction for saving
            robot_joint_state = obs["robot_obs"]["robot_joint_pos"].cpu().numpy().flatten().astype(np.float64)
            processed_action = obs["robot_obs"]["processed_actions"].cpu().numpy().flatten().astype(np.float64)
            rgb_image_np = obs["robot_obs"]["rgb_image"].squeeze(0).cpu().numpy()  # shape: (1, 480, 640, 3) -> (480, 640, 3); from cuda to cpu
            rgb_image_bgr = cv2.cvtColor(rgb_image_np, cv2.COLOR_RGB2BGR)  # RGB to CV2 BGR format

            data_state = robot_joint_state[JOINT_STATE_ID]
            data_action = np.zeros(28)
            # Map processed_action to data_action
            if "OpenArm" in args_cli.task:
                data_action = processed_action
            else:                    
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
    print(f"Maximum episodes to collect: {MAX_EPISODES}")
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
