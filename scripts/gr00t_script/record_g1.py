# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Data collection for Isaac Lab environments.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--dataset_file", type=str, default="./datasets/g1_pour_dataset.hdf5", help="File path to export recorded demos.")
parser.add_argument("--num_demos", type=int, default=10, help="Number of demonstrations to record.")
parser.add_argument(
    "--num_success_steps",  # Kept for compatibility with record_demos structure, but success is per-episode
    type=int,
    default=1, # For G1, success is determined at the end of an episode/trajectory.
    help="Number of continuous steps with task success for concluding a demo. (Used differently here, success is at trajectory end)",
)
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Cabinet-Pour-G1-Abs-v0",
    choices=["Isaac-Cabinet-Pour-G1-Abs-v0", "Isaac-Stack-Cube-G1-Abs-v0", "Isaac-PickPlace-G1-Abs-v0"],
    help="Name of the task. Options: 'Isaac-Cabinet-Pour-G1-Abs-v0', 'Isaac-Stack-Cube-G1-Abs-v0', 'Isaac-PickPlace-G1-Abs-v0'."
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

import os
import contextlib
from typing import cast
import gymnasium as gym
import torch
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab_tasks.manager_based.manipulation.playground_g1.mdp import observations as playground_obs
from isaaclab_tasks.manager_based.manipulation.playground_g1.cabinet_pour_g1_env_cfg import CabinetPourG1EnvCfg

# Omniverse logger / UI
import omni.ui as ui
from isaaclab.envs.ui import EmptyWindow
from isaaclab_mimic.ui.instruction_display import InstructionDisplay, show_subtask_instructions

from isaaclab.managers import DatasetExportMode
from isaaclab.envs import ManagerBasedRLEnv
# Conditionally import task_done based on the task
if "Stack-Cube-G1" in args_cli.task:
    from isaaclab_tasks.manager_based.manipulation.stack_g1.mdp.terminations import task_done
elif "PickPlace-G1" in args_cli.task:
    from isaaclab_tasks.manager_based.manipulation.pick_place_g1.mdp.terminations import task_done
elif "Cabinet-Pour-G1" in args_cli.task:
    from isaaclab_tasks.manager_based.manipulation.playground_g1.mdp.terminations import task_done


""" Customized modules """
from utils.trajectory_player import TrajectoryPlayer
from utils.trajectory_generators import (
    GraspPickPlaceTrajectoryGenerator,
    StackCubesTrajectoryGenerator,
    KitchenTasksTrajectoryGenerator,
)

""" Constants """
STEPS_PER_MOVEMENT_SEGMENT = 75  # long-distance/orientation movement
STEPS_PER_SHORTSHIFT_SEGMENT = 30  # Short-distance/orientation movement
STEPS_PER_GRASP_SEGMENT = 15  # Hand grasp
STABILIZATION_STEPS = 30 # Step 30 times for stabilization after env.reset()
MAX_EPISODES = 20  # Fallback if --num_demos is 0, will be overridden by args_cli.num_demos
CABINET_POUR_STATES = ["OPEN_DRAWER", "PICK_AND_PLACE_MUG", "POUR_BOTTLE"]

def main():
    num_demos_to_collect = args_cli.num_demos if args_cli.num_demos > 0 else MAX_EPISODES
    # get directory path and file name (without extension) from cli arguments
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # create environment configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=not args_cli.disable_fabric)
    env_cfg = cast(CabinetPourG1EnvCfg, env_cfg)    
    env_cfg.terminations = None  # type: ignore # Allow episodes to run full trajectory length
    
    # Configure HDF5 recorder
    env_cfg.recorders = ActionStateRecorderManagerCfg(
        dataset_export_dir_path=output_dir,
        dataset_filename=output_file_name,
        dataset_export_mode=DatasetExportMode.EXPORT_SUCCEEDED_ONLY, # dataset_export_mode=DatasetExportMode.EXPORT_ALL, # or dataset_export_mode=DatasetExportMode.EXPORT_SUCCEEDED_ONLY,
    )
    env_cfg.observations.policy.concatenate_terms = False

    # create environment
    env = cast(ManagerBasedRLEnv, gym.make(args_cli.task, cfg=env_cfg).unwrapped)

    # Flags and counters
    should_reset_env = True
    should_generate_and_play_trajectory = True
    current_recorded_demo_count = 0
    current_attempt_number = 0
    last_commanded_poses = None

    # State machine for the Isaac-Cabinet-Pour-G1-Abs-v0 task
    current_state_index = 0
    
    # UI Setup (A new tab will be created in the IsaacSim UI)
    instruction_display = InstructionDisplay("Trajectory Player") # Device name is a placeholder
    demo_label_ui = None
    subtask_label_ui = None
    window = EmptyWindow(env, "G1 Auto-Record Status")
    with window.ui_window_elements["main_vstack"]:
        demo_label_ui = ui.Label(f"Recorded {current_recorded_demo_count} successful demonstrations.")
        subtask_label_ui = ui.Label("Waiting for trajectory...")
        instruction_display.set_labels(subtask_label_ui, demo_label_ui)

    # reset environment
    obs, _ = env.reset()
    # Pass initial observation to TrajectoryPlayer to set default poses
    trajectory_player = TrajectoryPlayer(env, initial_obs=obs, 
                                         steps_per_movement_segment=STEPS_PER_MOVEMENT_SEGMENT, 
                                         steps_per_grasp_segment=STEPS_PER_GRASP_SEGMENT, 
                                         steps_per_shortshift_segment=STEPS_PER_SHORTSHIFT_SEGMENT)
    # Instantiate generators
    kitchen_generator = KitchenTasksTrajectoryGenerator(obs)
    # Get the idle action based on the initial reset pose
    idle_action_np = trajectory_player.get_idle_action_np()
    idle_actions_tensor = torch.tensor(idle_action_np, dtype=torch.float, device=args_cli.device).repeat(env.num_envs, 1)

    # simulate environment
    with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
        while simulation_app.is_running() and current_recorded_demo_count < num_demos_to_collect:
            if should_generate_and_play_trajectory:
                if should_reset_env:
                    env.recorder_manager.reset() # Clear internal buffers for all envs in manager
                    obs, _ = env.reset() # Get new observations for the next trajectory
                    # Step environment with idle action to stabilize after reset
                    for _ in range(STABILIZATION_STEPS):
                        obs, _, _, _, _ = env.step(idle_actions_tensor)
                    
                    print(f"\n===== Start attempt {current_attempt_number + 1} (Collected: {current_recorded_demo_count}/{num_demos_to_collect}) =====")
                    current_attempt_number += 1
                    last_commanded_poses = None
                    current_state_index = 0 # Reset state machine
                    if demo_label_ui: # Update UI for new attempt
                        instruction_display.show_demo(f"Recorded {current_recorded_demo_count} successful demonstrations.")

                # 1. Generate the trajectory based on the task and state
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
                            waypoints, last_commanded_poses = kitchen_generator.generate_open_drawer_sub_trajectory(obs=obs, initial_poses=last_commanded_poses)
                        elif current_state == "PICK_AND_PLACE_MUG":
                            waypoints, last_commanded_poses = kitchen_generator.generate_pick_and_place_mug_sub_trajectory(obs=obs, initial_poses=last_commanded_poses)
                        elif current_state == "POUR_BOTTLE":
                            waypoints, last_commanded_poses = kitchen_generator.generate_pour_bottle_sub_trajectory(obs=obs, initial_poses=last_commanded_poses, home_poses=initial_poses)
                else: # Other tasks
                    if "Stack-Cube-G1" in args_cli.task:
                        generator = StackCubesTrajectoryGenerator(obs=obs, initial_poses=initial_poses)
                        waypoints = generator.generate()
                    elif "PickPlace-G1" in args_cli.task:
                        generator = GraspPickPlaceTrajectoryGenerator(obs=obs, initial_poses=initial_poses)
                        waypoints = generator.generate()

                # Load the generated waypoints into the player
                trajectory_player.set_waypoints(waypoints)
                
                # 2. Prepare the playback trajectory
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
                else: # Playback for the current sub-task finished
                    current_attempt_was_successful = False
                    # Check if the current sub-task was successful
                    if "Cabinet-Pour-G1" in args_cli.task:
                        current_state = CABINET_POUR_STATES[current_state_index]
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
                        current_state = CABINET_POUR_STATES[current_state_index]
                        is_final_subtask = (current_state == "POUR_BOTTLE")
                        
                        if current_attempt_was_successful:
                            print(f"--- Sub-task '{current_state}' SUCCEEDED. ---")
                            if is_final_subtask:
                                print(f"--- Episode SUCCEEDED. ---")
                                all_env_ids = torch.arange(env.num_envs, device=env.device).tolist()
                                successful_mask_cpu = torch.tensor([True])
                                env.recorder_manager.record_pre_reset(all_env_ids, force_export_or_skip=False)
                                env.recorder_manager.set_success_to_episodes(all_env_ids, successful_mask_cpu.unsqueeze(1).to(device=env.device))
                                env.recorder_manager.export_episodes(all_env_ids)
                                print(f"Env Attempt {current_attempt_number} result: Successful")
                                should_reset_env = True
                                current_state_index = 0
                            else:
                                # Move to the next sub-task without resetting
                                current_state_index += 1
                                should_reset_env = False
                        else: # current_attempt_was_successful is False
                            print(f"--- Sub-task '{current_state}' FAILED. ---")
                            all_env_ids = torch.arange(env.num_envs, device=env.device).tolist()
                            successful_mask_cpu = torch.tensor([False])
                            env.recorder_manager.record_pre_reset(all_env_ids, force_export_or_skip=True) # Force skip failed
                            env.recorder_manager.set_success_to_episodes(all_env_ids, successful_mask_cpu.unsqueeze(1).to(device=env.device))
                            # env.recorder_manager.export_episodes(all_env_ids) # Don't export failed
                            print(f"Env Attempt {current_attempt_number} result: Failed")
                            should_reset_env = True
                            current_state_index = 0
                        
                        should_generate_and_play_trajectory = True
                        continue

                    else: # Logic for non-Cabinet-Pour tasks
                        successful_mask_cpu = torch.tensor([current_attempt_was_successful])
                        all_env_ids = torch.arange(env.num_envs, device=env.device).tolist()
                        env.recorder_manager.record_pre_reset(all_env_ids, force_export_or_skip=False)
                        env.recorder_manager.set_success_to_episodes(all_env_ids, successful_mask_cpu.unsqueeze(1).to(device=env.device))
                        env.recorder_manager.export_episodes(all_env_ids)
                        print(f"Env Attempt {current_attempt_number} result: {'Successful' if current_attempt_was_successful else 'Failed'}")
                        should_generate_and_play_trajectory = True
                        should_reset_env = True

            # apply actions
            obs, _, _, _, _ = env.step(actions)
            
            if subtask_label_ui is not None:
                subtasks_obs = obs.get("subtask_terms", {})
                if isinstance(subtasks_obs, dict) and subtasks_obs:
                    show_subtask_instructions(instruction_display, subtasks_obs, obs, env.cfg)

            # Update demo count display
            if env.recorder_manager.exported_successful_episode_count > current_recorded_demo_count:
                current_recorded_demo_count = env.recorder_manager.exported_successful_episode_count
                label_text = f"Recorded {current_recorded_demo_count} successful demonstrations."
                print(label_text)
                if demo_label_ui:
                    demo_label_ui.text = label_text
            
            if args_cli.num_demos > 0 and env.recorder_manager.exported_successful_episode_count >= args_cli.num_demos:
                print(f"All {args_cli.num_demos} demonstrations recorded. Exiting the app.")
                break

            if env.sim.is_stopped():
                break

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
