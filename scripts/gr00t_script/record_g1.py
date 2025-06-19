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
parser.add_argument("--dataset_file", type=str, default="./datasets/g1_dataset.hdf5", help="File path to export recorded demos.")
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
    default="Isaac-Stack-Cube-G1-Abs-v0",
    choices=["Isaac-Stack-Cube-G1-Abs-v0", "Isaac-BlockStack-G1-Abs-v0", "Isaac-PickPlace-G1-Abs-v0"],
    help="Name of the task. Options: 'Isaac-Stack-Cube-G1-Abs-v0', 'Isaac-BlockStack-G1-Abs-v0', 'Isaac-PickPlace-G1-Abs-v0'."
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
import gymnasium as gym
import torch
##########
import isaaclab_tasks.manager_based.manipulation.pick_place_g1  # noqa: F401
##########
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg

# Omniverse logger / UI
import omni.ui as ui
from isaaclab.envs.ui import EmptyWindow
from isaaclab_mimic.ui.instruction_display import InstructionDisplay, show_subtask_instructions
# Conditionally import task_done based on the task
if "Stack-Cube-G1" in args_cli.task or "BlockStack-G1" in args_cli.task: # Assuming BlockStack also uses a similar done condition
    from isaaclab_tasks.manager_based.manipulation.stack_g1.mdp.terminations import task_done
elif "PickPlace-G1" in args_cli.task:
    from isaaclab_tasks.manager_based.manipulation.pick_place_g1.mdp.terminations import task_done
    from isaaclab.managers import DatasetExportMode # Only needed for pick_place if it uses it differently

""" Customized modules """
from utils.trajectory_player import TrajectoryPlayer

""" Constants """
# EPISODE_FRAMES_LEN = STEPS_PER_MOVEMENT_SEGMENT * 4 + STEPS_PER_GRASP_SEGMENT * 2 # frames (steps)
STEPS_PER_MOVEMENT_SEGMENT = 100
STEPS_PER_GRASP_SEGMENT = 50
STABILIZATION_STEPS = 30 # Step 30 times for stabilization after env.reset()
MAX_EPISODES = 1000  # Fallback if --num_demos is 0, will be overridden by args_cli.num_demos

def main():
    num_demos_to_collect = args_cli.num_demos if args_cli.num_demos > 0 else MAX_EPISODES
    # get directory path and file name (without extension) from cli arguments
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # create environment configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=not args_cli.disable_fabric)
    env_cfg.terminations = None # Allow episodes to run full trajectory length
    
    # Configure HDF5 recorder
    env_cfg.recorders = ActionStateRecorderManagerCfg(
        dataset_export_dir_path=output_dir,
        dataset_filename=output_file_name,
        # dataset_export_mode=DatasetExportMode.EXPORT_ALL, # or dataset_export_mode=DatasetExportMode.EXPORT_SUCCEEDED_ONLY,
    )
    env_cfg.observations.policy.concatenate_terms = False

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    # print(f"[INFO]: Gym observation space: {env.observation_space}")
    # print(f"[INFO]: Gym action space: {env.action_space}")

    # Flags and counters
    should_reset_recording_instance = False
    running_recording_instance = False # Start recording when trajectory plays
    current_recorded_demo_count = 0
    current_attempt_number = 0 # Starts at 0, increments to 1 for the first attempt
    should_generate_and_play_trajectory = True

    # UI Setup (A new tab will be created in the IsaacSim UI)
    instruction_display = InstructionDisplay("Trajectory Player") # Device name is a placeholder
    demo_label_ui = None
    subtask_label_ui = None
    window = EmptyWindow(env.unwrapped, "G1 Auto-Record Status") # Use unwrapped env for UI
    with window.ui_window_elements["main_vstack"]:
        demo_label_ui = ui.Label(f"Recorded {current_recorded_demo_count} successful demonstrations.")
        subtask_label_ui = ui.Label("Waiting for trajectory...")
        instruction_display.set_labels(subtask_label_ui, demo_label_ui)
    subtasks = {}

    # reset environment
    env.unwrapped.sim.reset() # Reset simulation first
    obs, _ = env.reset()
    # Pass initial observation to TrajectoryPlayer to set default poses
    trajectory_player = TrajectoryPlayer(env.unwrapped, initial_obs=obs, steps_per_movement_segment=STEPS_PER_MOVEMENT_SEGMENT, steps_per_grasp_segment=STEPS_PER_GRASP_SEGMENT)
    # Get the idle action based on the initial reset pose
    idle_action_np = trajectory_player.get_idle_action_np()
    idle_actions_tensor = torch.tensor(idle_action_np, dtype=torch.float, device=args_cli.device).repeat(env.unwrapped.num_envs, 1)

    # simulate environment
    with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
        while simulation_app.is_running() and current_recorded_demo_count < num_demos_to_collect:
            if should_generate_and_play_trajectory:
                # obs, _ = env.reset() # Reset is handled in should_reset_recording_instance block
                # Step environment with idle action to stabilize after reset
                for _ in range(STABILIZATION_STEPS):
                    obs, _, _, _, _ = env.step(idle_actions_tensor)
                
                print(f"\n===== Start attempt {current_attempt_number + 1} (Collected: {current_recorded_demo_count}/{num_demos_to_collect}) =====")
                current_attempt_number += 1
                # 1. Generate the full trajectory by passing the current observation
                trajectory_player.generate_auto_stack_cubes_trajectory(obs=obs)
                # 2. Prepare the playback trajectory
                trajectory_player.prepare_playback_trajectory()
                # 3. Set to False to play this trajectory
                should_generate_and_play_trajectory = False
                running_recording_instance = True # Start recording for this trajectory

            actions = idle_actions_tensor.clone()  # Initialize actions as idle actions if not set
            
            if running_recording_instance and trajectory_player.is_playing_back:
                playback_action_tuple = trajectory_player.get_formatted_action_for_playback()
                if playback_action_tuple is not None:
                    action_array_28D_np = playback_action_tuple[0]
                    actions = torch.tensor(action_array_28D_np, dtype=torch.float, device=args_cli.device).repeat(env.unwrapped.num_envs, 1)
                    obs, _, _, _, _ = env.step(actions) # obs is a dict
                    if subtask_label_ui is not None:
                        if not subtasks: # Check if subtasks is an empty dict for the single environment
                            subtasks = obs.get("subtask_terms", {}) # Get from the single env obs
                        if subtasks: # Check if subtasks is not None and not empty
                            show_subtask_instructions(instruction_display, subtasks, obs, env.unwrapped.cfg)
                else: # Playback finished
                    running_recording_instance = False # Stop recording
                    # Determine success for all environments
                    # task_done returns a tensor of shape (num_envs,)
                    successful_mask_cpu = task_done(env.unwrapped).cpu()
                    
                    # env_ids for the recorder manager should be the actual integer IDs
                    all_env_ids = torch.arange(env.unwrapped.num_envs, device=env.unwrapped.device)
                    env.unwrapped.recorder_manager.record_pre_reset(all_env_ids, force_export_or_skip=False)
                    
                    env.unwrapped.recorder_manager.set_success_to_episodes(
                        all_env_ids, successful_mask_cpu.unsqueeze(1).to(device=env.unwrapped.device)
                    )
                    env.unwrapped.recorder_manager.export_episodes(all_env_ids)
                    
                    should_reset_recording_instance = True
                    # Print per-environment results
                    print(f"Env Attempt {current_attempt_number} result: {'Successful' if successful_mask_cpu[0] else 'Failed'}")
            elif not running_recording_instance: # E.g. between trajectories or if not started
                env.unwrapped.sim.render() # Keep rendering

            # Update demo count display
            if env.unwrapped.recorder_manager.exported_successful_episode_count > current_recorded_demo_count:
                current_recorded_demo_count = env.unwrapped.recorder_manager.exported_successful_episode_count
                label_text = f"Recorded {current_recorded_demo_count} successful demonstrations."
                print(label_text)
                if demo_label_ui: # if demo_label_ui and env.num_envs == 1:
                    demo_label_ui.text = label_text

            if should_reset_recording_instance:
                env.unwrapped.sim.reset()
                env.unwrapped.recorder_manager.reset() # Clear internal buffers for all envs in manager
                obs, _ = env.reset() # Get new observations for the next trajectory
                should_reset_recording_instance = False
                should_generate_and_play_trajectory = True # Trigger new trajectory generation
                subtasks = {} # Reset subtasks for the new episode
                if demo_label_ui: # Update UI for new attempt
                    instruction_display.show_demo(f"Recorded {current_recorded_demo_count} successful demonstrations.")
            
            if args_cli.num_demos > 0 and env.unwrapped.recorder_manager.exported_successful_episode_count >= args_cli.num_demos:
                print(f"All {args_cli.num_demos} demonstrations recorded. Exiting the app.")
                break

            if env.unwrapped.sim.is_stopped():
                break

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
