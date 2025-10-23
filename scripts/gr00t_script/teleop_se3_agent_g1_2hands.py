# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run a keyboard teleoperation with Isaac Lab manipulation environments."""

"""
Script to run keyboard teleoperation for Isaac Lab manipulation environments.
Launch Isaac Sim Simulator first.
"""

# =========================
# Imports and CLI Arguments
# =========================
import argparse
from typing import cast

from isaaclab.app import AppLauncher

# Argument parsing
parser = argparse.ArgumentParser(description="Keyboard teleoperation for Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--teleop_device", type=str, default="keyboard", help="Device for interacting with environment")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Cabinet-Pour-G1-Abs-v0",
    choices=["Isaac-Cabinet-Pour-G1-Abs-v0", "Isaac-Stack-Cube-G1-Abs-v0", "Isaac-BlockStack-G1-Abs-v0", "Isaac-PickPlace-G1-Abs-v0"],
    help="Name of the task. Options: 'Isaac-Stack-Cube-G1-Abs-v0', 'Isaac-BlockStack-G1-Abs-v0', 'Isaac-PickPlace-G1-Abs-v0'."
)
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")

# Add AppLauncher CLI args and parse
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True  # Always enable cameras for this script

# =========================
# Simulator Launch
# =========================
# Import pinocchio before AppLauncher to force the use of the version installed by IsaacLab
import pinocchio  # noqa: F401

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb
carb_settings_iface = carb.settings.get_settings()

G1_HAND_TYPE = "inspire"   # ["trihand", "inspire"]
carb_settings_iface.set_string("/unitree_g1_env/hand_type", G1_HAND_TYPE)

# =========================
# Main Teleoperation Logic
# =========================
import gymnasium as gym
import numpy as np
import torch
from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg
from isaaclab.envs import ManagerBasedRLEnv
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from utils.trajectory_player import TrajectoryPlayer
from utils.trajectory_generators import FileBasedTrajectoryGenerator
from utils.quaternion_utils import quat_xyzw_to_wxyz
from scipy.spatial.transform import Rotation as R
import warnings
from qpsolvers.warnings import SparseConversionWarning
# Suppress specific warnings from qpsolvers
warnings.filterwarnings("ignore", category=SparseConversionWarning, module="qpsolvers.conversions.ensure_sparse_matrices")

def pre_process_actions(
    # LIVE TELEOP: (delta_pose_6D_numpy, gripper_command_bool)
    # PLAYBACK: tuple[np.ndarray] where np.ndarray is 14D [pos(3), quat_xyzw(4), hand_joints(7)]
    teleop_data: tuple[np.ndarray, bool] | list[tuple[np.ndarray, np.ndarray, np.ndarray]] | tuple[np.ndarray, ...],
    active_hand: str,
    num_envs: int,
    device: str,
    previous_target_left_eef_pos_w: np.ndarray,
    previous_target_left_eef_quat_wxyz_w: np.ndarray,
    previous_left_gripper_bool: bool,
    previous_target_right_eef_pos_w: np.ndarray,
    previous_target_right_eef_quat_wxyz_w: np.ndarray,
    previous_right_gripper_bool: bool,
    last_processed_keyboard_gripper_toggle_state: bool, # State of keyboard gripper from *before* current teleop_data
    trajectory_player: TrajectoryPlayer
) -> tuple[torch.Tensor, np.ndarray, np.ndarray, bool, np.ndarray, np.ndarray, bool, bool]:
    """Convert teleop data to the format expected by the environment action space."""
    # teleop_data can be one of two things:
    # 1. From TrajectoryPlayer (playback): a single 28D numpy array  [left_arm_eef(7), right_arm_eef(7), hand_joints(14)]) in wxyz
    if trajectory_player.is_playing_back and isinstance(teleop_data, tuple) and len(teleop_data) == 1 and isinstance(teleop_data[0], np.ndarray):
        action_array_28D_np = teleop_data[0]
        target_left_eef_pos_w = action_array_28D_np[0:3]
        target_left_eef_quat_wxyz_w = action_array_28D_np[3:7]
        target_right_eef_pos_w = action_array_28D_np[7:10]  # Right arm position starts at index 7
        target_right_eef_quat_wxyz_w = action_array_28D_np[10:14]  # Right arm quaternion starts at index 10
        # Gripper bools for live control state are not changed by playback
        current_left_gripper_bool = previous_left_gripper_bool
        current_right_gripper_bool = previous_right_gripper_bool
        # Keyboard toggle state also remains unchanged by playback
        new_keyboard_toggle_state_to_remember = last_processed_keyboard_gripper_toggle_state
    # Live teleoperation (e.g., keyboard): tuple (delta_pose_6D_np, gripper_cmd_bool)
    elif isinstance(teleop_data, tuple) and len(teleop_data) == 2 and isinstance(teleop_data[0], np.ndarray):
        delta_pose_6D_np, current_keyboard_gripper_toggle_state = teleop_data
        # Start with previous states
        target_left_eef_pos_w = previous_target_left_eef_pos_w
        target_left_eef_quat_wxyz_w = previous_target_left_eef_quat_wxyz_w
        current_left_gripper_bool = previous_left_gripper_bool
        target_right_eef_pos_w = previous_target_right_eef_pos_w
        target_right_eef_quat_wxyz_w = previous_target_right_eef_quat_wxyz_w
        current_right_gripper_bool = previous_right_gripper_bool

        # Check if a gripper command was issued in this step by seeing if keyboard toggle changed
        gripper_command_issued_this_step = (current_keyboard_gripper_toggle_state != last_processed_keyboard_gripper_toggle_state)

        if active_hand == "right":
            # Compose new orientation using axis-angle delta for right hand
            target_right_eef_pos_w = previous_target_right_eef_pos_w + delta_pose_6D_np[:3]
            delta_rot = R.from_rotvec(delta_pose_6D_np[3:6])
            target_right_eef_quat_wxyz_w = (delta_rot * R.from_quat(previous_target_right_eef_quat_wxyz_w[[1,2,3,0]])).as_quat()
            target_right_eef_quat_wxyz_w = target_right_eef_quat_wxyz_w[[3,0,1,2]]
            if gripper_command_issued_this_step:
                current_right_gripper_bool = bool(current_keyboard_gripper_toggle_state)
            # Else: current_left_gripper_bool remains its previous_left_gripper_bool value
        elif active_hand == "left":
            target_left_eef_pos_w = previous_target_left_eef_pos_w + delta_pose_6D_np[:3]
            delta_rot = R.from_rotvec(delta_pose_6D_np[3:6])
            target_left_eef_quat_wxyz_w = (delta_rot * R.from_quat(previous_target_left_eef_quat_wxyz_w[[1,2,3,0]])).as_quat()
            target_left_eef_quat_wxyz_w = target_left_eef_quat_wxyz_w[[3,0,1,2]]
            if gripper_command_issued_this_step:
                current_left_gripper_bool = bool(current_keyboard_gripper_toggle_state)
            # Else: current_left_gripper_bool remains its previous_left_gripper_bool value

        # Create hand joint positions using TrajectoryPlayer's utility function
        hand_positions = trajectory_player.create_hand_joint_positions(
            left_hand_bool=current_left_gripper_bool,
            right_hand_bool=current_right_gripper_bool
        )

        # Concatenate all components to form the final action array (wxyz)
        left_arm_eef = np.concatenate([target_left_eef_pos_w, target_left_eef_quat_wxyz_w])
        right_arm_eef = np.concatenate([target_right_eef_pos_w, target_right_eef_quat_wxyz_w])
        # [left_arm_eef (7), right_arm_eef (7), hand_positions (14)]
        action_array_28D_np = np.concatenate([left_arm_eef, right_arm_eef, hand_positions])
        new_keyboard_toggle_state_to_remember = current_keyboard_gripper_toggle_state
    else:
        raise ValueError(f"Unexpected teleop_data format for G1 task: {teleop_data}")

    actions = torch.tensor(action_array_28D_np, dtype=torch.float, device=device).repeat(num_envs, 1)
    return (  # type: ignore
        actions,
        target_left_eef_pos_w, target_left_eef_quat_wxyz_w, bool(current_left_gripper_bool),
        target_right_eef_pos_w, target_right_eef_quat_wxyz_w, bool(current_right_gripper_bool),
        bool(new_keyboard_toggle_state_to_remember)
    )

def main():
    """
    Main entry point for keyboard teleoperation with Isaac Lab manipulation environment.
    Handles environment setup, teleop interface, and simulation loop.
    """
    # --- Helper functions for callbacks and state management ---
    def reset_env_and_player():
        nonlocal should_reset_recording_instance
        if allow_env_reset:
            should_reset_recording_instance = True
            if trajectory_player.is_playing_back:
                trajectory_player.is_playing_back = False
                print("Playback stopped due to environment reset request.")
        else:
            print("[INFO] Environment reset is currently disabled (allow_env_reset=False)")

    def toggle_active_hand():
        nonlocal active_hand
        active_hand = "left" if active_hand == "right" else "right"
        print(f"[INFO] Active hand switched to: {active_hand.upper()}.")

    def play_open_drawer_trajectory():
        """Generate and play the open drawer trajectory."""
        generator = FileBasedTrajectoryGenerator(obs, filepath=f"scripts/gr00t_script/configs/open_drawer_waypoints_{G1_HAND_TYPE}.yaml")
        trajectory_player.set_waypoints(generator.generate())
        trajectory_player.prepare_playback_trajectory()

    def setup_teleop_interface_and_callbacks(teleop_interface_obj, trajectory_player_obj, reset_env_func, toggle_hand_func):
        """
        Sets up teleoperation callbacks and prints control information.
        """
        teleop_interface_obj.add_callback("P", lambda: trajectory_player_obj.record_current_pose(obs, current_left_gripper_bool, current_right_gripper_bool))
        teleop_interface_obj.add_callback("L", lambda: trajectory_player_obj.load_and_playback())
        teleop_interface_obj.add_callback("M", trajectory_player_obj.clear_waypoints)
        teleop_interface_obj.add_callback("N", lambda: trajectory_player_obj.save_waypoints())
        teleop_interface_obj.add_callback("R", reset_env_func)
        teleop_interface_obj.add_callback("U", toggle_hand_func)
        teleop_interface_obj.add_callback("O", play_open_drawer_trajectory)

        print("\n--- Teleoperation Interface Controls ---")
        print(teleop_interface_obj)
        print("\n--- Trajectory Player Controls for Unitree G1 ---")
        print("  P: Record current EE pose as waypoint.")
        print("  L: Prepare and start playback of recorded trajectory.")
        print("  M: Clear all recorded waypoints from memory.")
        print("  N: Save current waypoints to 'waypoints.json'.")
        print("  R: Reset environment (also stops playback).")
        print("  U: Toggle active hand for teleoperation (LEFT/RIGHT).")
        print("  O: Play the open drawer trajectory.")
        print("------------------------------------\n")

    # --- Environment and teleop setup ---
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)

    # Disable the termination term for the teleoperation script
    env_cfg.terminations = None  # type: ignore

    # Conditionally disable randomization events based on the loaded environment.
    if hasattr(env_cfg.events, "randomize_cube_positions") or hasattr(env_cfg.events, "randomize_cube1_positions"):
        print("[INFO] Disabling cube position randomization for teleoperation.")
        if hasattr(env_cfg.events, "randomize_cube1_positions"):
            setattr(env_cfg.events, "randomize_cube1_positions", None)
        if hasattr(env_cfg.events, "randomize_cube_positions"):
            setattr(env_cfg.events, "randomize_cube_positions", None)
    elif hasattr(env_cfg.events, "reset_bottle"):
        print("[INFO] Disabling object pose randomization for teleoperation (bottle, mug, etc.).")
        setattr(env_cfg.events, "reset_bottle", None)
        setattr(env_cfg.events, "reset_mug", None)
        setattr(env_cfg.events, "reset_mug_mat", None)

    env = cast(ManagerBasedRLEnv, gym.make(args_cli.task, cfg=env_cfg).unwrapped)
    print(f"The environment '{args_cli.task}' uses absolute 6D pose control for the right arm eef and right hand.")

    # Initialize environment, TrajectoryPlayer, and teleop interface
    obs, _ = env.reset()
    
    # Teleop state flags
    should_reset_recording_instance = False
    teleoperation_active = True
    active_hand = "right"
    allow_env_reset = True
    
    # Gripper states
    current_left_gripper_bool = False
    current_right_gripper_bool = False
    last_processed_keyboard_gripper_toggle_state = False

    # Setup teleop interface
    teleop_interface = Se3Keyboard(
        Se3KeyboardCfg(
            pos_sensitivity=0.005 * args_cli.sensitivity,
            rot_sensitivity=0.02 * args_cli.sensitivity
        )
    )
    trajectory_player = TrajectoryPlayer(env, initial_obs=obs)
    setup_teleop_interface_and_callbacks(teleop_interface, trajectory_player, reset_env_and_player, toggle_active_hand)

    teleop_interface.reset()
    previous_target_left_eef_pos_w = trajectory_player.initial_left_arm_pos_w
    previous_target_left_eef_quat_wxyz_w = trajectory_player.initial_left_arm_quat_wxyz_w
    previous_target_right_eef_pos_w = trajectory_player.initial_right_arm_pos_w
    previous_target_right_eef_quat_wxyz_w = trajectory_player.initial_right_arm_quat_wxyz_w

    # --- Main simulation loop ---
    while simulation_app.is_running():
        with torch.inference_mode():
            if should_reset_recording_instance:
                print("[INFO] Resetting environment and teleop state due to 'should_reset_recording_instance' flag.")
                obs, _ = env.reset()
                teleop_interface.reset()
                last_processed_keyboard_gripper_toggle_state = False # Keyboard gripper is reset to False (open)
                previous_target_left_eef_pos_w = trajectory_player.initial_left_arm_pos_w
                previous_target_left_eef_quat_wxyz_w = trajectory_player.initial_left_arm_quat_wxyz_w
                previous_target_right_eef_pos_w = trajectory_player.initial_right_arm_pos_w
                previous_target_right_eef_quat_wxyz_w = trajectory_player.initial_right_arm_quat_wxyz_w
                
                current_left_gripper_bool = False
                current_right_gripper_bool = False
                active_hand = "right" # Reset active hand to right
                should_reset_recording_instance = False

            raw_teleop_device_output = teleop_interface.advance()
            actions_to_step = None

            if trajectory_player.is_playing_back:
                playback_action_tuple = trajectory_player.get_formatted_action_for_playback(obs=obs)
                if playback_action_tuple is not None:
                    (
                        actions_to_step,
                        previous_target_left_eef_pos_w, previous_target_left_eef_quat_wxyz_w, current_left_gripper_bool,
                        previous_target_right_eef_pos_w, previous_target_right_eef_quat_wxyz_w, current_right_gripper_bool,
                        last_processed_keyboard_gripper_toggle_state
                    ) = pre_process_actions(  # type: ignore
                        playback_action_tuple,
                        active_hand, # active_hand state is maintained but not used by playback to form action
                        env.num_envs,
                        env.device,
                        previous_target_left_eef_pos_w, previous_target_left_eef_quat_wxyz_w, current_left_gripper_bool,
                        previous_target_right_eef_pos_w, previous_target_right_eef_quat_wxyz_w, current_right_gripper_bool,
                        last_processed_keyboard_gripper_toggle_state, # Pass through
                        trajectory_player
                    )
            elif teleoperation_active:
                raw_teleop_device_output_np = raw_teleop_device_output.cpu().numpy()
                processed_input_for_action_fn = (raw_teleop_device_output_np[:-1], bool(raw_teleop_device_output_np[-1]-1))
                if actions_to_step is None:
                    (
                        actions_to_step,
                        previous_target_left_eef_pos_w, previous_target_left_eef_quat_wxyz_w, current_left_gripper_bool,
                        previous_target_right_eef_pos_w, previous_target_right_eef_quat_wxyz_w, current_right_gripper_bool,
                        new_keyboard_toggle_state_to_remember
                    ) = pre_process_actions(
                        processed_input_for_action_fn,
                        active_hand,
                        env.num_envs,
                        env.device,
                        previous_target_left_eef_pos_w, previous_target_left_eef_quat_wxyz_w, current_left_gripper_bool, # Pass current as previous for input
                        previous_target_right_eef_pos_w, previous_target_right_eef_quat_wxyz_w, current_right_gripper_bool, # Pass current as previous for input
                        last_processed_keyboard_gripper_toggle_state, # Pass the keyboard toggle state from *before* current advance()
                        trajectory_player
                    )
                    last_processed_keyboard_gripper_toggle_state = new_keyboard_toggle_state_to_remember

            if actions_to_step is not None:
                # actions_to_step: [left_arm_eef(7), right_arm_eef(7), left_hand(7), right_hand(7)]
                obs, _, _, _, _ = env.step(actions_to_step)
            else:
                env.sim.render()

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
