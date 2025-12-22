# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utility class for recording and playing back end-effector trajectories."""

import json
import os

import numpy as np
from scipy.spatial.transform import Rotation, Slerp # type: ignore
from termcolor import colored

# GraspPoseCalculator is no longer needed here
from . import constants as C
from .quaternion_utils import quat_xyzw_to_wxyz, quat_wxyz_to_xyzw


class TrajectoryPlayer:
    """
    Handles recording, saving, loading, and playing back end-effector trajectories for G1.

    A trajectory is a sequence of waypoints, where each waypoint includes the
    right EEF's absolute position, orientation (as a quaternion), and the
    target joint positions for the right hand.
    """
    def __init__(self, env, initial_obs: dict, steps_per_movement_segment=90, steps_per_grasp_segment=15, steps_per_shortshift_segment=30):
        """
        Initializes the TrajectoryPlayer for G1.

        Args:
            env: The Isaac Lab environment instance.
            initial_obs: Optional dictionary containing initial observations from env.reset().
            steps_per_movement_segment: The number of simulation steps for interpolating movement segments.
            steps_per_grasp_segment: The number of simulation steps for interpolating grasp/release segments.
            steps_per_shortshift_segment: The number of simulation steps for interpolating short-shift segments.
        """
        self.env = env

        # Extract initial poses and target info using the helper
        (self.initial_left_arm_pos_w, self.initial_left_arm_quat_wxyz_w, 
         self.initial_right_arm_pos_w, self.initial_right_arm_quat_wxyz_w,
         *_) = self.extract_essential_obs_data(initial_obs) # All other data (cubes, cans) not used at init
        
        print(colored(f"[INFO] TrajectoryPlayer using Left Arm Pos: {self.initial_left_arm_pos_w}, Quat: {self.initial_left_arm_quat_wxyz_w}", "yellow"))
        print(colored(f"[INFO] TrajectoryPlayer using Right Arm Pos: {self.initial_right_arm_pos_w}, Quat: {self.initial_right_arm_quat_wxyz_w}", "yellow"))
        
        if initial_obs.get("scene_obs", {}).get("object_obs") is not None:
            (_, _, _, _, cube1_pos, cube1_quat, cube2_pos, cube2_quat, cube3_pos, cube3_quat, *_ ) = self.extract_essential_obs_data(initial_obs)
            print(colored(f"[INFO] Cube 1 Pose: {cube1_pos}, Quat: {cube1_quat}", "yellow"))
            print(colored(f"[INFO] Cube 2 Pose: {cube2_pos}, Quat: {cube2_quat}", "yellow"))
            print(colored(f"[INFO] Cube 3 Pose: {cube3_pos}, Quat: {cube3_quat}", "yellow"))
        elif initial_obs.get("scene_obs", {}).get("drawer_pose") is not None:
            (_, _, _, _, _, _, _, _, _, _, _, _, _, drawer_pos, drawer_quat, bottle_pos, bottle_quat, mug_pos, mug_quat, mug_mat_pos, mug_mat_quat) = self.extract_essential_obs_data(initial_obs)
            print(colored(f"[INFO] Drawer Position: {drawer_pos}, Quat: {drawer_quat}", "yellow"))
            print(colored(f"[INFO] Mug Position: {mug_pos}, Quat: {mug_quat}", "yellow"))
            print(colored(f"[INFO] Mug Mat Position: {mug_mat_pos}, Quat: {mug_mat_quat}", "yellow"))
            print(colored(f"[INFO] Bottle Position: {bottle_pos}, Quat: {bottle_quat}", "yellow"))

        # {"left_arm_eef"(7), "right_arm_eef"(7), "left_hand", "right_hand"}
        self.recorded_waypoints = []
        self.playback_trajectory_actions = []
        self.current_playback_idx = 0
        self.is_playing_back = False
        self.steps_per_movement_segment = steps_per_movement_segment
        self.steps_per_grasp_segment = steps_per_grasp_segment
        self.steps_per_shortshift_segment = steps_per_shortshift_segment

        # Get hand joint names from the action manager
        self.pink_hand_joint_names = self.env.action_manager._terms["arm_action_cfg"].cfg.hand_joint_names
        
        # Joint tracking data
        self.joint_tracking_records = []
        self.joint_tracking_active = False

    @staticmethod
    def extract_essential_obs_data(obs: dict) -> tuple:
        """
        Helper to extract common observation data from the first environment.
        For cube stacking, it expects `object_obs` to contain poses for three cubes.
        For can pick-place, it expects `target_object_pose`.
        """
        left_eef_pos = obs["robot_obs"]["left_eef_pos"][0].cpu().numpy()
        left_eef_quat = obs["robot_obs"]["left_eef_quat"][0].cpu().numpy()
        right_eef_pos = obs["robot_obs"]["right_eef_pos"][0].cpu().numpy()
        right_eef_quat = obs["robot_obs"]["right_eef_quat"][0].cpu().numpy()

        object_obs = None
        cube1_pos, cube1_quat, cube2_pos, cube2_quat, cube3_pos, cube3_quat = None, None, None, None, None, None
        target_can_pos, target_can_quat, target_can_color_id = None, None, None
        drawer_pos, drawer_quat, bottle_pos, bottle_quat, mug_pos, mug_quat, mug_mat_pos, mug_mat_quat = None, None, None, None, None, None, None, None
        
        if obs.get("scene_obs", {}).get("object_obs") is not None: # For cube stacking
            object_obs = obs["scene_obs"]["object_obs"][0].cpu().numpy()
            if len(object_obs) >= 21: # 3 cubes * (3 pos + 4 quat)
                cube1_pos, cube1_quat = object_obs[0 : 3], object_obs[3 : 7]
                cube2_pos, cube2_quat = object_obs[7 : 10], object_obs[10 : 14]
                cube3_pos, cube3_quat = object_obs[14 : 17], object_obs[17 : 21]
        
        if obs.get("scene_obs", {}).get("target_object_pose") is not None: # For can pick-place
            target_can_pose_obs = obs["scene_obs"]["target_object_pose"][0].cpu().numpy()
            target_can_pos, target_can_quat = target_can_pose_obs[:3], target_can_pose_obs[3:7]
            target_can_color_id = target_can_pose_obs[-1]

        if obs.get("scene_obs", {}).get("drawer_pose") is not None:
            object_obs = obs["scene_obs"]["drawer_pose"][0].cpu().numpy()
            drawer_pos, drawer_quat = object_obs[:3], object_obs[3:7]
        if obs.get("scene_obs", {}).get("bottle_pose") is not None:
            object_obs = obs["scene_obs"]["bottle_pose"][0].cpu().numpy()
            bottle_pos, bottle_quat = object_obs[:3], object_obs[3:7]
        if obs.get("scene_obs", {}).get("mug_pose") is not None:
            object_obs = obs["scene_obs"]["mug_pose"][0].cpu().numpy()
            mug_pos, mug_quat = object_obs[:3], object_obs[3:7]
        if obs.get("scene_obs", {}).get("mug_mat_pose") is not None:
            object_obs = obs["scene_obs"]["mug_mat_pose"][0].cpu().numpy()
            mug_mat_pos, mug_mat_quat = object_obs[:3], object_obs[3:7]

        return (left_eef_pos, left_eef_quat, right_eef_pos, right_eef_quat,
                cube1_pos, cube1_quat, cube2_pos, cube2_quat, cube3_pos, cube3_quat,
                target_can_pos, target_can_quat, target_can_color_id,
                drawer_pos, drawer_quat, bottle_pos, bottle_quat, mug_pos, mug_quat, mug_mat_pos, mug_mat_quat)

    def get_idle_action_np(self) -> np.ndarray:
        """
        Constructs a 28D numpy array representing an idle action.
        This action aims to keep the arms at their initial (or default) poses with hands open.

        Returns:
            np.ndarray: The 28D action array [left_eef(7), right_eef(7), hands(14)].
        """
        hand_positions = self.create_hand_joint_positions(left_hand_bool=False, right_hand_bool=False) # Open hands
        idle_action_np = np.concatenate([
            self.initial_left_arm_pos_w,
            self.initial_left_arm_quat_wxyz_w,
            self.initial_right_arm_pos_w,
            self.initial_right_arm_quat_wxyz_w,
            hand_positions
        ])
        return idle_action_np
    
    def record_current_pose(self, obs: dict, current_left_gripper_bool: bool, current_right_gripper_bool: bool):
        """
        Record the current end-effector link pose and orientation for both right and left, and gripper bools.

        Args:
            obs: Observation dictionary from the environment.
            current_left_gripper_bool: Boolean state of the left gripper (True for closed).
            current_right_gripper_bool: Boolean state of the right gripper (True for closed).
        """
        # Get the end-effector link pose and orientation using the helper
        (left_arm_eef_pos, left_arm_eef_orient_wxyz, right_arm_eef_pos, right_arm_eef_orient_wxyz,
         cube1_pos, cube1_quat, cube2_pos, cube2_quat, cube3_pos, cube3_quat,
         target_can_pos, target_can_quat, target_can_color_id,
         drawer_pos, drawer_quat, bottle_pos, bottle_quat, mug_pos, mug_quat, mug_mat_pos, mug_mat_quat) = self.extract_essential_obs_data(obs) # Ignore cube/can data for manual recording

        # Store as structured dict per user request
        waypoint = {
            "left_arm_eef": np.concatenate([left_arm_eef_pos.flatten(), left_arm_eef_orient_wxyz.flatten()]),
            "right_arm_eef": np.concatenate([right_arm_eef_pos.flatten(), right_arm_eef_orient_wxyz.flatten()]),
            "left_hand_bool": int(current_left_gripper_bool),
            "right_hand_bool": int(current_right_gripper_bool)
        }
        self.recorded_waypoints.append(waypoint)
        
        print(f"Waypoint {len(self.recorded_waypoints)}th recorded:")
        print(f"    Left Arm EEF: [{(', '.join('{:.8f}'.format(x) for x in waypoint['left_arm_eef']))}]")
        print(f"    Right Arm EEF: [{(', '.join('{:.8f}'.format(x) for x in waypoint['right_arm_eef']))}]")
        if cube1_pos is not None and cube2_pos is not None and cube3_pos is not None :
            print(f"    Cube 1 Pose: {cube1_pos}, {cube1_quat}")
            print(f"    Cube 2 Pose: {cube2_pos}, {cube2_quat}")
            print(f"    Cube 3 Pose: {cube3_pos}, {cube3_quat}")
        if target_can_pos is not None:
            print(f"    Target Can Pose: {target_can_pos}, {target_can_quat}")
            print(f"    Target Can Color ID: {target_can_color_id}")
        if drawer_pos is not None and bottle_pos is not None and mug_pos is not None and mug_mat_pos is not None:
            print(f"    Drawer Pose: {drawer_pos}, {drawer_quat}")
            print(f"    Bottle Pose: {bottle_pos}, {bottle_quat}")
            print(f"    Mug Pose: {mug_pos}, {mug_quat}")
            print(f"    Mug Mat Pose: {mug_mat_pos}, {mug_mat_quat}")
        
        return


    def clear_waypoints(self):
        """
        Clears all recorded waypoints and stops playback if active.
        """
        self.recorded_waypoints = []
        self.playback_trajectory_actions = []
        if self.is_playing_back:
            self.is_playing_back = False
            print("Playback stopped and waypoints cleared.")


    def _record_joint_state(self, sim_time, reference_joints, current_joints):
        """
        Records joint states during trajectory playback.

        Args:
            reference_joints: List or array of reference/target joint angles
            current_joints: List or array of current joint angles
        """
        if not self.joint_tracking_active or not self.is_playing_back:
            return

        entry = {
            "timestamp": float(sim_time),
            "reference_joints": [float(x) for x in reference_joints],
            "current_joints": [float(x) for x in current_joints]
        }

        self.joint_tracking_records.append(entry)


    def _clear_joint_tracking_data(self):
        """
        Clears recorded joint tracking data and resets timing.
        """
        self.joint_tracking_records = []
        self.joint_tracking_active = False
        
    def _determine_segment_times(self, i, num_segments, left_hand_bools, right_hand_bools, left_arm_eef_pos, right_arm_eef_pos, left_rotations, right_rotations):
        """
        Determine the segment times for the trajectory based on the movement type and distance between waypoints.
        """
        # Determine if this segment involves a gripper action
        is_gripper_segment = (left_hand_bools[i] != left_hand_bools[i + 1]) or (
            right_hand_bools[i] != right_hand_bools[i + 1]
        )

        # Calculate positional and rotational differences between waypoints
        pos_diff_left = np.linalg.norm(left_arm_eef_pos[i + 1] - left_arm_eef_pos[i])
        pos_diff_right = np.linalg.norm(right_arm_eef_pos[i + 1] - right_arm_eef_pos[i])

        relative_rot_left = left_rotations[i + 1] * left_rotations[i].inv()
        angle_diff_left_deg = np.rad2deg(relative_rot_left.magnitude())

        relative_rot_right = right_rotations[i + 1] * right_rotations[i].inv()
        angle_diff_right_deg = np.rad2deg(relative_rot_right.magnitude())

        # Check if the maximum movement and rotation across both arms are small
        is_small_movement = (
            max(pos_diff_left, pos_diff_right) < C.TRAJECTORY_SMALL_MOVEMENT_POS_THRESHOLD and max(angle_diff_left_deg, angle_diff_right_deg) < C.TRAJECTORY_SMALL_MOVEMENT_ANGLE_THRESHOLD
        )

        if is_gripper_segment:
            num_points_in_segment = self.steps_per_grasp_segment
        elif is_small_movement:
            num_points_in_segment = self.steps_per_shortshift_segment
        else:
            num_points_in_segment = self.steps_per_movement_segment
        
        # Exclude the last point for all but the final segment to avoid duplicates
        return np.linspace(0, 1, num_points_in_segment, endpoint=(i == num_segments - 1))
    

    def _interpolate_arm_eef(self, arm_eef_pos, rotations, segment_times, i, smoother="smoothstep"):
        """ 
        Interpolate arm end-effector (Slerp for orientation and linear for position)
        """
        def smoothstep(t):
            """Smooth interpolation function (3t² - 2t³)"""
            return t * t * (3.0 - 2.0 * t)

        def smootherstep(t):
            """Even smoother (minimum jerk) interpolation function (6t⁵ - 15t⁴ + 10t³)"""
            return 6.0 * t**5 - 15.0 * t**4 + 10.0 * t**3

        if smoother == "smoothstep":
            smooth_times = smoothstep(segment_times)
        elif smoother == "smootherstep":
            smooth_times = smootherstep(segment_times)
        else:
            smooth_times = segment_times

        # Position interpolation
        interp_pos = (
            arm_eef_pos[i, None] * (1 - smooth_times[:, None])
            + arm_eef_pos[i + 1, None] * smooth_times[:, None]
        )
        
        # Orientation interpolation
        key_rots = Rotation.concatenate([rotations[i], rotations[i + 1]])
        slerp = Slerp([0, 1], key_rots)
        interp_orient_xyzw = slerp(smooth_times).as_quat()
        interp_orient_wxyz = quat_xyzw_to_wxyz(interp_orient_xyzw)
        
        return interp_pos, interp_orient_wxyz

    def _get_hand_joint_positions(self, left_hand_bools, right_hand_bools, i):
        # Set initial positions using _create_hand_joint_positions
        hand_joint_positions = self.create_hand_joint_positions(
            left_hand_bool=left_hand_bools[i], right_hand_bool=right_hand_bools[i]
        )
        next_hand_joint_positions = self.create_hand_joint_positions(
            left_hand_bool=left_hand_bools[i + 1], right_hand_bool=right_hand_bools[i + 1]
        )
        return hand_joint_positions, next_hand_joint_positions

    def prepare_playback_trajectory(self, is_continuation: bool = False):
        """
        Generates the interpolated trajectory steps from the recorded waypoints.

        Uses linear interpolation for end-effector position and Slerp for end-effector orientation.
        Uses linear interpolation for hand joints.

        Args:
            is_continuation: If True, skips the first waypoint to avoid a pause when chaining trajectories.
        """
        if len(self.recorded_waypoints) < 2:
            print("Not enough waypoints (need at least 2). Playback not started.")
            self.is_playing_back = False
            return

        self.playback_trajectory_actions = []
        left_arm_eef_pos = np.array([wp["left_arm_eef"][:3] for wp in self.recorded_waypoints])
        left_arm_eef_orient_wxyz = np.array([wp["left_arm_eef"][3:7] for wp in self.recorded_waypoints])
        left_hand_bools = [wp["left_hand_bool"] for wp in self.recorded_waypoints]
        right_arm_eef_pos = np.array([wp["right_arm_eef"][:3] for wp in self.recorded_waypoints])
        right_arm_eef_orient_wxyz = np.array([wp["right_arm_eef"][3:7] for wp in self.recorded_waypoints])
        right_hand_bools = [wp["right_hand_bool"] for wp in self.recorded_waypoints]

        # Convert wxyz quaternions to xyzw format for SciPy Rotation
        left_orient_xyzw = quat_wxyz_to_xyzw(left_arm_eef_orient_wxyz)
        left_rotations = Rotation.from_quat(left_orient_xyzw)
        
        right_orient_xyzw = quat_wxyz_to_xyzw(right_arm_eef_orient_wxyz)
        right_rotations = Rotation.from_quat(right_orient_xyzw)

        # Interpolate each segment
        num_segments = len(self.recorded_waypoints) - 1
        for i in range(num_segments):
            segment_times = self._determine_segment_times(i, num_segments, left_hand_bools, right_hand_bools, left_arm_eef_pos, right_arm_eef_pos, left_rotations, right_rotations)

            # Interpolate right arm end-effector (Slerp for orientation and linear for position)
            interp_right_pos, interp_right_orient_wxyz = self._interpolate_arm_eef(right_arm_eef_pos, right_rotations, segment_times, i)
            
            # Interpolate left arm end-effector
            interp_left_pos, interp_left_orient_wxyz = self._interpolate_arm_eef(left_arm_eef_pos, left_rotations, segment_times, i)
            
            # Interpolate hand joint states base on the order of the pink_hand_joint_names
            hand_joint_positions, next_hand_joint_positions = self._get_hand_joint_positions(left_hand_bools, right_hand_bools, i)

            # Store the interpolated 28D data for this segment [left_arm_eef(7), right_arm_eef(7), hand_joints(14)]
            for j in range(len(segment_times)):
                # If this is a continuation of a previous trajectory, skip the very first point (j=0 of i=0), which is a duplicate of the last point of the previous trajectory.
                if is_continuation and i == 0 and j == 0:
                    continue

                interp_hand_positions = hand_joint_positions * (1 - segment_times[j]) + next_hand_joint_positions * segment_times[j]

                action_array = np.concatenate([
                    np.concatenate([interp_left_pos[j], interp_left_orient_wxyz[j]]),   # left_arm_eef (7)
                    np.concatenate([interp_right_pos[j], interp_right_orient_wxyz[j]]), # right_arm_eef (7)
                    interp_hand_positions  # hand_joints (14)
                ])
                self.playback_trajectory_actions.append(action_array)

        self.current_playback_idx = 0
        self.is_playing_back = True
        self._clear_joint_tracking_data()  # Clear any previous tracking data
        self.joint_tracking_active = True  # Start joint tracking for this playback
        print(f"Playback trajectory prepared with {len(self.playback_trajectory_actions)} steps.")


    def get_formatted_action_for_playback(self, obs: dict):
        """
        Gets the next action command from the playback trajectory for G1.

        Returns:
            [left_arm_eef(7), right_arm_eef(7), hand_joints(14)]
        """
        if not self.is_playing_back or self.current_playback_idx >= len(self.playback_trajectory_actions):
            if self.joint_tracking_active:
                self.joint_tracking_active = False
            self.is_playing_back = False
            if self.current_playback_idx > 0 and len(self.playback_trajectory_actions) > 0:
                print("Playback finished.")
            return None

        # Get the action array for the current step
        action_array = self.playback_trajectory_actions[self.current_playback_idx]
        self.current_playback_idx += 1
        
        # Get current simulation time, reference/current joint positions from observation
        try:
            robot_art = self.env.unwrapped.scene.articulations["robot"]
            sim_time = float(robot_art.data._sim_timestamp)
            reference_joints = robot_art.data.joint_pos_target[0].cpu().numpy().tolist()
            current_joints = robot_art.data.joint_pos[0].cpu().numpy().tolist()
            
            self._record_joint_state(sim_time, reference_joints, current_joints)
                    
        except Exception as e:
            print(f"[TrajectoryPlayer ERROR] Error recording joint state: {e}")
            import traceback
            traceback.print_exc()

        return (action_array,)


    def save_waypoints(self, filepath=C.WAYPOINTS_JSON_PATH):
        """
        Saves the recorded waypoints to a JSON file.

        Args:
            filepath: The path to save the JSON file. Directories will be created if they don't exist.
        """
        if not self.recorded_waypoints:
            print("No waypoints to save.")
            return
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        waypoints_to_save = []
        for wp in self.recorded_waypoints:
            # Convert numpy arrays to lists for JSON serialization
            waypoints_to_save.append({
                "left_arm_eef": wp["left_arm_eef"].tolist(),
                "right_arm_eef": wp["right_arm_eef"].tolist(),
                "left_hand_bool": int(wp["left_hand_bool"]),
                "right_hand_bool": int(wp["right_hand_bool"])
            })
        try:
            with open(filepath, 'w') as f:
                json.dump(waypoints_to_save, f, indent=4)
            print(f"Waypoints saved to {filepath}")
        except Exception as e:
            print(f"[TrajectoryPlayer ERROR] Error saving waypoints to {filepath}: {e}")
            import traceback
            traceback.print_exc()

    def set_waypoints(self, waypoints: list):
        """
        Sets the internal waypoints from a provided list.

        Args:
            waypoints: A list of waypoint dictionaries.
        """
        if not isinstance(waypoints, list):
            print(f"[ERROR] Waypoints must be a list, but got {type(waypoints)}.")
            self.recorded_waypoints = []
            return
        self.recorded_waypoints = waypoints
        print(f"Set {len(self.recorded_waypoints)} waypoints.")

    def create_hand_joint_positions(self, left_hand_bool: bool, right_hand_bool: bool) -> np.ndarray:
        """Creates a hand joint positions array following the order of pink_hand_joint_names.
        
        Args:
            left_hand_bool: Boolean indicating if left hand should be closed (True) or open (False)
            right_hand_bool: Boolean indicating if right hand should be closed (True) or open (False)
            
        Returns:
            numpy.ndarray: Array of joint positions based on the open/closed states of the hands.
        """
        hand_joint_positions = np.zeros(len(self.pink_hand_joint_names))
        for idx, joint_name in enumerate(self.pink_hand_joint_names):
            if joint_name not in C.HAND_JOINT_POSITIONS:
                # This case should ideally not happen if pink_hand_joint_names are correctly subset of HAND_JOINT_POSITIONS keys
                #print(f"[TrajectoryPlayer WARNING] Joint name '{joint_name}' not found in HAND_JOINT_POSITIONS. Using 0.0.")
                continue
            
            # for G1-trihand and G1-inspire
            if "right" in joint_name or "R" in joint_name:
                hand_joint_positions[idx] = C.HAND_JOINT_POSITIONS[joint_name]["closed"] if right_hand_bool else C.HAND_JOINT_POSITIONS[joint_name]["open"]
            elif "left" in joint_name or "L" in joint_name:
                hand_joint_positions[idx] = C.HAND_JOINT_POSITIONS[joint_name]["closed"] if left_hand_bool else C.HAND_JOINT_POSITIONS[joint_name]["open"]
            # for OpenArm with Leaphand-right
            else:
                hand_joint_positions[idx] = C.HAND_JOINT_POSITIONS[joint_name]["closed"] if right_hand_bool else C.HAND_JOINT_POSITIONS[joint_name]["open"]

        return hand_joint_positions
    