# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utility class for recording and playing back end-effector trajectories."""

import json
import os
import yaml
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation, Slerp # type: ignore


# GraspPoseCalculator is now a direct runtime dependency
from .grasp_pose_calculator import GraspPoseCalculator
from .quaternion_utils import quat_xyzw_to_wxyz, quat_wxyz_to_xyzw


# === Constants for G1 Trajectory Generation ===
DEFAULT_LEFT_HAND_BOOL = False  # False for open

# Constants for red and blue basket pose
CAN_RADIUS = 0.025 #
RED_BASKET_CENTER = np.array([0.4 - CAN_RADIUS, -0.05 - CAN_RADIUS, 0.81])  # 
BLUE_BASKET_CENTER = np.array([0.4 - CAN_RADIUS*3, -0.2 - CAN_RADIUS, 0.81])  # 
        
# Define placement orientations for baskets (e.g., 90-degree yaw)
RED_BASKET_PLACEMENT_YAW_DEGREES = 0.0
RED_BASKET_PLACEMENT_QUAT_WXYZ = quat_xyzw_to_wxyz(Rotation.from_euler('z', RED_BASKET_PLACEMENT_YAW_DEGREES, degrees=True).as_quat())

BLUE_BASKET_PLACEMENT_YAW_DEGREES = -20.0
BLUE_BASKET_PLACEMENT_QUAT_WXYZ = quat_xyzw_to_wxyz(Rotation.from_euler('z', BLUE_BASKET_PLACEMENT_YAW_DEGREES, degrees=True).as_quat())

# Define joint positions for open and closed states (Left/right hand joint positions are opposite.)
# Using a leading underscore to indicate it's intended for internal use within this module.
_HAND_JOINT_POSITIONS = {
    "left_hand_index_0_joint":   {"open": 0.0, "closed": -0.8},
    "left_hand_middle_0_joint":  {"open": 0.0, "closed": -0.8},
    "left_hand_thumb_0_joint":   {"open": 0.0, "closed": 0.0},
    
    "right_hand_index_0_joint":  {"open": 0.0, "closed": 1.0},
    "right_hand_middle_0_joint": {"open": 0.0, "closed": 1.0},
    "right_hand_thumb_0_joint":  {"open": 0.0, "closed": 0.0},
    
    "left_hand_index_1_joint":   {"open": 0.0, "closed": -0.8},
    "left_hand_middle_1_joint":  {"open": 0.0, "closed": -0.8},
    "left_hand_thumb_1_joint":   {"open": 0.0, "closed": 0.8},
    
    "right_hand_index_1_joint":  {"open": 0.0, "closed": 0.9},
    "right_hand_middle_1_joint": {"open": 0.0, "closed": 0.9},
    "right_hand_thumb_1_joint":  {"open": 0.0, "closed": -0.0},
    
    "left_hand_thumb_2_joint":   {"open": 0.0, "closed": 0.8},
    "right_hand_thumb_2_joint":  {"open": 0.0, "closed": -0.0},
}

# Default paths for saving waypoints and joint tracking logs
WAYPOINTS_JSON_PATH = os.path.join("logs", "teleoperation", "waypoints.json")
JOINT_TRACKING_LOG_PATH = os.path.join("logs", "teleoperation", "joint_tracking_log.json")

# === Constants for Cube Stacking Trajectory ===
CUBE_HEIGHT = 0.06 # Actual height of the cube
CUBE_STACK_ON_CUBE_Z_OFFSET = CUBE_HEIGHT + 0.005 # Target Z for top cube relative to bottom cube's origin (0.06 cube height + 0.005 buffer)
CUBE_STACK_PRE_GRASP_OFFSET_POS_CUBE_FRAME = np.array([-0.090, -0.001, 0.18])  # Relative to cube's origin and orientation
CUBE_STACK_PRE_GRASP_EULER_XYZ_DEG_CUBE_FRAME = np.array([-90.0, 30.0, 0.0]) # Relative to cube's orientation
CUBE_STACK_GRASP_APPROACH_DISTANCE_Z_WORLD = 0.05 # World Z-axis downward movement from pre-grasp EEF Z
CUBE_STACK_INTERMEDIATE_LIFT_HEIGHT_ABOVE_BASE = 0.25 # Z-offset for intermediate waypoints, relative to base of target stack cube

# === Constants for cabinet pouring tasks ===
# 1. Open the drawer handle (Relative to drawer's origin and orientation)
PRE_APPROACH_OFFSET_POS = np.array([-0.150, -0.050, 0.119])
APPROACH_OFFSET_POS     = np.array([-0.120, -0.004, 0.119])
PULL_OFFSET_POS         = np.array([-0.305, -0.055, 0.119])
PRE_APPROACH_OFFSET_QUAT = np.array([90, 50, 0])  # degrees, relative to drawer's orientation
# 2. Pick-and-place the mug (Relative to mug's origin and orientation)
MUG_PRE_GRASP_POS       = np.array([0.0, 0.0, 0.15])
MUG_APPROACH_POS        = np.array([0.0, 0.0, 0.10])
MUG_LIFT_POS            = np.array([0.0, 0.0, 0.2])
MUG_GRASP_QUAT          = np.array([90, 30, -70])

MAT_PLACE_POS           = np.array([0.0, 0.0, 0.15])
MAT_RETRACT_POS         = np.array([0.0, 0.0, 0.2])
MAT_PLACE_QUAT          = np.array([90, 30, -45])

DRAWER_PUSH_DIRECTION_LOCAL = np.array([0.185, 0, 0])   # -0.120 - (-0.305) = 0.185
DRAWER_PUSH_APPROACH_POS    = np.array([0.05, 0.0, 0.0])   # 0, 0, 0
DRAWER_PUSH_QUAT            = np.array([90, 50, 0])
# 3. pick the bottle, pour it, and place it back
BOTTLE_GRASP_QUAT       = np.array([0.0, 0.0, -180])  # degrees, relative to bottle's orientation (yaw = 180 deg)
BOTTLE_GRASP_POS        = np.array([-0.075, -0.105, -0.07])
BOTTLE_LIFT_UP_OFFSET   = np.array([0.0, 0.0, 0.10])
BOTTLE_PRE_POUR_OFFSET  = np.array([0.09316173, -0.20843283, 0.1252327])    # TODO: Need to switch to w.r.t. the mug's frame (instead of mat's frame)

BOTTLE_POURING_OFFSET = np.array([-0.09339046, -0.13, 0.175])
BOTTLE_POUR_ANGLE_DEG   = -100



class TrajectoryPlayer:
    """
    Handles recording, saving, loading, and playing back end-effector trajectories for G1.

    A trajectory is a sequence of waypoints, where each waypoint includes the
    right EEF's absolute position, orientation (as a quaternion), and the
    target joint positions for the right hand.
    """
    def __init__(self, env, initial_obs: dict, steps_per_movement_segment=100, steps_per_grasp_segment=50):
        """
        Initializes the TrajectoryPlayer for G1.

        Args:
            env: The Isaac Lab environment instance.
            initial_obs: Optional dictionary containing initial observations from env.reset().
            steps_per_movement_segment: The number of simulation steps for interpolating movement segments.
            steps_per_grasp_segment: The number of simulation steps for interpolating grasp/release segments.
        """
        self.env = env

        # Extract initial poses and target info using the helper
        (self.initial_left_arm_pos_w, self.initial_left_arm_quat_wxyz_w, 
         self.initial_right_arm_pos_w, self.initial_right_arm_quat_wxyz_w,
         *_) = self.extract_essential_obs_data(initial_obs) # All other data (cubes, cans) not used at init
        
        # # Assign the initial target poses for right EEFs to match the offset frame in FrameTransformerCfg
        # self.initial_right_arm_pos_w = [0.0640, -0.24,  0.9645]
        # self.initial_right_arm_quat_wxyz_w = [0.9828103, -0.10791296, -0.01653928, -0.14887986]
                
        print(f"[INFO] TrajectoryPlayer using Left Arm Pos: {self.initial_left_arm_pos_w}, Quat: {self.initial_left_arm_quat_wxyz_w}")
        print(f"[INFO] TrajectoryPlayer using Right Arm Pos: {self.initial_right_arm_pos_w}, Quat: {self.initial_right_arm_quat_wxyz_w}")
        
        if initial_obs.get("policy", {}).get("object_obs") is not None:
            (_, _, _, _, cube1_pos, cube1_quat, cube2_pos, cube2_quat, cube3_pos, cube3_quat, *_ ) = self.extract_essential_obs_data(initial_obs)
            print(f"[INFO] Cube 1 Pose: {cube1_pos}, Quat: {cube1_quat}")
            print(f"[INFO] Cube 2 Pose: {cube2_pos}, Quat: {cube2_quat}")
            print(f"[INFO] Cube 3 Pose: {cube3_pos}, Quat: {cube3_quat}")
        elif initial_obs.get("policy", {}).get("drawer_pos") is not None:
            (_, _, _, _, _, _, _, _, _, _, _, _, _, drawer_pos, drawer_quat, bottle_pos, bottle_quat, mug_pos, mug_quat, mug_mat_pos, mug_mat_quat) = self.extract_essential_obs_data(initial_obs)
            print(f"[INFO] Drawer Position: {drawer_pos}, Quat: {drawer_quat}")
            print(f"[INFO] Mug Position: {mug_pos}, Quat: {mug_quat}")
            print(f"[INFO] Mug Mat Position: {mug_mat_pos}, Quat: {mug_mat_quat}")
            print(f"[INFO] Bottle Position: {bottle_pos}, Quat: {bottle_quat}")

        # {"left_arm_eef"(7), "right_arm_eef"(7), "left_hand", "right_hand"}
        self.recorded_waypoints = []
        self.playback_trajectory_actions = []
        self.current_playback_idx = 0
        self.is_playing_back = False
        self.grasp_calculator = GraspPoseCalculator() # For can pick-place
        self.steps_per_movement_segment = steps_per_movement_segment
        self.steps_per_grasp_segment = steps_per_grasp_segment

        # Get hand joint names from the action manager
        self.pink_hand_joint_names = self.env.action_manager._terms["pink_ik_cfg"].cfg.hand_joint_names
        
        # print("[TrajectoryPlayer] Initialized with hand joint names:", self.pink_hand_joint_names)
        # ['left_hand_index_0_joint', 'left_hand_middle_0_joint', 'left_hand_thumb_0_joint', 'right_hand_index_0_joint', 'right_hand_middle_0_joint', 'right_hand_thumb_0_joint', 'left_hand_index_1_joint', 'left_hand_middle_1_joint', 'left_hand_thumb_1_joint', 'right_hand_index_1_joint', 'right_hand_middle_1_joint', 'right_hand_thumb_1_joint', 'left_hand_thumb_2_joint', 'right_hand_thumb_2_joint']
        
        # Joint tracking data
        self.joint_tracking_records = []
        self.joint_tracking_active = False

    def extract_essential_obs_data(self, obs: dict) -> tuple:
        """
        Helper to extract common observation data from the first environment.
        For cube stacking, it expects `object_obs` to contain poses for three cubes.
        For can pick-place, it expects `target_object_pose` and `target_object_id`.
        """
        left_eef_pos = obs["policy"]["left_eef_pos"][0].cpu().numpy()
        left_eef_quat = obs["policy"]["left_eef_quat"][0].cpu().numpy()
        right_eef_pos = obs["policy"]["right_eef_pos"][0].cpu().numpy()
        right_eef_quat = obs["policy"]["right_eef_quat"][0].cpu().numpy()

        object_obs = None
        cube1_pos, cube1_quat, cube2_pos, cube2_quat, cube3_pos, cube3_quat = None, None, None, None, None, None
        target_can_pos, target_can_quat, target_can_color_id = None, None, None
        drawer_pos, drawer_quat, bottle_pos, bottle_quat, mug_pos, mug_quat, mug_mat_pos, mug_mat_quat = None, None, None, None, None, None, None, None
        
        if "object_obs" in obs["policy"] and obs["policy"]["object_obs"] is not None: # For cube stacking
            object_obs = obs["policy"]["object_obs"][0].cpu().numpy()
            if len(object_obs) >= 21: # 3 cubes * (3 pos + 4 quat)
                cube1_pos, cube1_quat = object_obs[0 : 3], object_obs[3 : 7]
                cube2_pos, cube2_quat = object_obs[7 : 10], object_obs[10 : 14]
                cube3_pos, cube3_quat = object_obs[14 : 17], object_obs[17 : 21]
        
        if "target_object_pose" in obs["policy"] and "target_object_id" in obs["policy"]: # For can pick-place
            target_can_pose_obs = obs["policy"]["target_object_pose"][0].cpu().numpy()
            target_can_pos, target_can_quat = target_can_pose_obs[:3], target_can_pose_obs[3:7]
            target_can_color_id = obs["policy"]["target_object_id"][0].cpu().numpy().item()

        if "drawer_pos" in obs["policy"]:
            object_obs = obs["policy"]["drawer_pos"][0].cpu().numpy()
            drawer_pos, drawer_quat = object_obs[:3], np.array([1.0, 0.0, 0.0, 0.0]) # No drawer_quat ... 
        if "bottle_pose" in obs["policy"]:
            object_obs = obs["policy"]["bottle_pose"][0].cpu().numpy()
            bottle_pos, bottle_quat = object_obs[:3], object_obs[3:7]
        if "mug_pose" in obs["policy"]:
            object_obs = obs["policy"]["mug_pose"][0].cpu().numpy()
            mug_pos, mug_quat = object_obs[:3], object_obs[3:7]
        if "mug_mat_pose" in obs["policy"]:
            object_obs = obs["policy"]["mug_mat_pose"][0].cpu().numpy()
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
         *_) = self.extract_essential_obs_data(obs) # Ignore cube/can data for manual recording

        # Extract and print right arm joint angles
        all_joint_pos = obs["policy"]["robot_joint_pos"][0].cpu().numpy()
        robot_articulation = self.env.unwrapped.scene.articulations["robot"]
        all_joint_names = robot_articulation.joint_names
        right_arm_joint_names = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_yaw_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint"]
        right_arm_joint_angles = {name: all_joint_pos[all_joint_names.index(name)] for name in right_arm_joint_names if name in all_joint_names}
        
        print(f"  Right Arm Joint Angles: {right_arm_joint_angles}")

        # Store as structured dict per user request
        waypoint = {
            "left_arm_eef": np.concatenate([left_arm_eef_pos.flatten(), left_arm_eef_orient_wxyz.flatten()]),
            "right_arm_eef": np.concatenate([right_arm_eef_pos.flatten(), right_arm_eef_orient_wxyz.flatten()]),
            "left_hand_bool": int(current_left_gripper_bool),
            "right_hand_bool": int(current_right_gripper_bool)
        }
        self.recorded_waypoints.append(waypoint)
        print(f"Waypoint {len(self.recorded_waypoints)} recorded: {waypoint}")
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


    def load_and_playback(self, filepath=WAYPOINTS_JSON_PATH):
        """
        Loads waypoints from a file and prepares the trajectory for playback.

        Args:
            filepath: The path to the JSON file containing waypoints.
        """
        self.load_waypoints(filepath)
        self.prepare_playback_trajectory()


    def record_joint_state(self, sim_time, reference_joints, current_joints):
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


    def save_joint_tracking_data(self):
        """
        Saves the recorded joint tracking data to a JSON file.
        """
        if not self.joint_tracking_records:
            print("No joint tracking data to save.")
            return

        os.makedirs(os.path.dirname(JOINT_TRACKING_LOG_PATH), exist_ok=True)
        try:
            with open(JOINT_TRACKING_LOG_PATH, 'w') as f:
                # Convert all entries to JSON strings with indentation and join with commas
                json_lines = ',\n'.join(f'  {json.dumps(entry)}' for entry in self.joint_tracking_records)
                f.write('[\n' + json_lines + '\n]\n')
            print(f"Joint tracking data saved to {JOINT_TRACKING_LOG_PATH}")
        except Exception as e:
            print(f"[TrajectoryPlayer ERROR] Error saving joint tracking data: {e}")
            import traceback
            traceback.print_exc()

    def clear_joint_tracking_data(self):
        """
        Clears recorded joint tracking data and resets timing.
        """
        self.joint_tracking_records = []
        self.joint_tracking_active = False
        
    def prepare_playback_trajectory(self):
        """
        Generates the interpolated trajectory steps from the recorded waypoints.

        Uses linear interpolation for end-effector position and Slerp for end-effector orientation.
        Uses linear interpolation for hand joints.
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
            # Determine if this segment involves a gripper action
            is_gripper_segment = (left_hand_bools[i] != left_hand_bools[i+1]) or \
                                 (right_hand_bools[i] != right_hand_bools[i+1])
            
            num_points_in_segment = self.steps_per_grasp_segment if is_gripper_segment else self.steps_per_movement_segment
            segment_times = np.linspace(0, 1, num_points_in_segment, endpoint=(i == num_segments - 1))

            # Exclude the last point for all but the final segment to avoid duplicates
            segment_times = np.linspace(0, 1, num_points_in_segment, endpoint=(i == num_segments - 1))

            # Interpolate right arm end-effector (Slerp for orientation and linear for position)
            right_key_rots = Rotation.concatenate([right_rotations[i], right_rotations[i+1]])
            right_slerp = Slerp([0, 1], right_key_rots)
            interp_right_orient_xyzw = right_slerp(segment_times).as_quat() # xyzw format in SciPy
            interp_right_orient_wxyz = quat_xyzw_to_wxyz(interp_right_orient_xyzw)
            interp_right_pos = right_arm_eef_pos[i, None] * (1 - segment_times[:, None]) + right_arm_eef_pos[i+1, None] * segment_times[:, None]
            
            # Interpolate left arm end-effector
            left_key_rots = Rotation.concatenate([left_rotations[i], left_rotations[i+1]])
            left_slerp = Slerp([0, 1], left_key_rots)
            interp_left_orient_xyzw = left_slerp(segment_times).as_quat()
            interp_left_orient_wxyz = quat_xyzw_to_wxyz(interp_left_orient_xyzw)
            interp_left_pos = left_arm_eef_pos[i, None] * (1 - segment_times[:, None]) + left_arm_eef_pos[i+1, None] * segment_times[:, None]
            
            
            # Interpolate hand joint states base on the order of the pink_hand_joint_names
            hand_joint_positions = np.zeros(len(self.pink_hand_joint_names))
            next_hand_joint_positions = np.zeros(len(self.pink_hand_joint_names))

            # Set initial positions using create_hand_joint_positions
            hand_joint_positions = self.create_hand_joint_positions(
                left_hand_bool=left_hand_bools[i],
                right_hand_bool=right_hand_bools[i]
            )
            next_hand_joint_positions = self.create_hand_joint_positions(
                left_hand_bool=left_hand_bools[i+1],
                right_hand_bool=right_hand_bools[i+1]
            )

            # Store the interpolated 28D data for this segment [left_arm_eef(7), right_arm_eef(7), hand_joints(14)]
            for j in range(len(segment_times)):
                interp_hand_positions = hand_joint_positions * (1 - segment_times[j]) + next_hand_joint_positions * segment_times[j]

                action_array = np.concatenate([
                    np.concatenate([interp_left_pos[j], interp_left_orient_wxyz[j]]),   # left_arm_eef (7)
                    np.concatenate([interp_right_pos[j], interp_right_orient_wxyz[j]]), # right_arm_eef (7)
                    interp_hand_positions  # hand_joints (14)
                ])
                self.playback_trajectory_actions.append(action_array)

        self.current_playback_idx = 0
        self.is_playing_back = True
        self.clear_joint_tracking_data()  # Clear any previous tracking data
        self.joint_tracking_active = True  # Start joint tracking for this playback
        print(f"Playback trajectory prepared with {len(self.playback_trajectory_actions)} steps.")


    def get_formatted_action_for_playback(self):
        """
        Gets the next action command from the playback trajectory for G1.

        Returns:
            [left_arm_eef(7), right_arm_eef(7), hand_joints(14)]
        """
        if not self.is_playing_back or self.current_playback_idx >= len(self.playback_trajectory_actions):
            if self.joint_tracking_active:
                # self.save_joint_tracking_data()  # Saved replaced with parquet()
                self.joint_tracking_active = False
            self.is_playing_back = False
            if self.current_playback_idx > 0 and len(self.playback_trajectory_actions) > 0:
                print("Playback finished.")
            return None

        # Get the action array for the current step
        action_array = self.playback_trajectory_actions[self.current_playback_idx]
        self.current_playback_idx += 1
        # print(f"Playback step {self.current_playback_idx}/{len(self.playback_trajectory_actions)}: {action_array}")

        # Get current simulation time, reference/current joint positions from observation
        try:
            robot_art = self.env.unwrapped.scene.articulations["robot"]
            sim_time = float(robot_art.data._sim_timestamp)
            reference_joints = robot_art.data.joint_pos_target[0].cpu().numpy().tolist()
            current_joints = robot_art.data.joint_pos[0].cpu().numpy().tolist()
            
            self.record_joint_state(sim_time, reference_joints, current_joints)
                    
        except Exception as e:
            print(f"[TrajectoryPlayer ERROR] Error recording joint state: {e}")
            import traceback
            traceback.print_exc()

        return (action_array,)


    def save_waypoints(self, filepath=WAYPOINTS_JSON_PATH):
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

    def load_waypoints(self, filepath=WAYPOINTS_JSON_PATH):
        """ Loads waypoints from a JSON file. """
        try:
            with open(filepath, 'r') as f:
                loaded_wps_list = json.load(f)
        except FileNotFoundError:
            print(f"Waypoint file {filepath} not found. No waypoints loaded.")
            return
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {filepath}. File might be corrupted.")
            return
        except Exception as e:
            print(f"Error loading waypoints from {filepath}: {e}")
            import traceback
            traceback.print_exc()
            return
        
        self.recorded_waypoints = []
        for wp_dict in loaded_wps_list:
            # Convert lists back to numpy arrays, load in the new order
            self.recorded_waypoints.append({
                "left_arm_eef": np.array(wp_dict["left_arm_eef"]),
                "right_arm_eef": np.array(wp_dict["right_arm_eef"]),
                "left_hand_bool": int(wp_dict["left_hand_bool"]),
                "right_hand_bool": int(wp_dict["right_hand_bool"])
            })
        print(f"Waypoints loaded from {filepath}. {len(self.recorded_waypoints)} waypoints found.")

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
            if joint_name not in _HAND_JOINT_POSITIONS:
                # This case should ideally not happen if pink_hand_joint_names are correctly subset of _HAND_JOINT_POSITIONS keys
                print(f"[TrajectoryPlayer WARNING] Joint name '{joint_name}' not found in _HAND_JOINT_POSITIONS. Using 0.0.")
                continue
            if "right" in joint_name:
                hand_joint_positions[idx] = _HAND_JOINT_POSITIONS[joint_name]["closed"] if right_hand_bool else _HAND_JOINT_POSITIONS[joint_name]["open"]
            elif "left" in joint_name:
                hand_joint_positions[idx] = _HAND_JOINT_POSITIONS[joint_name]["closed"] if left_hand_bool else _HAND_JOINT_POSITIONS[joint_name]["open"]
        return hand_joint_positions

    def generate_auto_grasp_pick_place_trajectory(self, obs: dict):
        """
        Generates a predefined 7-waypoint trajectory for grasping a cube and placing it.

        Args:
            obs: The observation dictionary from the environment, containing current robot and object states.
        """
        (_, _, current_right_eef_pos_w, current_right_eef_quat_wxyz_w,
         _, _, _, _, _, _, # Cube poses not used by this function
         target_can_pos_w, target_can_quat_wxyz_w, target_can_color_id
         ) = self.extract_essential_obs_data(obs)
        # print(f"Current Right EEF Pose: pos={current_right_eef_pos_w}, quat_wxyz={current_right_eef_quat_wxyz_w}")
        print(f"Target Can Pose: pos={target_can_pos_w}, quat_wxyz={target_can_quat_wxyz_w}, color= {'red can' if target_can_color_id == 0 else 'blue can'}")

        self.clear_waypoints()
        # 1. Calculate target grasp pose for the right EEF
        target_grasp_right_eef_pos_w, target_grasp_right_eef_quat_wxyz_w = \
            self.grasp_calculator.calculate_target_ee_pose(target_can_pos_w, target_can_quat_wxyz_w)
        # print(f"Calculated Target Grasp Right EEF Pose: pos={target_grasp_right_eef_pos_w}, quat_wxyz={target_grasp_right_eef_quat_wxyz_w}")

        # Waypoint 1: Current EEF pose (right hand open)
        wp1_left_arm_eef = np.concatenate([self.initial_left_arm_pos_w, self.initial_left_arm_quat_wxyz_w])
        wp1_right_arm_eef = np.concatenate([current_right_eef_pos_w, current_right_eef_quat_wxyz_w])
        waypoint1 = {
            "left_arm_eef": wp1_left_arm_eef,
            "right_arm_eef": wp1_right_arm_eef,
            "left_hand_bool": int(DEFAULT_LEFT_HAND_BOOL),
            "right_hand_bool": 0
        }
        self.recorded_waypoints.append(waypoint1)

        # Waypoint 2: Target grasp EEF pose (right hand open - pre-grasp)
        wp2_left_arm_eef = np.concatenate([self.initial_left_arm_pos_w, self.initial_left_arm_quat_wxyz_w])
        wp2_right_arm_eef = np.concatenate([target_grasp_right_eef_pos_w, target_grasp_right_eef_quat_wxyz_w])
        waypoint2 = {
            "left_arm_eef": wp2_left_arm_eef,
            "right_arm_eef": wp2_right_arm_eef,
            "left_hand_bool": int(DEFAULT_LEFT_HAND_BOOL),
            "right_hand_bool": 0
        }
        self.recorded_waypoints.append(waypoint2)

        # Waypoint 3: Target grasp EEF pose (right hand closed - grasp)
        waypoint3 = {**waypoint2, "right_hand_bool": 1}
        self.recorded_waypoints.append(waypoint3)

        # Determine placement pose based on target object color
        if target_can_color_id == 0: # Red Can
            basket_base_target_pos_w = RED_BASKET_CENTER
            basket_target_quat_wxyz = RED_BASKET_PLACEMENT_QUAT_WXYZ
            print(f"Targeting RED basket at {basket_base_target_pos_w} with orientation {basket_target_quat_wxyz}")
        else: # Blue Can
            basket_base_target_pos_w = BLUE_BASKET_CENTER
            basket_target_quat_wxyz = BLUE_BASKET_PLACEMENT_QUAT_WXYZ
            print(f"Targeting BLUE basket at {basket_base_target_pos_w} with orientation {basket_target_quat_wxyz}")
        
        # Use X, Y from basket center, and Z from grasped object's EEF height for placement
        placement_target_pos_w = np.array([basket_base_target_pos_w[0], basket_base_target_pos_w[1], target_grasp_right_eef_pos_w[2] - 0.06]) # Fine tune -0.06

        # Waypoint 4: Intermediate lift pose
        lift_pos_x = (target_grasp_right_eef_pos_w[0] + placement_target_pos_w[0]) / 2
        lift_pos_y = (target_grasp_right_eef_pos_w[1] + placement_target_pos_w[1]) / 2
        lift_pos_z = max(target_grasp_right_eef_pos_w[2], placement_target_pos_w[2]) + 0.10 # Lift higher
        lift_pos_w = np.array([lift_pos_x, lift_pos_y, lift_pos_z])

        quat_grasp_xyzw = target_grasp_right_eef_quat_wxyz_w[[1, 2, 3, 0]]
        quat_placement_xyzw = basket_target_quat_wxyz[[1, 2, 3, 0]]
        key_rots = Rotation.from_quat([quat_grasp_xyzw, quat_placement_xyzw])
        slerp_interpolator = Slerp([0, 1], key_rots)
        lift_quat_xyzw = slerp_interpolator(0.5).as_quat()
        lift_quat_wxyz_w = lift_quat_xyzw[[3, 0, 1, 2]]

        waypoint4 = {
            "left_arm_eef": wp2_left_arm_eef,
            "right_arm_eef": np.concatenate([lift_pos_w, lift_quat_wxyz_w]),
            "left_hand_bool": int(DEFAULT_LEFT_HAND_BOOL),
            "right_hand_bool": 1
        }
        self.recorded_waypoints.append(waypoint4)

        # Waypoint 5: Move right arm EEF to basket placement pose (hand closed)
        waypoint5 = {
            "left_arm_eef": wp2_left_arm_eef,
            "right_arm_eef": np.concatenate([placement_target_pos_w, basket_target_quat_wxyz]),
            "left_hand_bool": int(DEFAULT_LEFT_HAND_BOOL),
            "right_hand_bool": 1
        }
        self.recorded_waypoints.append(waypoint5)

        # Waypoint 6: At RED_PLATE pose, open right hand
        waypoint6 = {**waypoint5, "right_hand_bool": 0}
        self.recorded_waypoints.append(waypoint6)

        # Waypoint 7: Move the right arm to the initial pose (hand open)
        waypoint7 = {**waypoint1, "right_hand_bool": 0} # Uses wp1's arm poses, ensures hand is open
        self.recorded_waypoints.append(waypoint7)

        # print(f"Generated {len(self.recorded_waypoints)} waypoints for auto grasp and place.")


    def _calculate_eef_world_pose_from_cube_relative(self,
                                               cube_pos_w: np.ndarray, cube_quat_wxyz_w: np.ndarray,
                                               eef_offset_pos_cube_frame: np.ndarray,
                                               eef_euler_xyz_deg_cube_frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Helper to calculate EEF world pose given cube world pose and EEF's relative pose to the cube."""
        R_w_cube = Rotation.from_quat(quat_wxyz_to_xyzw(cube_quat_wxyz_w))
        R_cube_eef_relative = Rotation.from_euler('xyz', eef_euler_xyz_deg_cube_frame, degrees=True)

        eef_target_pos_w = cube_pos_w + R_w_cube.apply(eef_offset_pos_cube_frame)
        R_w_eef_target = R_w_cube * R_cube_eef_relative # R_world_eef = R_world_cube * R_cube_eef
        eef_target_quat_wxyz_w = quat_xyzw_to_wxyz(R_w_eef_target.as_quat())
        return eef_target_pos_w, eef_target_quat_wxyz_w

    def _flatten_quat_around_world_z(self, quat_wxyz: np.ndarray, target_yaw_rad: Optional[float] = None) -> np.ndarray:
        """
        Takes a WXYZ quaternion, converts to Euler angles (ZYX order for world frame),
        zeros out roll (X) and pitch (Y). If target_yaw_rad is provided, it sets the yaw.
        Otherwise, it preserves the original yaw. Converts back to WXYZ quaternion.
        This effectively makes the object flat relative to the XY plane.
        """
        r = Rotation.from_quat(quat_wxyz_to_xyzw(quat_wxyz))
        euler_zyx = r.as_euler('zyx', degrees=False)  # zyx order: yaw, pitch, roll
        
        if target_yaw_rad is not None:
            euler_zyx[0] = target_yaw_rad # Set yaw
        # else: keep original yaw euler_zyx[0]
        
        euler_zyx[1] = 0.0  # Zero out pitch
        euler_zyx[2] = 0.0  # Zero out roll
        
        flat_rotation = Rotation.from_euler('zyx', euler_zyx, degrees=False)
        return quat_xyzw_to_wxyz(flat_rotation.as_quat())

    def generate_auto_stack_cubes_trajectory(self, obs: dict):
        """
        Generates a predefined trajectory for stacking three cubes.
        Order: Cube1 (e.g. red, bottom), Cube2 (e.g. green, middle), Cube3 (e.g. yellow, top).
        Cubes are stacked flat (zero roll, zero pitch).
        """
        (initial_left_pos, initial_left_quat,
         current_right_eef_pos_w, current_right_eef_quat_wxyz_w, # This is Waypoint 0
         cube1_pos_w, cube1_quat_wxyz_w, # Bottom cube
         cube2_pos_w, cube2_quat_wxyz_w, # Middle cube
         cube3_pos_w, cube3_quat_wxyz_w, # Top cube
         *_) = self.extract_essential_obs_data(obs)

        print("--- Generating Cube Stacking Trajectory ---")
        print(f"  Initial Right EEF (W0 Start): Pos={current_right_eef_pos_w}, Quat={current_right_eef_quat_wxyz_w}")
        print(f"  Cube 1 (Bottom) Initial: Pos={cube1_pos_w}, Quat={cube1_quat_wxyz_w}")
        print(f"  Cube 2 (Middle) Initial: Pos={cube2_pos_w}, Quat={cube2_quat_wxyz_w}")
        print(f"  Cube 3 (Top)    Initial: Pos={cube3_pos_w}, Quat={cube3_quat_wxyz_w}")

        self.clear_waypoints()
        left_arm_eef_static = np.concatenate([initial_left_pos, initial_left_quat])
        
        def add_waypoint(right_eef_pos, right_eef_quat, right_hand_closed_bool):
            wp = {
                "left_arm_eef": left_arm_eef_static,
                "right_arm_eef": np.concatenate([right_eef_pos, right_eef_quat]),
                "left_hand_bool": int(DEFAULT_LEFT_HAND_BOOL), 
                "right_hand_bool": int(right_hand_closed_bool)
            }
            self.recorded_waypoints.append(wp)

        # --- Calculate the fixed relative transformation (EEF in Cube's frame at grasp) ---
        # This is a one-time calculation using a virtual cube at the origin to find how the EEF
        # is positioned and oriented relative to a cube it's grasping.
        # This relative transform (t_cube_eef_in_cube_at_grasp, R_cube_eef_at_grasp)
        # will then be applied to the actual target cube poses.
        _temp_cube_pos = np.array([0.,0.,0.])
        _temp_cube_quat_flat_upright = np.array([1.,0.,0.,0.]) # Assume generic cube is flat for this calculation
        _eef_pregrasp_pos_rel_generic_cube, _eef_pregrasp_quat_rel_generic_cube = \
            self._calculate_eef_world_pose_from_cube_relative( # Output is effectively in generic cube's frame
                _temp_cube_pos, _temp_cube_quat_flat_upright, # Cube at origin, world frame = cube frame
                CUBE_STACK_PRE_GRASP_OFFSET_POS_CUBE_FRAME, CUBE_STACK_PRE_GRASP_EULER_XYZ_DEG_CUBE_FRAME)
        
        # Grasp pose is approached by moving down in Z from the pre-grasp pose (relative to the generic cube)
        _eef_grasp_pos_rel_generic_cube = _eef_pregrasp_pos_rel_generic_cube - np.array([0,0, CUBE_STACK_GRASP_APPROACH_DISTANCE_Z_WORLD])
        _eef_grasp_quat_rel_generic_cube = _eef_pregrasp_quat_rel_generic_cube 
        
        # This is T_cube_eef_grasp: transform of EEF in cube's frame when grasping
        t_cube_eef_in_cube_at_grasp = _eef_grasp_pos_rel_generic_cube # Position of EEF origin in cube's frame
        R_cube_eef_at_grasp = Rotation.from_quat(quat_wxyz_to_xyzw(_eef_grasp_quat_rel_generic_cube)) # Orientation of EEF in cube's frame

        # --- Waypoint 0: Current EEF pose (right hand open) ---
        add_waypoint(self.initial_right_arm_pos_w, self.initial_right_arm_quat_wxyz_w, False)

        # --- Process Cube 2 (grasp and place on Cube 1) ---
        # 1.1 Pre-grasp Cube 2 (approach based on its *current* orientation) -- waypoint 1
        pre_grasp_c2_pos_w, pre_grasp_c2_quat_w = self._calculate_eef_world_pose_from_cube_relative(
            cube2_pos_w, cube2_quat_wxyz_w, CUBE_STACK_PRE_GRASP_OFFSET_POS_CUBE_FRAME, CUBE_STACK_PRE_GRASP_EULER_XYZ_DEG_CUBE_FRAME)
        add_waypoint(pre_grasp_c2_pos_w, pre_grasp_c2_quat_w, False)
        # 1.2 Approach Cube 2 (move down in world Z) -- waypoint 2
        grasp_c2_pos_w = pre_grasp_c2_pos_w - np.array([0,0, CUBE_STACK_GRASP_APPROACH_DISTANCE_Z_WORLD])
        add_waypoint(grasp_c2_pos_w, pre_grasp_c2_quat_w, False) # Same orientation as pre-grasp
        # 1.3 Grasp Cube 2 -- waypoint 3
        add_waypoint(grasp_c2_pos_w, pre_grasp_c2_quat_w, True)
        # 1.4 Intermediate to Cube 1 -- waypoint 4
        intermediate_c1_pos_w = cube1_pos_w * 1 / 5 + cube2_pos_w * 4 / 5   # Weighted average towards Cube 2
        intermediate_c1_pos_w[2] = cube1_pos_w[2] + 4.0 * CUBE_HEIGHT # In case, the blue cube is too close to the green cube
        # Calculate intermediate orientation for Cube 2 placement
        cube1_yaw_rad_for_c2_stack = Rotation.from_quat(quat_wxyz_to_xyzw(cube1_quat_wxyz_w)).as_euler('zyx')[0]
        target_c2_on_c1_final_eef_quat_w = self._calculate_final_eef_orientation_for_stack(cube1_quat_wxyz_w, cube1_yaw_rad_for_c2_stack, R_cube_eef_at_grasp)
        intermediate_c1_orient_slerp = Slerp([0, 1], Rotation.from_quat(quat_wxyz_to_xyzw(np.array([pre_grasp_c2_quat_w, target_c2_on_c1_final_eef_quat_w]))))
        intermediate_c1_quat_w = quat_xyzw_to_wxyz(intermediate_c1_orient_slerp(0.5).as_quat()) # Midpoint orientation
        add_waypoint(intermediate_c1_pos_w, intermediate_c1_quat_w, True)        
        # 1.5 Stack Cube 2 on Cube 1 (Cube 2 will be flat, aligned in yaw with Cube 1)
        target_c2_on_c1_pos_w = cube1_pos_w + np.array([0,0, CUBE_STACK_ON_CUBE_Z_OFFSET])
        # Flatten Cube1's orientation to get target yaw for Cube2, ensuring Cube2 is placed flat.
        cube1_yaw_rad = Rotation.from_quat(quat_wxyz_to_xyzw(cube1_quat_wxyz_w)).as_euler('zyx')[0]
        target_c2_on_c1_quat_w_flat = self._flatten_quat_around_world_z(cube1_quat_wxyz_w, target_yaw_rad=cube1_yaw_rad)

        R_w_target_c2_flat = Rotation.from_quat(quat_wxyz_to_xyzw(target_c2_on_c1_quat_w_flat))
        # EEF pos = cube_target_pos + R_world_cube * t_eef_in_cube_frame
        stack_c2_eef_pos_w = target_c2_on_c1_pos_w + R_w_target_c2_flat.apply(t_cube_eef_in_cube_at_grasp)
        # EEF quat = R_world_cube * R_eef_in_cube_frame
        stack_c2_eef_quat_w = quat_xyzw_to_wxyz((R_w_target_c2_flat * R_cube_eef_at_grasp).as_quat())
        add_waypoint(stack_c2_eef_pos_w, stack_c2_eef_quat_w, True)
        # 1.6 Release Cube 2
        add_waypoint(stack_c2_eef_pos_w, stack_c2_eef_quat_w, False)
        # 1.7 Lift from Cube 2 with two times of CUBE_HEIGHT
        lift_from_c2_pos_w = stack_c2_eef_pos_w + np.array([0,0, 1.0 * CUBE_HEIGHT])
        add_waypoint(lift_from_c2_pos_w, stack_c2_eef_quat_w, False)

        # --- Process Cube 3 (grasp and place on Cube 2+1) ---
        # 2.1 Pre-grasp Cube 3 (approach based on its *current* orientation)
        pre_grasp_c3_pos_w, pre_grasp_c3_quat_w = self._calculate_eef_world_pose_from_cube_relative(
            cube3_pos_w, cube3_quat_wxyz_w, CUBE_STACK_PRE_GRASP_OFFSET_POS_CUBE_FRAME, CUBE_STACK_PRE_GRASP_EULER_XYZ_DEG_CUBE_FRAME)
        add_waypoint(pre_grasp_c3_pos_w, pre_grasp_c3_quat_w, False)
        # 2.2 Approach Cube 3 in z-axis (move down in world Z)
        grasp_c3_pos_w = pre_grasp_c3_pos_w - np.array([0,0, CUBE_STACK_GRASP_APPROACH_DISTANCE_Z_WORLD])
        add_waypoint(grasp_c3_pos_w, pre_grasp_c3_quat_w, False)
        # 2.3 Grasp Cube 3
        add_waypoint(grasp_c3_pos_w, pre_grasp_c3_quat_w, True)
        # 2.4 Intermediate to Cube 1 (+2)
        intermediate_c1_pos_w = cube1_pos_w * 1 / 5 + cube3_pos_w * 4 / 5  # Weighted average towards Cube 3
        intermediate_c1_pos_w[2] = cube1_pos_w[2] + 4.5 * CUBE_HEIGHT  # Relative to Cube1's base
        # Calculate intermediate orientation for Cube 3 placement
        cube1_yaw_rad_for_c3_stack = Rotation.from_quat(quat_wxyz_to_xyzw(cube1_quat_wxyz_w)).as_euler('zyx')[0]
        target_c3_on_c2_final_eef_quat_w = self._calculate_final_eef_orientation_for_stack(cube1_quat_wxyz_w, cube1_yaw_rad_for_c3_stack, R_cube_eef_at_grasp)
        intermediate_c2_actual_orient_slerp = Slerp([0, 1], Rotation.from_quat(quat_wxyz_to_xyzw(np.array([pre_grasp_c3_quat_w, target_c3_on_c2_final_eef_quat_w]))))
        intermediate_c2_actual_quat_w = quat_xyzw_to_wxyz(intermediate_c2_actual_orient_slerp(0.5).as_quat()) # Midpoint orientation
        add_waypoint(intermediate_c1_pos_w, intermediate_c2_actual_quat_w, True)
        # 2.5 Stack Cube 3 on Cube 2 (+1)
        target_c3_on_c2_pos_w = cube1_pos_w + np.array([0,0, 2 * CUBE_STACK_ON_CUBE_Z_OFFSET])
        # Flatten actual_cube2_quat_w_stacked_flat to get target yaw for Cube3. It's already flat, but this ensures consistency.
        cube2_stacked_yaw_rad = Rotation.from_quat(quat_wxyz_to_xyzw(cube1_quat_wxyz_w)).as_euler('zyx')[0]
        target_c3_on_c2_quat_w_flat = self._flatten_quat_around_world_z(cube1_quat_wxyz_w, target_yaw_rad=cube2_stacked_yaw_rad)

        R_w_target_c3_flat = Rotation.from_quat(quat_wxyz_to_xyzw(target_c3_on_c2_quat_w_flat))
        stack_c3_eef_pos_w = target_c3_on_c2_pos_w + R_w_target_c3_flat.apply(t_cube_eef_in_cube_at_grasp)
        stack_c3_eef_quat_w = quat_xyzw_to_wxyz((R_w_target_c3_flat * R_cube_eef_at_grasp).as_quat())
        add_waypoint(stack_c3_eef_pos_w, stack_c3_eef_quat_w, True)
        # 2.6 Release Cube 3
        add_waypoint(stack_c3_eef_pos_w, stack_c3_eef_quat_w, False)
        # 2.7 Lift from Cube 3
        lift_from_c3_pos_w = stack_c3_eef_pos_w + np.array([0,0, 0.5 * CUBE_HEIGHT]) # 1.0 -> 0.5 to avoid IK issues
        add_waypoint(lift_from_c3_pos_w, stack_c3_eef_quat_w, False)

        # --- Final: Return to initial right arm pose ---
        add_waypoint(self.initial_right_arm_pos_w, self.initial_right_arm_quat_wxyz_w, False)
        
        # print(f"  Generated {len(self.recorded_waypoints)} waypoints for auto cube stacking.")
        # print("  --- Generated Waypoint Details ---")
        # for i, wp in enumerate(self.recorded_waypoints):
        #     print(f"  Waypoint {i}:")
        #     print(f"    Right Arm EEF: Pos={wp['right_arm_eef'][:3]}, Quat={wp['right_arm_eef'][3:7]}, GripperOpen={not wp['right_hand_bool']}")
        # print("--- End of Cube Stacking Trajectory Generation ---\n")

    def _calculate_final_eef_orientation_for_stack(self,
                                                 base_cube_quat_w: np.ndarray,
                                                 target_stacked_cube_yaw_rad: float,
                                                 R_cube_eef_at_grasp: Rotation) -> np.ndarray:
        """Calculates the EEF's world orientation when a cube it's holding is stacked flatly."""
        target_stacked_cube_quat_w_flat = self._flatten_quat_around_world_z(base_cube_quat_w, target_yaw_rad=target_stacked_cube_yaw_rad)
        R_w_target_stacked_cube_flat = Rotation.from_quat(quat_wxyz_to_xyzw(target_stacked_cube_quat_w_flat))
        # EEF quat = R_world_cube * R_eef_in_cube_frame
        final_eef_quat_w = quat_xyzw_to_wxyz((R_w_target_stacked_cube_flat * R_cube_eef_at_grasp).as_quat())
        return final_eef_quat_w


    def _add_waypoint(self, right_eef_pos, right_eef_quat, right_hand_closed_bool, left_eef_pos, left_eef_quat, left_hand_closed_bool):
        """Helper to append a waypoint to the recorded_waypoints list."""
        wp = {
            "left_arm_eef": np.concatenate([left_eef_pos, left_eef_quat]),
            "right_arm_eef": np.concatenate([right_eef_pos, right_eef_quat]),
            "left_hand_bool": int(left_hand_closed_bool),
            "right_hand_bool": int(right_hand_closed_bool)
        }
        self.recorded_waypoints.append(wp)

    def generate_open_drawer_sub_trajectory(self, obs: dict):
        """Generates a trajectory to open the drawer."""
        self.clear_waypoints()
        (initial_left_pos, initial_left_quat, current_right_eef_pos_w, current_right_eef_quat_wxyz_w,
         *_, drawer_pos, drawer_quat, _, _, _, _, _, _) = self.extract_essential_obs_data(obs)

        # 1.0. Add the current right/left EEF position and orientation as the first waypoint
        self._add_waypoint(current_right_eef_pos_w, current_right_eef_quat_wxyz_w, False, initial_left_pos, initial_left_quat, False)

        R_world_drawer = Rotation.from_quat(quat_wxyz_to_xyzw(drawer_quat))
        # 1.1. Move to a prepared position before opening the handle
        prepare_left_pos, prepare_left_quat = np.array([0.075, 0.220, 0.950]), np.array([1.0, 0.0, 0.0, 0.0])  # TODO: Set it as constants
        pre_approach_handle_pos = drawer_pos + R_world_drawer.apply(PRE_APPROACH_OFFSET_POS)
        R_world_gripper = R_world_drawer * Rotation.from_euler('xyz', PRE_APPROACH_OFFSET_QUAT, degrees=True)
        approach_handle_quat = quat_xyzw_to_wxyz(R_world_gripper.as_quat())
        self._add_waypoint(pre_approach_handle_pos, approach_handle_quat, False, prepare_left_pos, prepare_left_quat, False)

        # 1.2. Approach the drawer handle
        approach_handle_pos = drawer_pos + R_world_drawer.apply(APPROACH_OFFSET_POS)
        self._add_waypoint(approach_handle_pos, approach_handle_quat, False, prepare_left_pos, prepare_left_quat, False)
        # 1.3. Grasp the drawer handle 
        self._add_waypoint(approach_handle_pos, approach_handle_quat, True, prepare_left_pos, prepare_left_quat, False)

        # 1.4. Pull out the drawer handle
        pulled_handle_pos = drawer_pos + R_world_drawer.apply(PULL_OFFSET_POS)
        self._add_waypoint(pulled_handle_pos, approach_handle_quat, True, prepare_left_pos, prepare_left_quat, False)

        # # Debugging output
        # print("[TrajectoryPlayer] Drawer open sub-trajectory generated with waypoints:")
        # print(f"[INFO] Drawer Position: {drawer_pos}, Quat: {drawer_quat}")
        # for i, wp in enumerate(self.recorded_waypoints):
        #     print(f"  Waypoint {i}:")  
        #     print(f"    Right Arm EEF: Pos={wp['right_arm_eef'][:3]}, Quat={wp['right_arm_eef'][3:7]}, GripperOpen={not wp['right_hand_bool']}")


    def generate_pick_and_place_mug_sub_trajectory(self, obs: dict):
        """Generates a trajectory to pick the mug, place it on the mat, and close the drawer."""
        self.clear_waypoints()
        (current_left_pos, current_left_quat, current_right_eef_pos_w, current_right_eef_quat_wxyz_w,
         *_, drawer_pos, drawer_quat, _, _, mug_pos, mug_quat, mug_mat_pos, mug_mat_quat) = self.extract_essential_obs_data(obs)

        # 2.0. Add the current right/left EEF position and orientation as the first waypoint
        self._add_waypoint(current_right_eef_pos_w, current_right_eef_quat_wxyz_w, False, current_left_pos, current_left_quat, False)


        # 2.1. Move the left EEF to a prepared position - with respect to the mug (Hands open)
        R_world_mug = Rotation.from_quat(quat_wxyz_to_xyzw(mug_quat))
        pre_grasp_mug_pos = mug_pos + MUG_PRE_GRASP_POS
        R_world_gripper = R_world_mug * Rotation.from_euler('xyz', MUG_GRASP_QUAT, degrees=True)
        grasp_mug_quat = quat_xyzw_to_wxyz(R_world_gripper.as_quat())
        self._add_waypoint(current_right_eef_pos_w, current_right_eef_quat_wxyz_w, False, pre_grasp_mug_pos, grasp_mug_quat, False)

        # 2.2. Approach the mug (Hands open)
        approach_mug_pos = mug_pos + MUG_APPROACH_POS
        self._add_waypoint(current_right_eef_pos_w, current_right_eef_quat_wxyz_w, False, approach_mug_pos, grasp_mug_quat, False)
        # 2.3. Grasp the mug (close the left hand)
        self._add_waypoint(current_right_eef_pos_w, current_right_eef_quat_wxyz_w, False, approach_mug_pos, grasp_mug_quat, True)

        # 2.4. Lift the mug away the drawer (keep the left hand closed)
        lift_mug_pos = approach_mug_pos + MUG_LIFT_POS
        self._add_waypoint(current_right_eef_pos_w, current_right_eef_quat_wxyz_w, False, lift_mug_pos, grasp_mug_quat, True)

        
        # 2.5. Approach the mug mat - with respect to the mat (Hands closed)
        R_world_mat = Rotation.from_quat(quat_wxyz_to_xyzw(mug_mat_quat))
        place_mug_on_mat_pos = mug_mat_pos + MAT_PLACE_POS
        R_world_gripper_target = R_world_mat * Rotation.from_euler('xyz', MAT_PLACE_QUAT, degrees=True)
        place_mug_quat = quat_xyzw_to_wxyz(R_world_gripper_target.as_quat())
        self._add_waypoint(current_right_eef_pos_w, current_right_eef_quat_wxyz_w, False, place_mug_on_mat_pos, place_mug_quat, True)

        # 2.6. Place the mug on the mug mat (Open the left hand)
        self._add_waypoint(current_right_eef_pos_w, current_right_eef_quat_wxyz_w, False, place_mug_on_mat_pos, place_mug_quat, False)

        # 2.7. Push back the opened drawer (right EEF), and lift the left EEF away from the mug
        push_approach_pos = current_right_eef_pos_w + DRAWER_PUSH_DIRECTION_LOCAL   # Right EEF
        mat_retract_pos = mug_mat_pos + MAT_RETRACT_POS  # Left EEF
        self._add_waypoint(push_approach_pos, current_right_eef_quat_wxyz_w, False, mat_retract_pos, place_mug_quat, False)

        # 2.8. Leave the right EEF away the drawer, restore the left EEF to the original pose (fixed, given poses)
        right_retract_pos, right_retract_quat = np.array([0.075, -0.205, 0.90]), [0.7329629, 0.5624222, 0.3036032, -0.2329629] # TODO: Set it as constants
        left_retract_pos, left_retract_quat = np.array([0.075, 0.22108203, 0.950]), [1.0, 0.0, 0.0, 0.0]
        self._add_waypoint(right_retract_pos, right_retract_quat, False, left_retract_pos, left_retract_quat, False)

        # 2.9. Restore the right EEF to a middle waypoint
        right_restore_pos, right_restore_quat = np.array([0.060, -0.340, 0.90]), np.array([0.9848078, 0.0, 0.0, -0.1736482]) # TODO: Set it as constants
        self._add_waypoint(right_restore_pos, right_restore_quat, False, left_retract_pos, left_retract_quat, False)
        
        # Debugging output
        print("[TrajectoryPlayer] pick-and-place mug sub-trajectory generated with waypoints:")
        print(f"[INFO] Drawer Position: {drawer_pos}, Quat: {drawer_quat}")
        print(f"[INFO] Mug Position: {mug_pos}, Quat: {mug_quat}")
        print(f"[INFO] Mug Mat Position: {mug_mat_pos}, Quat: {mug_mat_quat}")
        for i, wp in enumerate(self.recorded_waypoints):
            print(f"  Waypoint {i}: Left Arm EEF: Pos={wp['left_arm_eef'][:3]}, Quat={wp['left_arm_eef'][3:7]}, GripperOpen={not wp['left_hand_bool']}")

    def generate_pour_bottle_sub_trajectory(self, obs: dict):
        """Generates a trajectory to pour the bottle into the mug."""
        self.clear_waypoints()
        (current_left_pos, current_left_quat, current_right_eef_pos_w, current_right_eef_quat_wxyz_w,
         *_, bottle_pos, bottle_quat, mug_pos, mug_quat, _, _) = self.extract_essential_obs_data(obs)

        # Start from the current pose
        self._add_waypoint(current_right_eef_pos_w, current_right_eef_quat_wxyz_w, False, current_left_pos, current_left_quat, False)

        # 3.1. Approach the bottle - with respect to the bottle (Hands open)
        grasp_bottle_pos = bottle_pos + BOTTLE_GRASP_POS
        grasp_bottle_quat = quat_xyzw_to_wxyz((Rotation.from_quat(quat_wxyz_to_xyzw(bottle_quat)) * Rotation.from_euler('xyz', BOTTLE_GRASP_QUAT, degrees=True)).as_quat())
        self._add_waypoint(grasp_bottle_pos, grasp_bottle_quat, False, current_left_pos, current_left_quat, False)

        # 3.2. Grasp the bottle (Close the right hand)
        self._add_waypoint(grasp_bottle_pos, grasp_bottle_quat, True, current_left_pos, current_left_quat, False)

        # 3.3. Lift up the bottle
        lift_bottle_pos = grasp_bottle_pos + BOTTLE_LIFT_UP_OFFSET
        self._add_waypoint(lift_bottle_pos, grasp_bottle_quat, True, current_left_pos, current_left_quat, False)

        # 3.4. Move the bottle toward the mug for pouring
        pre_pour_pos = mug_pos + BOTTLE_PRE_POUR_OFFSET
        self._add_waypoint(pre_pour_pos, grasp_bottle_quat, True, current_left_pos, current_left_quat, False)

        # 3.5. Pour the bottle with -100 degree rotation in the Y-axis
        pouring_pos = mug_pos + BOTTLE_POURING_OFFSET
        pouring_rot = Rotation.from_quat(quat_wxyz_to_xyzw(grasp_bottle_quat)) * Rotation.from_euler('x', BOTTLE_POUR_ANGLE_DEG, degrees=True)
        pouring_quat = quat_xyzw_to_wxyz(pouring_rot.as_quat())
        self._add_waypoint(pouring_pos, pouring_quat, True, current_left_pos, current_left_quat, False)

        # 3.6. Restore the bottle to the vertical pose
        self._add_waypoint(pre_pour_pos, grasp_bottle_quat, True, current_left_pos, current_left_quat, False)

        # 3.7. Shift the bottle back to the original position with 5-cm height
        self._add_waypoint(lift_bottle_pos, grasp_bottle_quat, True, current_left_pos, current_left_quat, False)

        # 3.8. Place the bottle back to the original position
        self._add_waypoint(grasp_bottle_pos, grasp_bottle_quat, True, current_left_pos, current_left_quat, False)

        # 3.9. Release the bottle
        self._add_waypoint(grasp_bottle_pos, grasp_bottle_quat, False, current_left_pos, current_left_quat, False)

        # Return home # TODO: Back to the initial poses
        self._add_waypoint(self.initial_right_arm_pos_w, self.initial_right_arm_quat_wxyz_w, False, self.initial_left_arm_pos_w, self.initial_left_arm_quat_wxyz_w, False)
        
        # Debugging output
        print("[TrajectoryPlayer] pour the bottle into the mug sub-trajectory generated with waypoints:")
        print(f"[INFO] bottle Position: {bottle_pos}, Quat: {bottle_quat}")
        print(f"[INFO] Mug Position: {mug_pos}, Quat: {mug_quat}")
        for i, wp in enumerate(self.recorded_waypoints):
            print(f"  Waypoint {i}: Left Arm EEF: Pos={wp['left_arm_eef'][:3]}, Quat={wp['left_arm_eef'][3:7]}, GripperOpen={not wp['left_hand_bool']}")


    def generate_open_drawer_trajectory(self, obs: dict, filepath="scripts/gr00t_script/configs/open_drawer_waypoints.yaml"):
        """
        Generates a trajectory for opening a drawer by loading waypoints from a YAML file.
        It prepends the current robot pose to ensure a smooth transition.
        """
        self.clear_waypoints()

        # Get the current EEF pose to start the trajectory smoothly
        (current_left_eef_pos_w, current_left_eef_quat_wxyz_w,
         current_right_eef_pos_w, current_right_eef_quat_wxyz_w,
         *_) = self.extract_essential_obs_data(obs)

        # Waypoint 0: Current pose
        start_waypoint = {
            "left_arm_eef": np.concatenate([current_left_eef_pos_w, current_left_eef_quat_wxyz_w]),
            "right_arm_eef": np.concatenate([current_right_eef_pos_w, current_right_eef_quat_wxyz_w]),
            "left_hand_bool": 0, # Assume hands are open at the start
            "right_hand_bool": 0
        }
        self.recorded_waypoints.append(start_waypoint)
        
        try:
            with open(filepath, 'r') as f:
                loaded_wps_list = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Waypoint file {filepath} not found. No waypoints loaded.")
            return
        except yaml.YAMLError as e:
            print(f"Error decoding YAML from {filepath}: {e}")
            return
        except Exception as e:
            print(f"Error loading waypoints from {filepath}: {e}")
            import traceback
            traceback.print_exc()
            return

        for wp_dict in loaded_wps_list:
            self.recorded_waypoints.append({
                "left_arm_eef": np.array(wp_dict["left_arm_eef"]),
                "right_arm_eef": np.array(wp_dict["right_arm_eef"]),
                "left_hand_bool": int(wp_dict["left_hand_bool"]),
                "right_hand_bool": int(wp_dict["right_hand_bool"])
            })
        
        print(f"Loaded {len(loaded_wps_list)} waypoints from {filepath} and prepended current pose.")
