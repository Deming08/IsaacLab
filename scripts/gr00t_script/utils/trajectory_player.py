# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utility class for recording and playing back end-effector trajectories."""

import json
import os

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


# GraspPoseCalculator is now a direct runtime dependency
from .grasp_pose_calculator import GraspPoseCalculator
from .quaternion_utils import quat_xyzw_to_wxyz, quat_wxyz_to_xyzw


# === Constants for G1 Trajectory Generation ===
DEFAULT_LEFT_HAND_BOOL = False  # False for open

# Constants for red and blue basket pose
CAN_RADIUS = 0.025 #
RED_BASKET_CENTER = np.array([0.45 + CAN_RADIUS, 0.0 - CAN_RADIUS, 0.81]) # 
BLUE_BASKET_CENTER = np.array([0.45 + CAN_RADIUS, -0.12 - CAN_RADIUS, 0.81])
        
# Define placement orientations for baskets (e.g., 90-degree yaw)
RED_BASKET_PLACEMENT_YAW_DEGREES = 10.0
RED_BASKET_PLACEMENT_QUAT_WXYZ = quat_xyzw_to_wxyz(Rotation.from_euler('z', RED_BASKET_PLACEMENT_YAW_DEGREES, degrees=True).as_quat())

BLUE_BASKET_PLACEMENT_YAW_DEGREES = -20.0
BLUE_BASKET_PLACEMENT_QUAT_WXYZ = quat_xyzw_to_wxyz(Rotation.from_euler('z', BLUE_BASKET_PLACEMENT_YAW_DEGREES, degrees=True).as_quat())

# Default paths for saving waypoints and joint tracking logs
WAYPOINTS_JSON_PATH = os.path.join("logs", "teleoperation", "waypoints.json")
JOINT_TRACKING_LOG_PATH = os.path.join("logs", "teleoperation", "joint_tracking_log.json")


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
         target_object_pos_w, target_object_quat_wxyz_w, target_object_color_id) = self.extract_essential_obs_data(initial_obs)
        print(f"[INFO] TrajectoryPlayer using Left Arm Pos: {self.initial_left_arm_pos_w}, Quat: {self.initial_left_arm_quat_wxyz_w}")
        print(f"[INFO] TrajectoryPlayer using Right Arm Pos: {self.initial_right_arm_pos_w}, Quat: {self.initial_right_arm_quat_wxyz_w}")
        print(f"[INFO] Target Object Pose: {target_object_pos_w}, Quat: {target_object_quat_wxyz_w}, Color: {'red can' if target_object_color_id == 0 else 'blue can'}")

        # {"left_arm_eef"(7), "right_arm_eef"(7), "left_hand", "right_hand"}
        self.recorded_waypoints = []
        # {"palm_position": np.array, "palm_orientation_wxyz": np.array, "hand_joint_positions": np.array}
        self.playback_trajectory_actions = []
        self.current_playback_idx = 0
        self.is_playing_back = False
        self.grasp_calculator = GraspPoseCalculator() # Instantiate GraspPoseCalculator
        self.steps_per_movement_segment = steps_per_movement_segment
        self.steps_per_grasp_segment = steps_per_grasp_segment

        # Get hand joint names from the action manager
        self.pink_hand_joint_names = self.env.action_manager._terms["pink_ik_cfg"].cfg.hand_joint_names
        # print("[TrajectoryPlayer] Initialized with hand joint names:", self.pink_hand_joint_names)
        # ['left_hand_index_0_joint', 'left_hand_middle_0_joint', 'left_hand_thumb_0_joint', 'right_hand_index_0_joint', 'right_hand_middle_0_joint', 'right_hand_thumb_0_joint', 'left_hand_index_1_joint', 'left_hand_middle_1_joint', 'left_hand_thumb_1_joint', 'right_hand_index_1_joint', 'right_hand_middle_1_joint', 'right_hand_thumb_1_joint', 'left_hand_thumb_2_joint', 'right_hand_thumb_2_joint']
        
        # Joint tracking data
        self.joint_tracking_records = []
        self.joint_tracking_active = False

    def extract_essential_obs_data(self, obs: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        """Helper to extract common observation data from the first environment."""
        left_eef_pos = obs["policy"]["left_eef_pos"][0].cpu().numpy()
        left_eef_quat = obs["policy"]["left_eef_quat"][0].cpu().numpy()
        right_eef_pos = obs["policy"]["right_eef_pos"][0].cpu().numpy()
        right_eef_quat = obs["policy"]["right_eef_quat"][0].cpu().numpy()

        target_object_obs_tensor = obs["policy"]["target_object_pose"][0].cpu().numpy()
        target_object_pos = target_object_obs_tensor[:3]
        target_object_quat = target_object_obs_tensor[3:7]
        target_object_color_id = int(target_object_obs_tensor[13])

        return (left_eef_pos, left_eef_quat, right_eef_pos, right_eef_quat,
                target_object_pos, target_object_quat, target_object_color_id)

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
    
    def record_current_pose(self, obs: dict, teleop_output=None):
        """
        Record the current end-effector link pose and orientation for both right and left, and gripper bools.
        Concatenate [right_arm_eef_pos, right_arm_eef_orient_wxyz, right_hand_bool, left_arm_eef_pos, left_arm_eef_orient_wxyz, left_hand_bool].
        """
        # Get the end-effector link pose and orientation using the helper
        (left_arm_eef_pos, left_arm_eef_orient_wxyz, right_arm_eef_pos, right_arm_eef_orient_wxyz, _, _, _,) = self.extract_essential_obs_data(obs)

        # Extract right gripper command from teleop_output
        right_gripper_bool = 0
        if teleop_output is not None and isinstance(teleop_output, (tuple, list)) and len(teleop_output) > 1:
            right_gripper_bool = int(teleop_output[1])

        # Store as structured dict per user request
        waypoint = {
            "left_arm_eef": np.concatenate([left_arm_eef_pos.flatten(), left_arm_eef_orient_wxyz.flatten()]),
            "right_arm_eef": np.concatenate([right_arm_eef_pos.flatten(), right_arm_eef_orient_wxyz.flatten()]),
            "left_hand_bool": int(DEFAULT_LEFT_HAND_BOOL),
            "right_hand_bool": int(right_gripper_bool)
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
        else:
            print("Waypoints cleared.")


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
            # Create time points for interpolation within the segment
            # This logic is specific to the 7-waypoint trajectory generated by generate_auto_grasp_pick_place_trajectory where:
            #   segment i=1 (waypoint 2 -> waypoint 3) is a grasp action.
            #   segment i=4 (waypoint 5 -> waypoint 6) is a release action.
            if len(self.recorded_waypoints) == 7 and (i == 1 or i == 4):
                num_points_in_segment = self.steps_per_grasp_segment
            else:
                num_points_in_segment = self.steps_per_movement_segment

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
                self.save_joint_tracking_data()  # Save data when playback ends
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
        # Define joint positions for open and closed states
        joint_positions = {
            "left_hand_index_0_joint":   {"open": 0.0, "closed": 0.8},
            "left_hand_middle_0_joint":  {"open": 0.0, "closed": 0.8},
            "left_hand_thumb_0_joint":   {"open": 0.0, "closed": 0.0},
            "right_hand_index_0_joint":  {"open": 0.0, "closed": 0.8},
            "right_hand_middle_0_joint": {"open": 0.0, "closed": 0.8},
            "right_hand_thumb_0_joint":  {"open": 0.0, "closed": 0.0},
            "left_hand_index_1_joint":   {"open": 0.0, "closed": 0.8},
            "left_hand_middle_1_joint":  {"open": 0.0, "closed": 0.8},
            "left_hand_thumb_1_joint":   {"open": 0.0, "closed": 0.8},
            "right_hand_index_1_joint":  {"open": 0.0, "closed": 0.8},
            "right_hand_middle_1_joint": {"open": 0.0, "closed": 0.8},
            "right_hand_thumb_1_joint":  {"open": 0.0, "closed": -0.8},
            "left_hand_thumb_2_joint":   {"open": 0.0, "closed": 0.8},
            "right_hand_thumb_2_joint":  {"open": 0.0, "closed": -0.8},
        }

        hand_joint_positions = np.zeros(len(self.pink_hand_joint_names))
        for idx, joint_name in enumerate(self.pink_hand_joint_names):
            if "right" in joint_name:
                hand_joint_positions[idx] = joint_positions[joint_name]["closed"] if right_hand_bool else joint_positions[joint_name]["open"]
            elif "left" in joint_name:
                hand_joint_positions[idx] = joint_positions[joint_name]["closed"] if left_hand_bool else joint_positions[joint_name]["open"]
        return hand_joint_positions

    def generate_auto_grasp_pick_place_trajectory(self, obs: dict):
        """
        Generates a predefined 7-waypoint trajectory for grasping a cube and placing it.

        Args:
            obs: The observation dictionary from the environment, containing current robot and object states.
        """
        (_, _, current_right_eef_pos_w, current_right_eef_quat_wxyz_w, target_object_pos_w, target_object_quat_wxyz_w, target_object_color_id) = self.extract_essential_obs_data(obs)
        print(f"Current Right EEF Pose: pos={current_right_eef_pos_w}, quat_wxyz={current_right_eef_quat_wxyz_w}")
        print(f"Target Object Pose: pos={target_object_pos_w}, quat_wxyz={target_object_quat_wxyz_w}, color= {'red can' if target_object_color_id == 0 else 'blue can'}")
        
        self.clear_waypoints()
        # 1. Calculate target grasp pose for the right EEF
        target_grasp_right_eef_pos_w, target_grasp_right_eef_quat_wxyz_w = \
            self.grasp_calculator.calculate_target_ee_pose(target_object_pos_w, target_object_quat_wxyz_w)
        print(f"Calculated Target Grasp Right EEF Pose: pos={target_grasp_right_eef_pos_w}, quat_wxyz={target_grasp_right_eef_quat_wxyz_w}")

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
        if target_object_color_id == 0: # Red Can
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

        print(f"Generated {len(self.recorded_waypoints)} waypoints for auto grasp and place.")
