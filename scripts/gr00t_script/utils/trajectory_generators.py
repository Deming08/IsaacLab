# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This module contains classes for generating predefined trajectories for various robotic tasks."""

import os
import yaml
import json
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation, Slerp # type: ignore

from .constants import *
from .grasp_pose_calculator import GraspPoseCalculator
from .quaternion_utils import quat_xyzw_to_wxyz, quat_wxyz_to_xyzw
from .trajectory_player import TrajectoryPlayer # For static method access
from .skills import OpenDrawerSkill, PickMugFromDrawerSkill, PlaceMugOnMatSkill, PourBottleSkill, ReturnBottleSkill

class BaseTrajectoryGenerator:
    """Base class for trajectory generators to provide a common interface and helpers."""

    def __init__(self, obs: dict):
        """
        Initializes the base trajectory generator.
        Args:
            obs: The initial observation dictionary from the environment.
        """
        self.obs = obs
        self.waypoints = []

    def generate(self, *args, **kwargs) -> list:
        """Generates and returns a list of waypoints. To be implemented by subclasses."""
        raise NotImplementedError

    def _add_waypoint(self, right_eef_pos, right_eef_quat, right_hand_closed_bool, left_eef_pos, left_eef_quat, left_hand_closed_bool):
        """Helper to append a waypoint to the recorded_waypoints list."""
        wp = {
            "left_arm_eef": np.concatenate([left_eef_pos, left_eef_quat]),
            "right_arm_eef": np.concatenate([right_eef_pos, right_eef_quat]),
            "left_hand_bool": int(left_hand_closed_bool),
            "right_hand_bool": int(right_hand_closed_bool)
        }
        self.waypoints.append(wp)

class GraspPickPlaceTrajectoryGenerator(BaseTrajectoryGenerator):
    """Generates a predefined 7-waypoint trajectory for grasping a can and placing it in a basket."""

    def __init__(self, obs: dict, initial_poses: dict):
        """
        Args:
            obs: The observation dictionary from the environment.
            initial_poses: A dict with initial poses, e.g., {'left_pos': ..., 'left_quat': ...}.
        """
        super().__init__(obs)
        self.grasp_calculator = GraspPoseCalculator()
        self.initial_left_arm_pos_w = initial_poses["left_pos"]
        self.initial_left_arm_quat_wxyz_w = initial_poses["left_quat"]

    def generate(self) -> list:
        """The main generation logic for the pick-and-place task."""
        self.waypoints = [] # Clear previous waypoints
        (_, _, current_right_eef_pos_w, current_right_eef_quat_wxyz_w,
         _, _, _, _, _, _, # Cube poses not used by this function
         target_can_pos_w, target_can_quat_wxyz_w, target_can_color_id,
         *_) = TrajectoryPlayer.extract_essential_obs_data(self.obs)
        print(f"Target Can Pose: pos={target_can_pos_w}, quat_wxyz={target_can_quat_wxyz_w}, color= {'red can' if target_can_color_id == 0 else 'blue can'}")

        # 1. Calculate target grasp pose for the right EEF
        target_grasp_right_eef_pos_w, target_grasp_right_eef_quat_wxyz_w = \
            self.grasp_calculator.calculate_target_ee_pose(target_can_pos_w, target_can_quat_wxyz_w)

        # Waypoint 1: Current EEF pose (right hand open)
        wp1_left_arm_eef = np.concatenate([self.initial_left_arm_pos_w, self.initial_left_arm_quat_wxyz_w])
        wp1_right_arm_eef = np.concatenate([current_right_eef_pos_w, current_right_eef_quat_wxyz_w])
        waypoint1 = {"left_arm_eef": wp1_left_arm_eef, "right_arm_eef": wp1_right_arm_eef, "left_hand_bool": int(DEFAULT_LEFT_HAND_BOOL), "right_hand_bool": 0}
        self.waypoints.append(waypoint1)

        # Waypoint 2: Target grasp EEF pose (right hand open - pre-grasp)
        wp2_left_arm_eef = np.concatenate([self.initial_left_arm_pos_w, self.initial_left_arm_quat_wxyz_w])
        wp2_right_arm_eef = np.concatenate([target_grasp_right_eef_pos_w, target_grasp_right_eef_quat_wxyz_w])
        waypoint2 = {"left_arm_eef": wp2_left_arm_eef, "right_arm_eef": wp2_right_arm_eef, "left_hand_bool": int(DEFAULT_LEFT_HAND_BOOL), "right_hand_bool": 0}
        self.waypoints.append(waypoint2)

        # Waypoint 3: Target grasp EEF pose (right hand closed - grasp)
        waypoint3 = {**waypoint2, "right_hand_bool": 1}
        self.waypoints.append(waypoint3)

        # Determine placement pose based on target object color
        if target_can_color_id == 0: # Red Can
            basket_base_target_pos_w = RED_BASKET_CENTER
            basket_target_quat_wxyz = RED_BASKET_PLACEMENT_QUAT_WXYZ
        else: # Blue Can
            basket_base_target_pos_w = BLUE_BASKET_CENTER
            basket_target_quat_wxyz = BLUE_BASKET_PLACEMENT_QUAT_WXYZ
        
        placement_target_pos_w = np.array([basket_base_target_pos_w[0], basket_base_target_pos_w[1], target_grasp_right_eef_pos_w[2] - 0.06])

        # Waypoint 4: Intermediate lift pose
        lift_pos_w = np.array([(target_grasp_right_eef_pos_w[0] + placement_target_pos_w[0]) / 2, (target_grasp_right_eef_pos_w[1] + placement_target_pos_w[1]) / 2, max(target_grasp_right_eef_pos_w[2], placement_target_pos_w[2]) + 0.10])
        key_rots = Rotation.from_quat([target_grasp_right_eef_quat_wxyz_w[[1,2,3,0]], basket_target_quat_wxyz[[1,2,3,0]]])
        lift_quat_wxyz_w = Slerp([0, 1], key_rots)(0.5).as_quat()[[3,0,1,2]]
        waypoint4 = {"left_arm_eef": wp2_left_arm_eef, "right_arm_eef": np.concatenate([lift_pos_w, lift_quat_wxyz_w]), "left_hand_bool": int(DEFAULT_LEFT_HAND_BOOL), "right_hand_bool": 1}
        self.waypoints.append(waypoint4)

        # Waypoint 5: Move right arm EEF to basket placement pose (hand closed)
        waypoint5 = {"left_arm_eef": wp2_left_arm_eef, "right_arm_eef": np.concatenate([placement_target_pos_w, basket_target_quat_wxyz]), "left_hand_bool": int(DEFAULT_LEFT_HAND_BOOL), "right_hand_bool": 1}
        self.waypoints.append(waypoint5)

        # Waypoint 6: At RED_PLATE pose, open right hand
        waypoint6 = {**waypoint5, "right_hand_bool": 0}
        self.waypoints.append(waypoint6)

        # Waypoint 7: Move the right arm to the initial pose (hand open)
        waypoint7 = {**waypoint1, "right_hand_bool": 0}
        self.waypoints.append(waypoint7)

        return self.waypoints

class StackCubesTrajectoryGenerator(BaseTrajectoryGenerator):
    """Generates the trajectory for stacking three cubes."""

    def __init__(self, obs: dict, initial_poses: dict):
        super().__init__(obs)
        self.initial_right_arm_pos_w = initial_poses["right_pos"]
        self.initial_right_arm_quat_wxyz_w = initial_poses["right_quat"]
        self.initial_left_arm_pos_w = initial_poses["left_pos"]
        self.initial_left_arm_quat_wxyz_w = initial_poses["left_quat"]

    def _calculate_eef_world_pose_from_cube_relative(self, cube_pos_w: np.ndarray, cube_quat_wxyz_w: np.ndarray, eef_offset_pos_cube_frame: np.ndarray, eef_euler_xyz_deg_cube_frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Helper to calculate EEF world pose given cube world pose and EEF's relative pose to the cube."""
        R_w_cube = Rotation.from_quat(quat_wxyz_to_xyzw(cube_quat_wxyz_w))
        R_cube_eef_relative = Rotation.from_euler('xyz', eef_euler_xyz_deg_cube_frame, degrees=True)
        eef_target_pos_w = cube_pos_w + R_w_cube.apply(eef_offset_pos_cube_frame)
        R_w_eef_target = R_w_cube * R_cube_eef_relative
        eef_target_quat_wxyz_w = quat_xyzw_to_wxyz(R_w_eef_target.as_quat())
        return eef_target_pos_w, eef_target_quat_wxyz_w

    def _flatten_quat_around_world_z(self, quat_wxyz: np.ndarray, target_yaw_rad: Optional[float] = None) -> np.ndarray:
        """Flattens a quaternion to have zero roll and pitch in the world frame."""
        r = Rotation.from_quat(quat_wxyz_to_xyzw(quat_wxyz))
        euler_zyx = r.as_euler('zyx', degrees=False)
        if target_yaw_rad is not None:
            euler_zyx[0] = target_yaw_rad
        euler_zyx[1] = 0.0
        euler_zyx[2] = 0.0
        return quat_xyzw_to_wxyz(Rotation.from_euler('zyx', euler_zyx, degrees=False).as_quat())

    def _calculate_final_eef_orientation_for_stack(self, base_cube_quat_w: np.ndarray, target_stacked_cube_yaw_rad: float, R_cube_eef_at_grasp: Rotation) -> np.ndarray:
        """Calculates the EEF's world orientation when a cube it's holding is stacked flatly."""
        target_stacked_cube_quat_w_flat = self._flatten_quat_around_world_z(base_cube_quat_w, target_yaw_rad=target_stacked_cube_yaw_rad)
        R_w_target_stacked_cube_flat = Rotation.from_quat(quat_wxyz_to_xyzw(target_stacked_cube_quat_w_flat))
        return quat_xyzw_to_wxyz((R_w_target_stacked_cube_flat * R_cube_eef_at_grasp).as_quat())

    def generate(self) -> list:
        """
        Generates a predefined trajectory for stacking three cubes.
        Order: Cube1 (e.g. red, bottom), Cube2 (e.g. green, middle), Cube3 (e.g. yellow, top).
        Cubes are stacked flat (zero roll, zero pitch).
        """
        (initial_left_pos, initial_left_quat, current_right_eef_pos_w, current_right_eef_quat_wxyz_w,
         cube1_pos_w, cube1_quat_wxyz_w, cube2_pos_w, cube2_quat_wxyz_w, cube3_pos_w, cube3_quat_wxyz_w,
         *_) = TrajectoryPlayer.extract_essential_obs_data(self.obs)

        self.waypoints = []
        
        def add_waypoint(right_eef_pos, right_eef_quat, right_hand_closed_bool):
            self._add_waypoint(right_eef_pos, right_eef_quat, right_hand_closed_bool, self.initial_left_arm_pos_w, self.initial_left_arm_quat_wxyz_w, DEFAULT_LEFT_HAND_BOOL)

        # --- Calculate the fixed relative transformation (EEF in Cube's frame at grasp) ---
        # This is a one-time calculation using a virtual cube at the origin to find how the EEF
        # is positioned and oriented relative to a cube it's grasping.
        # This relative transform (t_cube_eef_in_cube_at_grasp, R_cube_eef_at_grasp)
        # will then be applied to the actual target cube poses.
        _temp_cube_pos = np.array([0.,0.,0.]); _temp_cube_quat_flat_upright = np.array([1.,0.,0.,0.])
        _eef_pregrasp_pos_rel_generic_cube, _eef_pregrasp_quat_rel_generic_cube = self._calculate_eef_world_pose_from_cube_relative(_temp_cube_pos, _temp_cube_quat_flat_upright, CUBE_STACK_PRE_GRASP_OFFSET_POS_CUBE_FRAME, CUBE_STACK_PRE_GRASP_EULER_XYZ_DEG_CUBE_FRAME)
        _eef_grasp_pos_rel_generic_cube = _eef_pregrasp_pos_rel_generic_cube - np.array([0,0, CUBE_STACK_GRASP_APPROACH_DISTANCE_Z_WORLD])
        t_cube_eef_in_cube_at_grasp = _eef_grasp_pos_rel_generic_cube
        R_cube_eef_at_grasp = Rotation.from_quat(quat_wxyz_to_xyzw(_eef_pregrasp_quat_rel_generic_cube))

        # --- Waypoint 0: Current EEF pose (right hand open) ---
        add_waypoint(self.initial_right_arm_pos_w, self.initial_right_arm_quat_wxyz_w, False)

        # --- Process Cube 2 (grasp and place on Cube 1) ---
        # 1.1 Pre-grasp Cube 2 (approach based on its *current* orientation) -- waypoint 1
        pre_grasp_c2_pos_w, pre_grasp_c2_quat_w = self._calculate_eef_world_pose_from_cube_relative(cube2_pos_w, cube2_quat_wxyz_w, CUBE_STACK_PRE_GRASP_OFFSET_POS_CUBE_FRAME, CUBE_STACK_PRE_GRASP_EULER_XYZ_DEG_CUBE_FRAME)
        add_waypoint(pre_grasp_c2_pos_w, pre_grasp_c2_quat_w, False)
        # 1.2 Approach Cube 2 (move down in world Z) -- waypoint 2
        grasp_c2_pos_w = pre_grasp_c2_pos_w - np.array([0,0, CUBE_STACK_GRASP_APPROACH_DISTANCE_Z_WORLD])
        add_waypoint(grasp_c2_pos_w, pre_grasp_c2_quat_w, False)
        # 1.3 Grasp Cube 2 -- waypoint 3
        add_waypoint(grasp_c2_pos_w, pre_grasp_c2_quat_w, True)
        # 1.4 Intermediate to Cube 1 -- waypoint 4
        intermediate_c1_pos_w = cube1_pos_w * 1/5 + cube2_pos_w * 4/5;  # Weighted average towards Cube 2
        intermediate_c1_pos_w[2] = cube1_pos_w[2] + 4.0 * CUBE_HEIGHT  # In case, the blue cube is too close to the green cube
        # Calculate intermediate orientation for Cube 2 placement
        cube1_yaw_rad_for_c2_stack = Rotation.from_quat(quat_wxyz_to_xyzw(cube1_quat_wxyz_w)).as_euler('zyx')[0]
        target_c2_on_c1_final_eef_quat_w = self._calculate_final_eef_orientation_for_stack(cube1_quat_wxyz_w, cube1_yaw_rad_for_c2_stack, R_cube_eef_at_grasp)
        intermediate_c1_quat_w = quat_xyzw_to_wxyz(Slerp([0, 1], Rotation.from_quat(quat_wxyz_to_xyzw(np.array([pre_grasp_c2_quat_w, target_c2_on_c1_final_eef_quat_w]))))(0.5).as_quat())
        add_waypoint(intermediate_c1_pos_w, intermediate_c1_quat_w, True)
        # 1.5 Stack Cube 2 on Cube 1 (Cube 2 will be flat, aligned in yaw with Cube 1)
        target_c2_on_c1_pos_w = cube1_pos_w + np.array([0,0, CUBE_STACK_ON_CUBE_Z_OFFSET])
        # Flatten Cube1's orientation to get target yaw for Cube2, ensuring Cube2 is placed flat.
        cube1_yaw_rad = Rotation.from_quat(quat_wxyz_to_xyzw(cube1_quat_wxyz_w)).as_euler('zyx')[0]
        target_c2_on_c1_quat_w_flat = self._flatten_quat_around_world_z(cube1_quat_wxyz_w, target_yaw_rad=cube1_yaw_rad)
        R_w_target_c2_flat = Rotation.from_quat(quat_wxyz_to_xyzw(target_c2_on_c1_quat_w_flat))
        # EEF pos = cube_target_pos + R_world_cube * t_eef_in_cube_frame
        stack_c2_eef_pos_w = target_c2_on_c1_pos_w + R_w_target_c2_flat.apply(t_cube_eef_in_cube_at_grasp)
        stack_c2_eef_quat_w = quat_xyzw_to_wxyz((R_w_target_c2_flat * R_cube_eef_at_grasp).as_quat())
        add_waypoint(stack_c2_eef_pos_w, stack_c2_eef_quat_w, True)
        # 1.6 Release Cube 2
        add_waypoint(stack_c2_eef_pos_w, stack_c2_eef_quat_w, False)
        # 1.7 Lift from Cube 2 with two times of CUBE_HEIGHT
        add_waypoint(stack_c2_eef_pos_w + np.array([0,0, 1.0 * CUBE_HEIGHT]), stack_c2_eef_quat_w, False)

        # --- Process Cube 3 (grasp and place on Cube 2+1) ---
        # 2.1 Pre-grasp Cube 3 (approach based on its *current* orientation)
        pre_grasp_c3_pos_w, pre_grasp_c3_quat_w = self._calculate_eef_world_pose_from_cube_relative(cube3_pos_w, cube3_quat_wxyz_w, CUBE_STACK_PRE_GRASP_OFFSET_POS_CUBE_FRAME, CUBE_STACK_PRE_GRASP_EULER_XYZ_DEG_CUBE_FRAME)
        add_waypoint(pre_grasp_c3_pos_w, pre_grasp_c3_quat_w, False)
        # 2.2 Approach Cube 3 in z-axis (move down in world Z)
        grasp_c3_pos_w = pre_grasp_c3_pos_w - np.array([0,0, CUBE_STACK_GRASP_APPROACH_DISTANCE_Z_WORLD])
        add_waypoint(grasp_c3_pos_w, pre_grasp_c3_quat_w, False)
        # 2.3 Grasp Cube 3
        add_waypoint(grasp_c3_pos_w, pre_grasp_c3_quat_w, True)
        # 2.4 Intermediate to Cube 1 (+2)
        intermediate_c1_pos_w = cube1_pos_w * 1/5 + cube3_pos_w * 4/5; intermediate_c1_pos_w[2] = cube1_pos_w[2] + 4.5 * CUBE_HEIGHT
        # Calculate intermediate orientation for Cube 3 placement
        cube1_yaw_rad_for_c3_stack = Rotation.from_quat(quat_wxyz_to_xyzw(cube1_quat_wxyz_w)).as_euler('zyx')[0]
        target_c3_on_c2_final_eef_quat_w = self._calculate_final_eef_orientation_for_stack(cube1_quat_wxyz_w, cube1_yaw_rad_for_c3_stack, R_cube_eef_at_grasp)
        intermediate_c2_actual_quat_w = quat_xyzw_to_wxyz(Slerp([0, 1], Rotation.from_quat(quat_wxyz_to_xyzw(np.array([pre_grasp_c3_quat_w, target_c3_on_c2_final_eef_quat_w]))))(0.5).as_quat())
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
        add_waypoint(stack_c3_eef_pos_w + np.array([0,0, 0.5 * CUBE_HEIGHT]), stack_c3_eef_quat_w, False)

        # --- Final: Return to initial right arm pose ---
        add_waypoint(self.initial_right_arm_pos_w, self.initial_right_arm_quat_wxyz_w, False)
        
        # print(f"  Generated {len(self.recorded_waypoints)} waypoints for auto cube stacking.")
        # print("  --- Generated Waypoint Details ---")
        # for i, wp in enumerate(self.recorded_waypoints):
        #     print(f"  Waypoint {i}:")
        #     print(f"    Right Arm EEF: Pos={wp['right_arm_eef'][:3]}, Quat={wp['right_arm_eef'][3:7]}, GripperOpen={not wp['right_hand_bool']}")
        # print("--- End of Cube Stacking Trajectory Generation ---\n")
        
        return self.waypoints

class KitchenTasksTrajectoryGenerator(BaseTrajectoryGenerator):
    """Generates trajectories for the multi-step kitchen environment."""

    def generate_open_drawer_sub_trajectory(self, obs: dict, initial_poses: Optional[dict] = None) -> tuple[list, dict]:
        """Generates a trajectory to open the drawer."""
        open_drawer_skill = OpenDrawerSkill(obs, initial_poses)
        return open_drawer_skill.get_full_trajectory()

    def generate_pick_and_place_mug_sub_trajectory(self, obs: dict, initial_poses: Optional[dict] = None) -> tuple[list, dict]:
        """Generates a trajectory to pick the mug, place it on the mat, and close the drawer."""
        # Refactored to sequence two skills
        pick_skill = PickMugFromDrawerSkill(obs, initial_poses)
        pick_waypoints, pick_final_poses = pick_skill.get_full_trajectory()

        place_skill = PlaceMugOnMatSkill(obs, initial_poses=pick_final_poses)
        place_waypoints, place_final_poses = place_skill.get_full_trajectory()

        # The first waypoint of the second skill is the same as the last of the first,
        # so we skip it to avoid a redundant waypoint.
        self.waypoints = pick_waypoints + place_waypoints[1:]
        
        return self.waypoints, place_final_poses

    def generate_pour_bottle_sub_trajectory(self, obs: dict, initial_poses: Optional[dict] = None, home_poses: Optional[dict] = None) -> tuple[list, dict]:
        """Generates a trajectory to pour the bottle into the mug."""
        pour_skill = PourBottleSkill(obs, initial_poses)
        pour_waypoints, pour_final_poses = pour_skill.get_full_trajectory()

        # Pass the final poses from the pour skill to the return skill.
        # Also, pass the optional home_poses dict.
        return_initial_poses = pour_final_poses
        if home_poses:
            return_initial_poses["home_poses"] = home_poses

        return_skill = ReturnBottleSkill(obs, initial_poses=return_initial_poses)
        return_waypoints, return_final_poses = return_skill.get_full_trajectory()

        # The first waypoint of the second skill is the same as the last of the first,
        # so we skip it to avoid a redundant waypoint.
        self.waypoints = pour_waypoints + return_waypoints[1:]

        return self.waypoints, return_final_poses

class FileBasedTrajectoryGenerator(BaseTrajectoryGenerator):
    """Generates a trajectory by loading waypoints from a YAML/JSON file."""

    def __init__(self, obs: dict, filepath: str):
        super().__init__(obs)
        self.filepath = filepath

    def generate(self) -> list:
        """Loads waypoints from file and prepends the current robot pose."""
        self.waypoints = []
        (current_left_eef_pos_w, current_left_eef_quat_wxyz_w,
         current_right_eef_pos_w, current_right_eef_quat_wxyz_w,
         *_) = TrajectoryPlayer.extract_essential_obs_data(self.obs)

        # Waypoint 0: Current pose
        start_waypoint = {
            "left_arm_eef": np.concatenate([current_left_eef_pos_w, current_left_eef_quat_wxyz_w]),
            "right_arm_eef": np.concatenate([current_right_eef_pos_w, current_right_eef_quat_wxyz_w]),
            "left_hand_bool": 0,
            "right_hand_bool": 0
        }
        self.waypoints.append(start_waypoint)

        try:
            with open(self.filepath, 'r') as f:
                if self.filepath.endswith(".yaml") or self.filepath.endswith(".yml"):
                    loaded_wps_list = yaml.safe_load(f)
                else: # Assume JSON
                    loaded_wps_list = json.load(f)
        except FileNotFoundError:
            print(f"Waypoint file {self.filepath} not found. No waypoints loaded.")
            return self.waypoints
        except (yaml.YAMLError, json.JSONDecodeError) as e:
            print(f"Error decoding file {self.filepath}: {e}")
            return self.waypoints
        
        for wp_dict in loaded_wps_list:
            self.waypoints.append({
                "left_arm_eef": np.array(wp_dict["left_arm_eef"]),
                "right_arm_eef": np.array(wp_dict["right_arm_eef"]),
                "left_hand_bool": int(wp_dict["left_hand_bool"]),
                "right_hand_bool": int(wp_dict["right_hand_bool"])
            })
        
        print(f"Loaded {len(loaded_wps_list)} waypoints from {self.filepath} and prepended current pose.")
        return self.waypoints