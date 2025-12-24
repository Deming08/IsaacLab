# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This module contains classes for generating predefined trajectories for various robotic tasks."""

import yaml
import json
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation, Slerp # type: ignore

from . import constants as C
from .quaternion_utils import quat_xyzw_to_wxyz, quat_wxyz_to_xyzw
from .trajectory_player import TrajectoryPlayer # For static method access
from .skills import (
    SubTask,
    # Can-sorting skills
    GraspCanSkill,
    PlaceCanInBasketSkill,
    # Cabinet-related skills
    OpenDrawerSkill,
    PickMugFromDrawerSkill,
    PlaceMugOnMatSkill,
    GraspBottleSkill,
    PourBottleSkill,
    ReturnBottleSkill,
    create_waypoint,
    
    generate_transit_or_transfer_motion,
    generate_retract_trajectory,
)

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
        self.home_poses = {
            "right_eef_pos": C.HOME_POSES["right_eef_pos"], "right_eef_quat": C.HOME_POSES["right_eef_quat"], 
            "left_eef_pos": C.HOME_POSES["left_eef_pos"], "left_eef_quat": C.HOME_POSES["left_eef_quat"],
            "right_hand_closed": False, "left_hand_closed": False
        }

    def generate(self, *args, **kwargs) -> list:
        """Generates and returns a list of waypoints. To be implemented by subclasses."""
        raise NotImplementedError

    def _add_waypoint(self, right_eef_pos, right_eef_quat, right_hand_closed_bool, left_eef_pos, left_eef_quat, left_hand_closed_bool):
        """Helper to append a waypoint to the recorded_waypoints list."""
        wp = {
            "left_eef": np.concatenate([left_eef_pos, left_eef_quat]),
            "right_eef": np.concatenate([right_eef_pos, right_eef_quat]),
            "left_hand_bool": int(left_hand_closed_bool),
            "right_hand_bool": int(right_hand_closed_bool)
        }
        self.waypoints.append(wp)

class GraspPickPlaceTrajectoryGenerator(BaseTrajectoryGenerator):
    """Generates a modular trajectory for grasping a can and placing it in a basket."""

    def __init__(self, obs: dict, initial_poses: dict):
        super().__init__(obs)
        self.initial_poses = initial_poses

    def generate(self) -> list:
        """Main trajectory generation orchestrating all sub-tasks."""

        # 1. Generate grasp trajectory to grasp the target can
        grasp_sub_task = SubTask(self.obs, initial_poses=self.initial_poses, skill=GraspCanSkill(self.obs))
        grasp_waypoints, grasp_final_poses = grasp_sub_task.get_full_trajectory()

        # 2. Generate place trajectory to place the can in the basket
        place_sub_task = SubTask(self.obs, initial_poses=grasp_final_poses, skill=PlaceCanInBasketSkill(self.obs))
        place_waypoints, place_final_poses = place_sub_task.get_full_trajectory()

        # 3. Generate return trajectory to the initial poses
        return_waypoints, _ = generate_transit_or_transfer_motion(self.obs, initial_poses=place_final_poses, target_poses=self.home_poses)

        self.waypoints = grasp_waypoints + place_waypoints + return_waypoints
        
        # print("=" * 60)
        # print(f"GraspPickPlaceTrajectoryGenerator Full waypoints {len(self.waypoints)}:")
        # for i, wp in enumerate(self.waypoints):
        #     print(f"Waypoint {i}: {wp}")
        # print("=" * 60)
        
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
            self._add_waypoint(right_eef_pos, right_eef_quat, right_hand_closed_bool, self.initial_left_arm_pos_w, self.initial_left_arm_quat_wxyz_w, C.DEFAULT_LEFT_HAND_BOOL)

        # --- Calculate the fixed relative transformation (EEF in Cube's frame at grasp) ---
        # This is a one-time calculation using a virtual cube at the origin to find how the EEF
        # is positioned and oriented relative to a cube it's grasping.
        # This relative transform (t_cube_eef_in_cube_at_grasp, R_cube_eef_at_grasp)
        # will then be applied to the actual target cube poses.
        _temp_cube_pos = np.array([0.,0.,0.]); _temp_cube_quat_flat_upright = np.array([1.,0.,0.,0.])
        _eef_pregrasp_pos_rel_generic_cube, _eef_pregrasp_quat_rel_generic_cube = self._calculate_eef_world_pose_from_cube_relative(_temp_cube_pos, _temp_cube_quat_flat_upright, C.CUBE_STACK_PRE_GRASP_OFFSET_POS_CUBE_FRAME, C.CUBE_STACK_PRE_GRASP_EULER_XYZ_DEG_CUBE_FRAME)
        _eef_grasp_pos_rel_generic_cube = _eef_pregrasp_pos_rel_generic_cube - np.array([0,0, C.CUBE_STACK_GRASP_APPROACH_DISTANCE_Z_WORLD])
        t_cube_eef_in_cube_at_grasp = _eef_grasp_pos_rel_generic_cube
        R_cube_eef_at_grasp = Rotation.from_quat(quat_wxyz_to_xyzw(_eef_pregrasp_quat_rel_generic_cube))

        # --- Waypoint 0: Current EEF pose (right hand open) ---
        add_waypoint(self.initial_right_arm_pos_w, self.initial_right_arm_quat_wxyz_w, False)

        # --- Process Cube 2 (grasp and place on Cube 1) ---
        # 1.1 Pre-grasp Cube 2 (approach based on its *current* orientation) -- waypoint 1
        pre_grasp_c2_pos_w, pre_grasp_c2_quat_w = self._calculate_eef_world_pose_from_cube_relative(cube2_pos_w, cube2_quat_wxyz_w, C.CUBE_STACK_PRE_GRASP_OFFSET_POS_CUBE_FRAME, C.CUBE_STACK_PRE_GRASP_EULER_XYZ_DEG_CUBE_FRAME)
        add_waypoint(pre_grasp_c2_pos_w, pre_grasp_c2_quat_w, False)
        # 1.2 Approach Cube 2 (move down in world Z) -- waypoint 2
        grasp_c2_pos_w = pre_grasp_c2_pos_w - np.array([0,0, C.CUBE_STACK_GRASP_APPROACH_DISTANCE_Z_WORLD])
        add_waypoint(grasp_c2_pos_w, pre_grasp_c2_quat_w, False)
        # 1.3 Grasp Cube 2 -- waypoint 3
        add_waypoint(grasp_c2_pos_w, pre_grasp_c2_quat_w, True)
        # 1.4 Intermediate to Cube 1 -- waypoint 4
        intermediate_c1_pos_w = cube1_pos_w * 1/5 + cube2_pos_w * 4/5;  # Weighted average towards Cube 2
        intermediate_c1_pos_w[2] = cube1_pos_w[2] + 4.0 * C.CUBE_HEIGHT  # In case, the blue cube is too close to the green cube
        # Calculate intermediate orientation for Cube 2 placement
        cube1_yaw_rad_for_c2_stack = Rotation.from_quat(quat_wxyz_to_xyzw(cube1_quat_wxyz_w)).as_euler('zyx')[0]
        target_c2_on_c1_final_eef_quat_w = self._calculate_final_eef_orientation_for_stack(cube1_quat_wxyz_w, cube1_yaw_rad_for_c2_stack, R_cube_eef_at_grasp)
        intermediate_c1_quat_w = quat_xyzw_to_wxyz(Slerp([0, 1], Rotation.from_quat(quat_wxyz_to_xyzw(np.array([pre_grasp_c2_quat_w, target_c2_on_c1_final_eef_quat_w]))))(0.5).as_quat())
        add_waypoint(intermediate_c1_pos_w, intermediate_c1_quat_w, True)
        # 1.5 Stack Cube 2 on Cube 1 (Cube 2 will be flat, aligned in yaw with Cube 1)
        target_c2_on_c1_pos_w = cube1_pos_w + np.array([0,0, C.CUBE_STACK_ON_CUBE_Z_OFFSET])
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
        add_waypoint(stack_c2_eef_pos_w + np.array([0,0, 1.0 * C.CUBE_HEIGHT]), stack_c2_eef_quat_w, False)

        # --- Process Cube 3 (grasp and place on Cube 2+1) ---
        # 2.1 Pre-grasp Cube 3 (approach based on its *current* orientation)
        pre_grasp_c3_pos_w, pre_grasp_c3_quat_w = self._calculate_eef_world_pose_from_cube_relative(cube3_pos_w, cube3_quat_wxyz_w, C.CUBE_STACK_PRE_GRASP_OFFSET_POS_CUBE_FRAME, C.CUBE_STACK_PRE_GRASP_EULER_XYZ_DEG_CUBE_FRAME)
        add_waypoint(pre_grasp_c3_pos_w, pre_grasp_c3_quat_w, False)
        # 2.2 Approach Cube 3 in z-axis (move down in world Z)
        grasp_c3_pos_w = pre_grasp_c3_pos_w - np.array([0,0, C.CUBE_STACK_GRASP_APPROACH_DISTANCE_Z_WORLD])
        add_waypoint(grasp_c3_pos_w, pre_grasp_c3_quat_w, False)
        # 2.3 Grasp Cube 3
        add_waypoint(grasp_c3_pos_w, pre_grasp_c3_quat_w, True)
        # 2.4 Intermediate to Cube 1 (+2)
        intermediate_c1_pos_w = cube1_pos_w * 1/5 + cube3_pos_w * 4/5; intermediate_c1_pos_w[2] = cube1_pos_w[2] + 4.5 * C.CUBE_HEIGHT
        # Calculate intermediate orientation for Cube 3 placement
        cube1_yaw_rad_for_c3_stack = Rotation.from_quat(quat_wxyz_to_xyzw(cube1_quat_wxyz_w)).as_euler('zyx')[0]
        target_c3_on_c2_final_eef_quat_w = self._calculate_final_eef_orientation_for_stack(cube1_quat_wxyz_w, cube1_yaw_rad_for_c3_stack, R_cube_eef_at_grasp)
        intermediate_c2_actual_quat_w = quat_xyzw_to_wxyz(Slerp([0, 1], Rotation.from_quat(quat_wxyz_to_xyzw(np.array([pre_grasp_c3_quat_w, target_c3_on_c2_final_eef_quat_w]))))(0.5).as_quat())
        add_waypoint(intermediate_c1_pos_w, intermediate_c2_actual_quat_w, True)
        # 2.5 Stack Cube 3 on Cube 2 (+1)
        target_c3_on_c2_pos_w = cube1_pos_w + np.array([0,0, 2 * C.CUBE_STACK_ON_CUBE_Z_OFFSET])
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
        add_waypoint(stack_c3_eef_pos_w + np.array([0,0, 0.5 * C.CUBE_HEIGHT]), stack_c3_eef_quat_w, False)

        # --- Final: Return to initial right arm pose ---
        add_waypoint(self.initial_right_arm_pos_w, self.initial_right_arm_quat_wxyz_w, False)
        
        # print(f"  Generated {len(self.recorded_waypoints)} waypoints for auto cube stacking.")
        # print("  --- Generated Waypoint Details ---")
        # for i, wp in enumerate(self.recorded_waypoints):
        #     print(f"  Waypoint {i}:")
        #     print(f"    Right Arm EEF: Pos={wp['right_eef'][:3]}, Quat={wp['right_eef'][3:7]}, GripperOpen={not wp['right_hand_bool']}")
        # print("--- End of Cube Stacking Trajectory Generation ---")
        return self.waypoints

class KitchenTasksTrajectoryGenerator(BaseTrajectoryGenerator):
    """Generates trajectories for the multi-step kitchen environment."""

    def generate_open_drawer_trajectory(self, obs: dict, initial_poses: Optional[dict] = None) -> tuple[list, dict]:
        """Generates a trajectory to open the drawer."""
        (_, _, _, _, *_, drawer_pos, drawer_quat, _, _, _, _, _, _) = TrajectoryPlayer.extract_essential_obs_data(obs)

        # 1. Define transit target pose
        R_world_drawer = Rotation.from_quat(quat_wxyz_to_xyzw(drawer_quat))
        pre_approach_handle_pos = drawer_pos + R_world_drawer.apply(C.DRAWER_HANDLE_APPROACH_POS)
        approach_handle_quat = quat_xyzw_to_wxyz((R_world_drawer * Rotation.from_euler('xyz', C.DRAWER_HANDLE_APPROACH_QUAT, degrees=True)).as_quat())
        
        transit_target_pose = {
            "right_pos": pre_approach_handle_pos, "right_quat": approach_handle_quat,
            "left_pos": C.ARM_PREPARE_POSES["left_pos"], "left_quat": C.ARM_PREPARE_POSES["left_quat"],
        }

        # 2. Create and execute the sub-task
        open_drawer_sub_task = SubTask(
            obs,
            initial_poses=initial_poses,
            transit_target_pose=transit_target_pose,
            skill=OpenDrawerSkill(obs),            
        )
        self.waypoints, final_poses = open_drawer_sub_task.get_full_trajectory()
        return self.waypoints, final_poses

    def generate_pick_and_place_mug_trajectory(self, obs: dict, initial_poses: Optional[dict] = None, home_poses: Optional[dict] = None) -> tuple[list, dict]:
        """Generates a trajectory to pick the mug, place it on the mat, and retract."""
        (_, _, _, _, *_, mug_pos, mug_quat, mug_mat_pos, mug_mat_quat) = TrajectoryPlayer.extract_essential_obs_data(obs)

        # 1. Sub-task to pick the mug
        approach_mug_pos = C.MUG_APPROACH_ABS_POS
        approach_mug_quat = quat_xyzw_to_wxyz(Rotation.from_euler('xyz', C.MUG_APPROACH_ABS_QUAT, degrees=True).as_quat())
        
        # print(f"mug_pos: {mug_pos}, mug_quat: {mug_quat}")
        # print(f"approach_mug_pos: {approach_mug_pos}, approach_mug_quat: {approach_mug_quat}")

        pick_sub_task = SubTask(
            obs,
            transit_target_pose={
                "left_pos": approach_mug_pos, "left_quat": approach_mug_quat,
                "right_hand_closed": True,
            },
            skill=PickMugFromDrawerSkill(obs),
            initial_poses=initial_poses
        )
        pick_waypoints, pick_final_poses = pick_sub_task.get_full_trajectory()
        
        # print(f"pick_waypoints: {len(pick_waypoints)} waypoints generated.")
        # # Print each waypoint for debugging
        # for i, wp in enumerate(pick_waypoints):
        #     print(f"  Waypoint {i}: Left Pos={wp['left_eef'][:3]}, Left Quat={wp['left_eef'][3:7]}, Left Hand Closed={wp['left_hand_bool']}")

        # 2. Sub-task to place the mug
        mug_on_mat_quat = quat_xyzw_to_wxyz(Rotation.from_euler('xyz', C.MAT_PLACE_ABS_QUAT, degrees=True).as_quat())
        pre_place_mat_pos = mug_mat_pos + C.MAT_APPROACH_POS

        place_sub_task = SubTask(
            obs,
            transit_target_pose={
                "left_pos": pre_place_mat_pos, "left_quat": mug_on_mat_quat,
            },
            skill=PlaceMugOnMatSkill(obs),
            initial_poses=pick_final_poses
        )
        place_waypoints, place_final_poses = place_sub_task.get_full_trajectory()

        # 3. Retract trajectory
        retract_waypoints, retract_final_poses = generate_retract_trajectory(obs, initial_poses=place_final_poses)

        # # Print each waypoint for debugging
        # print(f"mug_mat_pos: {mug_mat_pos}, mug_mat_quat: {mug_mat_quat}")
        # print(f"place_waypoints: {len(place_waypoints)} waypoints generated.")
        # for i, wp in enumerate(place_waypoints):
        #     print(f"  Waypoint {i}: Left Pos={wp['left_eef'][:3]}, Left Quat={wp['left_eef'][3:7]}, Left Hand Closed={wp['left_hand_bool']}")

        self.waypoints = pick_waypoints + place_waypoints[1:] + retract_waypoints[1:]
        return self.waypoints, retract_final_poses

    def generate_pour_bottle_trajectory(self, obs: dict, initial_poses: Optional[dict] = None, home_poses: Optional[dict] = None) -> tuple[list, dict]:
        """Generates a trajectory to pour the bottle into the mug and return home."""
        (_, _, _, _, *_, bottle_pos, bottle_quat, mug_pos, _, _, _) = TrajectoryPlayer.extract_essential_obs_data(obs)

        # 1. Sub-task to grasp the bottle
        grasp_sub_task = SubTask(
            obs,
            skill=GraspBottleSkill(obs),
            initial_poses=initial_poses
        )
        grasp_wps, grasp_poses = grasp_sub_task.get_full_trajectory()

        # 2. Sub-task to pour the bottle
        pre_pour_pos = mug_pos + C.BOTTLE_PRE_POUR_MAT_POS
        pour_sub_task = SubTask(
            obs,
            transit_target_pose={
                "right_pos": pre_pour_pos, "right_quat": grasp_poses["right_eef_quat"], # Maintain orientation
            },
            skill=PourBottleSkill(obs),
            initial_poses=grasp_poses
        )
        pour_wps, pour_poses = pour_sub_task.get_full_trajectory()

        # 3. Sub-task to return the bottle
        pre_return_pos = bottle_pos + C.BOTTLE_GRASP_POS + C.BOTTLE_LIFT_POS
        return_sub_task = SubTask(
            obs,
            transit_target_pose={
                "right_pos": pre_return_pos, "right_quat": pour_poses["right_eef_quat"], # Maintain vertical orientation
            },
            skill=ReturnBottleSkill(obs),
            initial_poses=pour_poses
        )
        return_wps, return_poses = return_sub_task.get_full_trajectory()
        
        # # Print each waypoint for debugging
        # print(f"bottle_pos: {bottle_pos}, bottle_quat: {bottle_quat}")
        # print(f"return_wps: {len(return_wps)} waypoints generated.")
        # for i, wp in enumerate(return_wps):
        #     print(f"  Waypoint {i}: Right Pos={wp['right_eef'][:3]}, Right Quat={wp['right_eef'][3:7]}, Right Hand Closed={wp['right_hand_bool']}")

        # 4. Go home
        target_home_poses = C.HOME_POSES  # home_poses if home_poses is not None else HOME_POSES
        home_wps, home_final_poses = generate_transit_or_transfer_motion(
            obs,
            initial_poses=return_poses,
            target_poses=target_home_poses
        )

        # # Print each waypoint for debugging
        # print(f"home_wps: {len(home_wps)} waypoints generated.")
        # for i, wp in enumerate(home_wps):
        #     print(f"  Waypoint {i}: Right Pos={wp['right_eef'][:3]}, Right Quat={wp['right_eef'][3:7]}, Right Hand Closed={wp['right_hand_bool']}")

        self.waypoints = grasp_wps + pour_wps[1:] + return_wps[1:] + home_wps[1:]
        return self.waypoints, home_final_poses


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
        self.waypoints.append(create_waypoint(current_left_eef_pos_w, current_left_eef_quat_wxyz_w, False, current_right_eef_pos_w, current_right_eef_quat_wxyz_w, False))

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
                "left_eef": np.array(wp_dict["left_arm_eef"]),
                "right_eef": np.array(wp_dict["right_arm_eef"]),
                "left_hand_bool": int(wp_dict["left_hand_bool"]),
                "right_hand_bool": int(wp_dict["right_hand_bool"])
            })
        
        print(f"Loaded {len(loaded_wps_list)} waypoints from {self.filepath} and prepended current pose.")
        return self.waypoints
