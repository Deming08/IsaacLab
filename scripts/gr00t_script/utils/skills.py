# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This module contains modular skill definitions based on the SkillMimicGen paradigm."""

from typing import Optional, Dict, Union
import numpy as np
from scipy.spatial.transform import Rotation

from . import constants as C
from .quaternion_utils import quat_xyzw_to_wxyz, quat_wxyz_to_xyzw
from .trajectory_player import TrajectoryPlayer


def create_waypoint(right_eef_pos, right_eef_quat, right_hand_closed_bool, left_eef_pos, left_eef_quat, left_hand_closed_bool):
    """Helper to create a waypoint dictionary."""
    wp = {
        "left_arm_eef": np.concatenate([left_eef_pos, left_eef_quat]),
        "right_arm_eef": np.concatenate([right_eef_pos, right_eef_quat]),
        "left_hand_bool": int(left_hand_closed_bool),
        "right_hand_bool": int(right_hand_closed_bool)
    }
    return wp


def generate_transit_or_transfer_motion(obs: Dict, initial_poses: Optional[dict] = None, target_poses: Optional[Union[dict, list]] = None) -> tuple[list, dict]:
    """A generic skill to move one or both arms to a target pose or a series of target poses."""
    (current_left_pos, current_left_quat, current_right_pos, current_right_quat, *_) = TrajectoryPlayer.extract_essential_obs_data(obs)
    
    if initial_poses:
        start_right_pos, start_right_quat = initial_poses["right_eef_pos"], initial_poses["right_eef_quat"]
        start_left_pos, start_left_quat = initial_poses["left_eef_pos"], initial_poses["left_eef_quat"]
        start_right_hand, start_left_hand = initial_poses.get("right_hand_closed", False), initial_poses.get("left_hand_closed", False)
    else:
        start_right_pos, start_right_quat = current_right_pos, current_right_quat
        start_left_pos, start_left_quat = current_left_pos, current_left_quat
        start_right_hand, start_left_hand = False, False
        
    init_waypoints = [create_waypoint(start_right_pos, start_right_quat, start_right_hand, start_left_pos, start_left_quat, start_left_hand)]

    motion_waypoints = []
    if target_poses:
        if not isinstance(target_poses, list):
            target_poses = [target_poses]

        last_right_pos, last_right_quat = start_right_pos, start_right_quat
        last_left_pos, last_left_quat = start_left_pos, start_left_quat
        last_right_hand, last_left_hand = start_right_hand, start_left_hand

        for poses in target_poses:
            # Default to last poses if no target is given for an arm
            target_right_pos, target_right_quat = poses.get("right_pos", last_right_pos), poses.get("right_quat", last_right_quat)
            target_left_pos, target_left_quat = poses.get("left_pos", last_left_pos), poses.get("left_quat", last_left_quat)
            # Gripper state is preserved from start unless specified in target
            target_right_hand, target_left_hand = poses.get("right_hand_closed", last_right_hand), poses.get("left_hand_closed", last_left_hand)

            motion_waypoints.append(create_waypoint(target_right_pos, target_right_quat, target_right_hand, target_left_pos, target_left_quat, target_left_hand))

            last_right_pos, last_right_quat = target_right_pos, target_right_quat
            last_left_pos, last_left_quat = target_left_pos, target_left_quat
            last_right_hand, last_left_hand = target_right_hand, target_left_hand

    waypoints = init_waypoints + motion_waypoints
    
    if not waypoints:
        return [], {}

    final_wp = waypoints[-1]
    final_poses = {
        "left_eef_pos": final_wp["left_arm_eef"][:3],
        "left_eef_quat": final_wp["left_arm_eef"][3:7],
        "right_eef_pos": final_wp["right_arm_eef"][:3],
        "right_eef_quat": final_wp["right_arm_eef"][3:7],
        "left_hand_closed": bool(final_wp["left_hand_bool"]),
        "right_hand_closed": bool(final_wp["right_hand_bool"]),
    }
    return waypoints, final_poses


def generate_retract_trajectory(obs: Dict, initial_poses: Optional[dict] = None) -> tuple[list, dict]:
    """Generates a multi-point trajectory to retract arms to specific intermediate poses."""
    retract_targets = [
        {
            "right_pos": C.RETRACT_WAYPOINTS["right_retract_pos"], "right_quat": C.RETRACT_WAYPOINTS["right_retract_quat"], "right_hand_closed": False,
            "left_pos": C.RETRACT_WAYPOINTS["left_retract_pos"], "left_quat": C.RETRACT_WAYPOINTS["left_retract_quat"], "left_hand_closed": False,
        },
        {
            "right_pos": C.RETRACT_WAYPOINTS["right_restore_pos"], "right_quat": C.RETRACT_WAYPOINTS["right_restore_quat"], "right_hand_closed": False,
        }
    ]
    return generate_transit_or_transfer_motion(obs, initial_poses=initial_poses, target_poses=retract_targets)


class Skill:
    """Base class for a single robotic skill, divided into phases."""

    def __init__(self, obs: Dict, initial_poses: Optional[dict] = None, left_hand_name: str = "left_hand", right_hand_name: str = "right_hand"):
        self.obs = obs
        self.initial_poses = initial_poses
        self.left_hand_name = left_hand_name
        self.right_hand_name = right_hand_name
        self.init_waypoints = []
        self.motion_waypoints = []
        self.terminal_waypoints = []
        self.waypoints = []

    def init_phase(self):
        """
        Generates waypoints for the initial phase.
        Definition: The preparation and approach phase before the main action.
        This often involves moving the robot's end-effector(s) from a starting pose to a pre-interaction pose.
        """
        raise NotImplementedError

    def motion_phase(self):
        """
        Generates waypoints for the core motion phase.
        Definition: The main action of the skill where the robot interacts with objects or performs the primary task.
        Examples include grasping, pulling, pushing, or intricate movements.
        """
        raise NotImplementedError

    def terminal_phase(self):
        """
        Generates waypoints for the terminal phase.
        Definition: The concluding part of the skill after the main action is complete.
        This often involves retracting the end-effector(s) to a safe or neutral position, releasing an object,
        or transitioning to a state ready for the next skill.
        """
        raise NotImplementedError

    def get_skill_trajectory(self) -> tuple[list, dict]:
        """Executes all phases and returns the complete waypoint trajectory."""
        self.init_phase()
        self.motion_phase()
        self.terminal_phase()
        self.waypoints = self.init_waypoints + self.motion_waypoints + self.terminal_waypoints
        
        if not self.waypoints:
            return [], {}

        final_wp = self.waypoints[-1]
        final_poses = {
            "left_eef_pos": final_wp["left_arm_eef"][:3],
            "left_eef_quat": final_wp["left_arm_eef"][3:7],
            "right_eef_pos": final_wp["right_arm_eef"][:3],
            "right_eef_quat": final_wp["right_arm_eef"][3:7],
            "left_hand_closed": bool(final_wp["left_hand_bool"]),
            "right_hand_closed": bool(final_wp["right_hand_bool"]),
        }
        return self.waypoints, final_poses


class SubTask:
    """A higher-level task combining a pre-transit motion and a skill."""

    def __init__(self, obs: Dict, pre_transit_target_poses: Optional[dict] = None, skill: Optional[Skill] = None, initial_poses: Optional[dict] = None):
        self.obs = obs
        self.pre_transit_target_poses = pre_transit_target_poses
        self.skill = skill
        self.initial_poses = initial_poses

    def get_full_trajectory(self) -> tuple[list, dict]:
        """Generates the full trajectory for the sub-task."""
        # Generate pre-transit motion
        transit_waypoints, transit_final_poses = generate_transit_or_transfer_motion(self.obs, initial_poses=self.initial_poses, target_poses=self.pre_transit_target_poses)

        # Generate skill trajectory
        if self.skill:
            self.skill.initial_poses = transit_final_poses
            skill_waypoints, skill_final_poses = self.skill.get_skill_trajectory()
            # Combine trajectories
            waypoints = transit_waypoints + skill_waypoints[1:]
            return waypoints, skill_final_poses
        
        return transit_waypoints, transit_final_poses


class OpenDrawerSkill(Skill):
    """Skill to open a drawer."""

    def init_phase(self):
        """
        Definition: From the pre-approaching pose to the drawing pose.
        """
        (initial_left_pos, initial_left_quat, current_right_eef_pos_w, current_right_eef_quat_wxyz_w,
         *_, drawer_pos, drawer_quat, _, _, _, _, _, _) = TrajectoryPlayer.extract_essential_obs_data(self.obs)

        start_poses = self.initial_poses
        if not start_poses:
             start_poses = {
                "right_eef_pos": current_right_eef_pos_w, "right_eef_quat": current_right_eef_quat_wxyz_w,
                "left_eef_pos": initial_left_pos, "left_eef_quat": initial_left_quat,
                "right_hand_closed": False, "left_hand_closed": False
             }

        self.R_world_drawer = Rotation.from_quat(quat_wxyz_to_xyzw(drawer_quat))
        self.approach_handle_quat = quat_xyzw_to_wxyz((self.R_world_drawer * Rotation.from_euler('xyz', C.DRAWER_HANDLE_APPROACH_QUAT, degrees=True)).as_quat())
        
        # Add the starting pre-approach waypoint
        self.init_waypoints.append(create_waypoint(start_poses["right_eef_pos"], start_poses["right_eef_quat"], False, start_poses["left_eef_pos"], start_poses["left_eef_quat"], False))

        # Move to the drawing pose
        self.approach_handle_pos = drawer_pos + self.R_world_drawer.apply(C.DRAWER_HANDLE_GRASP_POS)
        self.init_waypoints.append(create_waypoint(self.approach_handle_pos, self.approach_handle_quat, False, start_poses["left_eef_pos"], start_poses["left_eef_quat"], False))

        self.drawer_pos = drawer_pos
        self.prepare_left_pos = start_poses["left_eef_pos"]
        self.prepare_left_quat = start_poses["left_eef_quat"]

    def motion_phase(self):
        """
        Definition: Grasping the drawer.
        """
        # Grasp the drawer handle
        self.motion_waypoints.append(create_waypoint(self.approach_handle_pos, self.approach_handle_quat, True, self.prepare_left_pos, self.prepare_left_quat, False))

    def terminal_phase(self):
        """
        Definition: Pull out the drawer completely.
        """
        # Pull out the drawer handle
        pulled_handle_pos = self.drawer_pos + self.R_world_drawer.apply(C.DRAWER_HANDLE_PULL_POS)
        self.terminal_waypoints.append(create_waypoint(pulled_handle_pos, self.approach_handle_quat, True, self.prepare_left_pos, self.prepare_left_quat, False))


class PickMugFromDrawerSkill(Skill):
    """Skill to pick a mug from inside a drawer."""

    def init_phase(self):
        """
        Definition: Approaching the mug.
        """
        (current_left_pos, current_left_quat, current_right_eef_pos_w, current_right_eef_quat_wxyz_w,
         *_, _, _, _, _, self.mug_pos, self.mug_quat, _, _) = TrajectoryPlayer.extract_essential_obs_data(self.obs)

        self.start_right_pos, self.start_right_quat, self.start_left_pos, self.start_left_quat = (self.initial_poses["right_eef_pos"], self.initial_poses["right_eef_quat"], self.initial_poses["left_eef_pos"], self.initial_poses["left_eef_quat"]) if self.initial_poses else (current_right_eef_pos_w, current_right_eef_quat_wxyz_w, current_left_pos, current_left_quat)
        
        # Start from the pre-approach pose
        self.init_waypoints.append(create_waypoint(self.start_right_pos, self.start_right_quat, True, self.start_left_pos, self.start_left_quat, False))

        # Approach the mug (left hand)
        self.grasp_mug_pos = self.mug_pos + C.MUG_GRASP_POS
        # mug_yaw = Rotation.from_quat(quat_wxyz_to_xyzw(self.mug_quat)).as_euler('zyx', degrees=True)[0]
        # self.grasp_mug_quat = quat_xyzw_to_wxyz((Rotation.from_euler('z', mug_yaw, degrees=True) * Rotation.from_euler('xyz', MUG_GRASP_QUAT, degrees=True)).as_quat())
        
        # Convert the mug's full quaternion to a Rotation object
        mug_rotation_full = Rotation.from_quat(quat_wxyz_to_xyzw(self.mug_quat))
        # Define the grasp rotation using the MUG_GRASP_QUAT constant
        grasp_rotation = Rotation.from_euler('xyz', C.MUG_GRASP_QUAT, degrees=True)
        # Calculate the final hand orientation
        final_hand_rotation = mug_rotation_full * grasp_rotation
        # Convert back to a quaternion for your system
        self.grasp_mug_quat = quat_xyzw_to_wxyz(final_hand_rotation.as_quat())
        
        self.init_waypoints.append(create_waypoint(self.start_right_pos, self.start_right_quat, True, self.grasp_mug_pos, self.grasp_mug_quat, False))
        

    def motion_phase(self):
        """
        Definition: Grasping the mug.
        """
        # Grasp the mug (close the left hand)
        self.motion_waypoints.append(create_waypoint(self.start_right_pos, self.start_right_quat, True, self.grasp_mug_pos, self.grasp_mug_quat, True))

    def terminal_phase(self):
        """
        Definition: Lifting the mug off the drawer.
        """
        # Lift the mug
        lift_mug_pos = self.mug_pos + C.MUG_LIFT_POS
        self.terminal_waypoints.append(create_waypoint(self.start_right_pos, self.start_right_quat, True, lift_mug_pos, self.grasp_mug_quat, True))


class PlaceMugOnMatSkill(Skill):
    """Skill to place a held mug onto a mat."""

    def init_phase(self):
        """
        Definition: Lowering the mug on the mat.
        """
        (current_left_pos, current_left_quat, current_right_eef_pos_w, current_right_eef_quat_wxyz_w,
         *_, _, _, _, _, _, _, self.mug_mat_pos, self.mug_mat_quat) = TrajectoryPlayer.extract_essential_obs_data(self.obs)

        self.start_right_pos, self.start_right_quat, self.start_left_pos, self.start_left_quat = (self.initial_poses["right_eef_pos"], self.initial_poses["right_eef_quat"], self.initial_poses["left_eef_pos"], self.initial_poses["left_eef_quat"]) if self.initial_poses else (current_right_eef_pos_w, current_right_eef_quat_wxyz_w, current_left_pos, current_left_quat)

        # Start from the pose of the last skill (left hand holding mug)
        self.init_waypoints.append(create_waypoint(self.start_right_pos, self.start_right_quat, True, self.start_left_pos, self.start_left_quat, True))

        # Lower the mug onto the mat
        self.place_mug_on_mat_pos = self.mug_mat_pos + C.MAT_PLACE_POS
        mat_yaw = Rotation.from_quat(quat_wxyz_to_xyzw(self.mug_mat_quat)).as_euler('zyx', degrees=True)[0]
        self.mug_on_mat_quat = quat_xyzw_to_wxyz((Rotation.from_euler('z', mat_yaw, degrees=True) * Rotation.from_euler('xyz', C.MAT_PLACE_ABS_QUAT, degrees=True)).as_quat())
        self.init_waypoints.append(create_waypoint(self.start_right_pos, self.start_right_quat, True, self.place_mug_on_mat_pos, self.mug_on_mat_quat, True))

    def motion_phase(self):
        """
        Definition: Releasing the mug.
        """
        # Open the left hand
        self.motion_waypoints.append(create_waypoint(self.start_right_pos, self.start_right_quat, True, self.place_mug_on_mat_pos, self.mug_on_mat_quat, False))

    def terminal_phase(self):
        """
        Definition: Lifting the hand and pushing the drawer.
        """
        # Push back the opened drawer (right EEF), and lift the left EEF away from the mug
        push_approach_pos = self.start_right_pos + C.DRAWER_PUSH_DIRECTION_OFFSET
        lift_pos = self.mug_mat_pos + C.MAT_APPROACH_POS
        self.terminal_waypoints.append(create_waypoint(push_approach_pos, self.start_right_quat, True, lift_pos, self.mug_on_mat_quat, False))


class GraspBottleSkill(Skill):
    """Skill to grasp a bottle."""

    def init_phase(self):
        """
        Definition: Approaching the bottle.
        """
        (current_left_pos, current_left_quat, current_right_eef_pos_w, current_right_eef_quat_wxyz_w,
         *_, self.bottle_pos, self.bottle_quat, _, _, _, _) = TrajectoryPlayer.extract_essential_obs_data(self.obs)

        self.start_right_pos, self.start_right_quat, self.start_left_pos, self.start_left_quat = (self.initial_poses["right_eef_pos"], self.initial_poses["right_eef_quat"], self.initial_poses["left_eef_pos"], self.initial_poses["left_eef_quat"]) if self.initial_poses else (current_right_eef_pos_w, current_right_eef_quat_wxyz_w, current_left_pos, current_left_quat)
        self.init_waypoints.append(create_waypoint(self.start_right_pos, self.start_right_quat, False, self.start_left_pos, self.start_left_quat, False))

        # Approach the bottle
        self.grasp_bottle_pos = self.bottle_pos + C.BOTTLE_GRASP_POS
        bottle_yaw = Rotation.from_quat(quat_wxyz_to_xyzw(self.bottle_quat)).as_euler('zyx', degrees=True)[0]
        self.grasp_bottle_quat = quat_xyzw_to_wxyz((Rotation.from_euler('z', bottle_yaw, degrees=True) * Rotation.from_euler('xyz', C.BOTTLE_GRASP_QUAT, degrees=True)).as_quat())
        self.init_waypoints.append(create_waypoint(self.grasp_bottle_pos, self.grasp_bottle_quat, False, self.start_left_pos, self.start_left_quat, False))

    def motion_phase(self):
        """
        Definition: Grasping the bottle.
        """
        # Grasp the bottle
        self.motion_waypoints.append(create_waypoint(self.grasp_bottle_pos, self.grasp_bottle_quat, True, self.start_left_pos, self.start_left_quat, False))

    def terminal_phase(self):
        """
        Definition: Lifting the bottle.
        """
        # Lift up the bottle
        lift_bottle_pos = self.grasp_bottle_pos + C.BOTTLE_LIFT_POS
        self.terminal_waypoints.append(create_waypoint(lift_bottle_pos, self.grasp_bottle_quat, True, self.start_left_pos, self.start_left_quat, False))


class PourBottleSkill(Skill):
    """Skill to pour a held bottle."""

    def init_phase(self):
        """
        Definition: No init_phase for this skill.
        """
        (current_left_pos, current_left_quat, current_right_eef_pos_w, current_right_eef_quat_wxyz_w,
         *_, _, _, self.mug_pos, _, _, _) = TrajectoryPlayer.extract_essential_obs_data(self.obs)
        
        self.start_right_pos, self.start_right_quat, self.start_left_pos, self.start_left_quat = (self.initial_poses["right_eef_pos"], self.initial_poses["right_eef_quat"], self.initial_poses["left_eef_pos"], self.initial_poses["left_eef_quat"]) if self.initial_poses else (current_right_eef_pos_w, current_right_eef_quat_wxyz_w, current_left_pos, current_left_quat)
        self.init_waypoints.append(create_waypoint(self.start_right_pos, self.start_right_quat, True, self.start_left_pos, self.start_left_quat, False))

    def motion_phase(self):
        """
        Definition: Pouring (inclining) the bottle.
        """
        # Pour the bottle
        pouring_pos = self.start_right_pos + C.BOTTLE_POURING_MAT_POS
        pouring_rot = Rotation.from_quat(quat_wxyz_to_xyzw(self.start_right_quat)) * Rotation.from_euler('xyz', C.BOTTLE_POURING_QUAT, degrees=True)
        pouring_quat = quat_xyzw_to_wxyz(pouring_rot.as_quat())
        self.motion_waypoints.append(create_waypoint(pouring_pos, pouring_quat, True, self.start_left_pos, self.start_left_quat, False))

    def terminal_phase(self):
        """
        Definition: Rotating the bottle back to the vertical pose.
        """
        # Restore the bottle to the vertical pose
        self.terminal_waypoints.append(create_waypoint(self.start_right_pos, self.start_right_quat, True, self.start_left_pos, self.start_left_quat, False))


class ReturnBottleSkill(Skill):
    """Skill to return a held bottle to a target location."""

    def init_phase(self):
        """
        Definition: Approaching the table (lowering the bottle).
        """
        (current_left_pos, current_left_quat, current_right_eef_pos_w, current_right_eef_quat_wxyz_w,
         *_, self.bottle_pos, _, _, _, _, _) = TrajectoryPlayer.extract_essential_obs_data(self.obs)

        self.start_right_pos, self.start_right_quat, self.start_left_pos, self.start_left_quat = (self.initial_poses["right_eef_pos"], self.initial_poses["right_eef_quat"], self.initial_poses["left_eef_pos"], self.initial_poses["left_eef_quat"]) if self.initial_poses else (current_right_eef_pos_w, current_right_eef_quat_wxyz_w, current_left_pos, current_left_quat)
        
        # Start from the pose of the last skill (right hand holding bottle vertically)
        self.init_waypoints.append(create_waypoint(self.start_right_pos, self.start_right_quat, True, self.start_left_pos, self.start_left_quat, False))

        # Lower bottle to place position
        self.place_bottle_pos = self.bottle_pos + C.BOTTLE_GRASP_POS
        self.place_bottle_quat = self.start_right_quat
        self.init_waypoints.append(create_waypoint(self.place_bottle_pos, self.place_bottle_quat, True, self.start_left_pos, self.start_left_quat, False))

    def motion_phase(self):
        """
        Definition: Release the bottle.
        """
        # Release the bottle
        self.motion_waypoints.append(create_waypoint(self.place_bottle_pos, self.place_bottle_quat, False, self.start_left_pos, self.start_left_quat, False))

    def terminal_phase(self):
        """
        Definition: Neglected in this case.
        """
        pass
