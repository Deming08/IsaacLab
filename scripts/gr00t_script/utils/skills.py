# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This module contains modular skill definitions based on the SkillMimicGen paradigm."""

from typing import Optional, Dict
import numpy as np
from scipy.spatial.transform import Rotation

from .constants import *
from .quaternion_utils import quat_xyzw_to_wxyz, quat_wxyz_to_xyzw
from .trajectory_player import TrajectoryPlayer


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

    def _add_waypoint(self, right_eef_pos, right_eef_quat, right_hand_closed_bool, left_eef_pos, left_eef_quat, left_hand_closed_bool):
        """Helper to append a waypoint to the recorded_waypoints list."""
        wp = {
            "left_arm_eef": np.concatenate([left_eef_pos, left_eef_quat]),
            "right_arm_eef": np.concatenate([right_eef_pos, right_eef_quat]),
            "left_hand_bool": int(left_hand_closed_bool),
            "right_hand_bool": int(right_hand_closed_bool)
        }
        return wp

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

    def get_full_trajectory(self) -> tuple[list, dict]:
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


class TransitOrTransferMotion(Skill):
    """A generic skill to move one or both arms to a target pose."""

    def __init__(self, obs: Dict, initial_poses: Optional[dict] = None, target_poses: Optional[dict] = None):
        super().__init__(obs, initial_poses)
        self.target_poses = target_poses

    def init_phase(self):
        """
        Definition: Sets the starting waypoint based on initial_poses or current observation.
        """
        (current_left_pos, current_left_quat, current_right_pos, current_right_quat, *_) = TrajectoryPlayer.extract_essential_obs_data(self.obs)
        
        if self.initial_poses:
            self.start_right_pos, self.start_right_quat = self.initial_poses["right_eef_pos"], self.initial_poses["right_eef_quat"]
            self.start_left_pos, self.start_left_quat = self.initial_poses["left_eef_pos"], self.initial_poses["left_eef_quat"]
            self.start_right_hand = self.initial_poses.get("right_hand_closed", False)
            self.start_left_hand = self.initial_poses.get("left_hand_closed", False)
        else:
            self.start_right_pos, self.start_right_quat = current_right_pos, current_right_quat
            self.start_left_pos, self.start_left_quat = current_left_pos, current_left_quat
            self.start_right_hand, self.start_left_hand = False, False
            
        self.init_waypoints.append(self._add_waypoint(self.start_right_pos, self.start_right_quat, self.start_right_hand, self.start_left_pos, self.start_left_quat, self.start_left_hand))

    def motion_phase(self):
        """
        Definition: Generates the waypoint to move to the target pose.
        If a target for an arm is not provided, it remains at its starting pose.
        """
        if not self.target_poses:
            return

        # Default to start poses if no target is given for an arm
        target_right_pos = self.target_poses.get("right_pos", self.start_right_pos)
        target_right_quat = self.target_poses.get("right_quat", self.start_right_quat)
        target_left_pos = self.target_poses.get("left_pos", self.start_left_pos)
        target_left_quat = self.target_poses.get("left_quat", self.start_left_quat)
        
        # Gripper state is preserved from start unless specified in target
        target_right_hand = self.target_poses.get("right_hand_closed", self.start_right_hand)
        target_left_hand = self.target_poses.get("left_hand_closed", self.start_left_hand)

        self.motion_waypoints.append(self._add_waypoint(target_right_pos, target_right_quat, target_right_hand, target_left_pos, target_left_quat, target_left_hand))

    def terminal_phase(self):
        """
        Definition: No action is taken in the terminal phase for a simple move.
        """
        pass


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
        self.approach_handle_quat = quat_xyzw_to_wxyz((self.R_world_drawer * Rotation.from_euler('xyz', PRE_APPROACH_OFFSET_QUAT, degrees=True)).as_quat())
        
        # Add the starting pre-approach waypoint
        self.init_waypoints.append(self._add_waypoint(start_poses["right_eef_pos"], start_poses["right_eef_quat"], False, start_poses["left_eef_pos"], start_poses["left_eef_quat"], False))

        # Move to the drawing pose
        self.approach_handle_pos = drawer_pos + self.R_world_drawer.apply(APPROACH_OFFSET_POS)
        self.init_waypoints.append(self._add_waypoint(self.approach_handle_pos, self.approach_handle_quat, False, start_poses["left_eef_pos"], start_poses["left_eef_quat"], False))

        self.drawer_pos = drawer_pos
        self.prepare_left_pos = start_poses["left_eef_pos"]
        self.prepare_left_quat = start_poses["left_eef_quat"]

    def motion_phase(self):
        """
        Definition: Grasping the drawer.
        """
        # Grasp the drawer handle
        self.motion_waypoints.append(self._add_waypoint(self.approach_handle_pos, self.approach_handle_quat, True, self.prepare_left_pos, self.prepare_left_quat, False))

    def terminal_phase(self):
        """
        Definition: Pull out the drawer completely.
        """
        # Pull out the drawer handle
        pulled_handle_pos = self.drawer_pos + self.R_world_drawer.apply(PULL_OFFSET_POS)
        self.terminal_waypoints.append(self._add_waypoint(pulled_handle_pos, self.approach_handle_quat, True, self.prepare_left_pos, self.prepare_left_quat, False))


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
        self.init_waypoints.append(self._add_waypoint(self.start_right_pos, self.start_right_quat, False, self.start_left_pos, self.start_left_quat, False))

        # Approach the mug (left hand)
        self.approach_mug_pos = self.mug_pos + MUG_APPROACH_POS
        mug_yaw = Rotation.from_quat(quat_wxyz_to_xyzw(self.mug_quat)).as_euler('zyx', degrees=True)[0]
        self.grasp_mug_quat = quat_xyzw_to_wxyz((Rotation.from_euler('z', mug_yaw, degrees=True) * Rotation.from_euler('xyz', MUG_GRASP_QUAT, degrees=True)).as_quat())
        self.init_waypoints.append(self._add_waypoint(self.start_right_pos, self.start_right_quat, False, self.approach_mug_pos, self.grasp_mug_quat, False))

    def motion_phase(self):
        """
        Definition: Grasping the mug.
        """
        # Grasp the mug (close the left hand)
        self.motion_waypoints.append(self._add_waypoint(self.start_right_pos, self.start_right_quat, False, self.approach_mug_pos, self.grasp_mug_quat, True))

    def terminal_phase(self):
        """
        Definition: Lifting the mug off the drawer.
        """
        # Lift the mug
        lift_mug_pos = self.mug_pos + MUG_LIFT_POS
        self.terminal_waypoints.append(self._add_waypoint(self.start_right_pos, self.start_right_quat, False, lift_mug_pos, self.grasp_mug_quat, True))


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
        self.init_waypoints.append(self._add_waypoint(self.start_right_pos, self.start_right_quat, False, self.start_left_pos, self.start_left_quat, True))

        # Lower the mug onto the mat
        self.place_mug_on_mat_pos = self.mug_mat_pos + MAT_PLACE_POS
        mat_yaw = Rotation.from_quat(quat_wxyz_to_xyzw(self.mug_mat_quat)).as_euler('zyx', degrees=True)[0]
        self.mug_on_mat_quat = quat_xyzw_to_wxyz((Rotation.from_euler('z', mat_yaw, degrees=True) * Rotation.from_euler('xyz', MAT_PLACE_QUAT, degrees=True)).as_quat())
        self.init_waypoints.append(self._add_waypoint(self.start_right_pos, self.start_right_quat, False, self.place_mug_on_mat_pos, self.mug_on_mat_quat, True))

    def motion_phase(self):
        """
        Definition: Releasing the mug.
        """
        # Open the left hand
        self.motion_waypoints.append(self._add_waypoint(self.start_right_pos, self.start_right_quat, False, self.place_mug_on_mat_pos, self.mug_on_mat_quat, False))

    def terminal_phase(self):
        """
        Definition: Lifting the hand and pushing the drawer.
        """
        # Push back the opened drawer (right EEF), and lift the left EEF away from the mug
        push_approach_pos = self.start_right_pos + DRAWER_PUSH_DIRECTION_LOCAL
        lift_pos = self.mug_mat_pos + PRE_MAT_PLACE_POS
        self.terminal_waypoints.append(self._add_waypoint(push_approach_pos, self.start_right_quat, False, lift_pos, self.mug_on_mat_quat, False))


class GraspBottleSkill(Skill):
    """Skill to grasp a bottle."""

    def init_phase(self):
        """
        Definition: Approaching the bottle.
        """
        (current_left_pos, current_left_quat, current_right_eef_pos_w, current_right_eef_quat_wxyz_w,
         *_, self.bottle_pos, self.bottle_quat, _, _, _, _) = TrajectoryPlayer.extract_essential_obs_data(self.obs)

        self.start_right_pos, self.start_right_quat, self.start_left_pos, self.start_left_quat = (self.initial_poses["right_eef_pos"], self.initial_poses["right_eef_quat"], self.initial_poses["left_eef_pos"], self.initial_poses["left_eef_quat"]) if self.initial_poses else (current_right_eef_pos_w, current_right_eef_quat_wxyz_w, current_left_pos, current_left_quat)
        self.init_waypoints.append(self._add_waypoint(self.start_right_pos, self.start_right_quat, False, self.start_left_pos, self.start_left_quat, False))

        # Approach the bottle
        self.grasp_bottle_pos = self.bottle_pos + BOTTLE_GRASP_POS
        bottle_yaw = Rotation.from_quat(quat_wxyz_to_xyzw(self.bottle_quat)).as_euler('zyx', degrees=True)[0]
        self.grasp_bottle_quat = quat_xyzw_to_wxyz((Rotation.from_euler('z', bottle_yaw, degrees=True) * Rotation.from_euler('xyz', BOTTLE_GRASP_QUAT, degrees=True)).as_quat())
        self.init_waypoints.append(self._add_waypoint(self.grasp_bottle_pos, self.grasp_bottle_quat, False, self.start_left_pos, self.start_left_quat, False))

    def motion_phase(self):
        """
        Definition: Grasping the bottle.
        """
        # Grasp the bottle
        self.motion_waypoints.append(self._add_waypoint(self.grasp_bottle_pos, self.grasp_bottle_quat, True, self.start_left_pos, self.start_left_quat, False))

    def terminal_phase(self):
        """
        Definition: Lifting the bottle.
        """
        # Lift up the bottle
        lift_bottle_pos = self.grasp_bottle_pos + BOTTLE_LIFT_UP_OFFSET
        self.terminal_waypoints.append(self._add_waypoint(lift_bottle_pos, self.grasp_bottle_quat, True, self.start_left_pos, self.start_left_quat, False))


class PourBottleSkill(Skill):
    """Skill to pour a held bottle."""

    def init_phase(self):
        """
        Definition: No init_phase for this skill.
        """
        (current_left_pos, current_left_quat, current_right_eef_pos_w, current_right_eef_quat_wxyz_w,
         *_, _, _, self.mug_pos, _, _, _) = TrajectoryPlayer.extract_essential_obs_data(self.obs)
        
        self.start_right_pos, self.start_right_quat, self.start_left_pos, self.start_left_quat = (self.initial_poses["right_eef_pos"], self.initial_poses["right_eef_quat"], self.initial_poses["left_eef_pos"], self.initial_poses["left_eef_quat"]) if self.initial_poses else (current_right_eef_pos_w, current_right_eef_quat_wxyz_w, current_left_pos, current_left_quat)
        self.init_waypoints.append(self._add_waypoint(self.start_right_pos, self.start_right_quat, True, self.start_left_pos, self.start_left_quat, False))

    def motion_phase(self):
        """
        Definition: Pouring (inclining) the bottle.
        """
        # Pour the bottle
        pouring_pos = self.start_right_pos + BOTTLE_POURING_OFFSET
        pouring_rot = Rotation.from_quat(quat_wxyz_to_xyzw(self.start_right_quat)) * Rotation.from_euler('xyz', BOTTLE_POURING_QUAT, degrees=True)
        pouring_quat = quat_xyzw_to_wxyz(pouring_rot.as_quat())
        self.motion_waypoints.append(self._add_waypoint(pouring_pos, pouring_quat, True, self.start_left_pos, self.start_left_quat, False))

    def terminal_phase(self):
        """
        Definition: Rotating the bottle back to the vertical pose.
        """
        # Restore the bottle to the vertical pose
        self.terminal_waypoints.append(self._add_waypoint(self.start_right_pos, self.start_right_quat, True, self.start_left_pos, self.start_left_quat, False))


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
        self.init_waypoints.append(self._add_waypoint(self.start_right_pos, self.start_right_quat, True, self.start_left_pos, self.start_left_quat, False))

        # Lower bottle to place position
        self.place_bottle_pos = self.bottle_pos + BOTTLE_GRASP_POS
        self.place_bottle_quat = self.start_right_quat
        self.init_waypoints.append(self._add_waypoint(self.place_bottle_pos, self.place_bottle_quat, True, self.start_left_pos, self.start_left_quat, False))

    def motion_phase(self):
        """
        Definition: Release the bottle.
        """
        # Release the bottle
        self.motion_waypoints.append(self._add_waypoint(self.place_bottle_pos, self.place_bottle_quat, False, self.start_left_pos, self.start_left_quat, False))

    def terminal_phase(self):
        """
        Definition: Neglected in this case.
        """
        pass



class HomeSkill(Skill):
    """Skill to return both arms to their neutral home positions."""

    def init_phase(self):
        """
        Definition: Set the starting waypoint from the current or previous skill's final pose.
        """
        (current_left_pos, current_left_quat, current_right_eef_pos_w, current_right_eef_quat_wxyz_w,
         *_) = TrajectoryPlayer.extract_essential_obs_data(self.obs)

        if self.initial_poses:
            self.start_right_pos = self.initial_poses["right_eef_pos"]
            self.start_right_quat = self.initial_poses["right_eef_quat"]
            self.start_left_pos = self.initial_poses["left_eef_pos"]
            self.start_left_quat = self.initial_poses["left_eef_quat"]
            left_hand_closed = self.initial_poses.get("left_hand_closed", False)
            right_hand_closed = self.initial_poses.get("right_hand_closed", False)
        else:
            self.start_right_pos = current_right_eef_pos_w
            self.start_right_quat = current_right_eef_quat_wxyz_w
            self.start_left_pos = current_left_pos
            self.start_left_quat = current_left_quat
            left_hand_closed = False
            right_hand_closed = False

        self.init_waypoints.append(self._add_waypoint(self.start_right_pos, self.start_right_quat, right_hand_closed, self.start_left_pos, self.start_left_quat, left_hand_closed))

    def motion_phase(self):
        """
        Definition: Move both arms to their predefined home positions.
        """
        # Use home_poses from initial_poses if available, otherwise use default HOME_POSES
        home_poses = (self.initial_poses or {}).get("home_poses", HOME_POSES)
        self.motion_waypoints.append(self._add_waypoint(home_poses["right_pos"], home_poses["right_quat"], False, home_poses["left_pos"], home_poses["left_quat"], False))

    def terminal_phase(self):
        """
        Definition: No action, as this is a final skill in a sequence.
        """
        pass


class RetractSkill(Skill):
    """Skill to retract arms to specific intermediate poses."""

    def init_phase(self):
        """
        Definition: Set the starting waypoint from the current or previous skill's final pose.
        """
        (current_left_pos, current_left_quat, current_right_eef_pos_w, current_right_eef_quat_wxyz_w,
         *_) = TrajectoryPlayer.extract_essential_obs_data(self.obs)

        if self.initial_poses:
            self.start_right_pos = self.initial_poses["right_eef_pos"]
            self.start_right_quat = self.initial_poses["right_eef_quat"]
            self.start_left_pos = self.initial_poses["left_eef_pos"]
            self.start_left_quat = self.initial_poses["left_eef_quat"]
            left_hand_closed = self.initial_poses.get("left_hand_closed", False)
            right_hand_closed = self.initial_poses.get("right_hand_closed", False)
        else:
            self.start_right_pos = current_right_eef_pos_w
            self.start_right_quat = current_right_eef_quat_wxyz_w
            self.start_left_pos = current_left_pos
            self.start_left_quat = current_left_quat
            left_hand_closed = False
            right_hand_closed = False

        self.init_waypoints.append(self._add_waypoint(self.start_right_pos, self.start_right_quat, right_hand_closed, self.start_left_pos, self.start_left_quat, left_hand_closed))

    def motion_phase(self):
        """
        Definition: Move arms through a series of predefined retraction waypoints.
        """
        # Restore arms to specific intermediate poses from constants
        self.motion_waypoints.append(self._add_waypoint(
            RETRACT_WAYPOINTS["right_retract_pos"], RETRACT_WAYPOINTS["right_retract_quat"], False,
            RETRACT_WAYPOINTS["left_retract_pos"], RETRACT_WAYPOINTS["left_retract_quat"], False
        ))
        self.motion_waypoints.append(self._add_waypoint(
            RETRACT_WAYPOINTS["right_restore_pos"], RETRACT_WAYPOINTS["right_restore_quat"], False,
            RETRACT_WAYPOINTS["left_retract_pos"], RETRACT_WAYPOINTS["left_retract_quat"], False
        ))

    def terminal_phase(self):
        """
        Definition: No action, as this is typically a final skill in a sub-task.
        """
        pass