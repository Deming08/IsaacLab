# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This module contains modular skill definitions based on the SkillMimicGen paradigm."""

from typing import Optional, List, Dict
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
        """Generates waypoints for the initial phase (e.g., approach)."""
        raise NotImplementedError

    def motion_phase(self):
        """Generates waypoints for the core motion phase (e.g., interact)."""
        raise NotImplementedError

    def terminal_phase(self):
        """Generates waypoints for the terminal phase (e.g., retract)."""
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


class OpenDrawerSkill(Skill):
    """Skill to open a drawer."""

    def init_phase(self):
        (initial_left_pos, initial_left_quat, current_right_eef_pos_w, current_right_eef_quat_wxyz_w,
         *_, drawer_pos, drawer_quat, _, _, _, _, _, _) = TrajectoryPlayer.extract_essential_obs_data(self.obs)

        start_right_pos, start_right_quat, start_left_pos, start_left_quat = (self.initial_poses["right_eef_pos"], self.initial_poses["right_eef_quat"], self.initial_poses["left_eef_pos"], self.initial_poses["left_eef_quat"]) if self.initial_poses else (current_right_eef_pos_w, current_right_eef_quat_wxyz_w, initial_left_pos, initial_left_quat)
        
        self.init_waypoints.append(self._add_waypoint(start_right_pos, start_right_quat, False, start_left_pos, start_left_quat, False))

        R_world_drawer = Rotation.from_quat(quat_wxyz_to_xyzw(drawer_quat))
        
        prepare_left_pos, prepare_left_quat = np.array([0.075, 0.220, 0.950]), np.array([1.0, 0.0, 0.0, 0.0])
        pre_approach_handle_pos = drawer_pos + R_world_drawer.apply(PRE_APPROACH_OFFSET_POS)
        self.approach_handle_quat = quat_xyzw_to_wxyz((R_world_drawer * Rotation.from_euler('xyz', PRE_APPROACH_OFFSET_QUAT, degrees=True)).as_quat())
        
        self.init_waypoints.append(self._add_waypoint(pre_approach_handle_pos, self.approach_handle_quat, False, prepare_left_pos, prepare_left_quat, False))
        
        self.drawer_pos = drawer_pos
        self.R_world_drawer = R_world_drawer
        self.prepare_left_pos = prepare_left_pos
        self.prepare_left_quat = prepare_left_quat

    def motion_phase(self):
        # Approach the drawer handle
        approach_handle_pos = self.drawer_pos + self.R_world_drawer.apply(APPROACH_OFFSET_POS)
        self.motion_waypoints.append(self._add_waypoint(approach_handle_pos, self.approach_handle_quat, False, self.prepare_left_pos, self.prepare_left_quat, False))
        
        # Grasp the drawer handle
        self.motion_waypoints.append(self._add_waypoint(approach_handle_pos, self.approach_handle_quat, True, self.prepare_left_pos, self.prepare_left_quat, False))

    def terminal_phase(self):
        # Pull out the drawer handle
        self.pulled_handle_pos = self.drawer_pos + self.R_world_drawer.apply(PULL_OFFSET_POS)
        self.motion_waypoints.append(self._add_waypoint(self.pulled_handle_pos, self.approach_handle_quat, True, self.prepare_left_pos, self.prepare_left_quat, False))


class PickMugFromDrawerSkill(Skill):
    """Skill to pick a mug from inside a drawer."""

    def init_phase(self):
        (current_left_pos, current_left_quat, current_right_eef_pos_w, current_right_eef_quat_wxyz_w,
         *_, _, _, _, _, self.mug_pos, self.mug_quat, _, _) = TrajectoryPlayer.extract_essential_obs_data(self.obs)

        start_right_pos, start_right_quat, start_left_pos, start_left_quat = (self.initial_poses["right_eef_pos"], self.initial_poses["right_eef_quat"], self.initial_poses["left_eef_pos"], self.initial_poses["left_eef_quat"]) if self.initial_poses else (current_right_eef_pos_w, current_right_eef_quat_wxyz_w, current_left_pos, current_left_quat)
        
        # Start from the pose from the previous skill (right hand on drawer handle)
        self.init_waypoints.append(self._add_waypoint(start_right_pos, start_right_quat, True, start_left_pos, start_left_quat, False))

        # Move the left EEF to a prepared position and release right hand
        pre_grasp_mug_pos = self.mug_pos + MUG_PRE_GRASP_POS
        mug_yaw = Rotation.from_quat(quat_wxyz_to_xyzw(self.mug_quat)).as_euler('zyx', degrees=True)[0]
        self.grasp_mug_quat = quat_xyzw_to_wxyz((Rotation.from_euler('z', mug_yaw, degrees=True) * Rotation.from_euler('xyz', MUG_GRASP_QUAT, degrees=True)).as_quat())
        self.init_waypoints.append(self._add_waypoint(start_right_pos, start_right_quat, False, pre_grasp_mug_pos, self.grasp_mug_quat, False))

        self.start_right_pos = start_right_pos
        self.start_right_quat = start_right_quat

    def motion_phase(self):
        # Approach the mug (left hand)
        approach_mug_pos = self.mug_pos + MUG_APPROACH_POS
        self.motion_waypoints.append(self._add_waypoint(self.start_right_pos, self.start_right_quat, False, approach_mug_pos, self.grasp_mug_quat, False))
        # Grasp the mug (close the left hand)
        self.motion_waypoints.append(self._add_waypoint(self.start_right_pos, self.start_right_quat, False, approach_mug_pos, self.grasp_mug_quat, True))

    def terminal_phase(self):
        # Lift the mug
        lift_mug_pos = self.mug_pos + MUG_LIFT_POS
        self.motion_waypoints.append(self._add_waypoint(self.start_right_pos, self.start_right_quat, False, lift_mug_pos, self.grasp_mug_quat, True))


class PlaceMugOnMatSkill(Skill):
    """Skill to place a held mug onto a mat and close the drawer."""

    def init_phase(self):
        (current_left_pos, current_left_quat, current_right_eef_pos_w, current_right_eef_quat_wxyz_w,
         *_, _, _, _, _, _, _, self.mug_mat_pos, self.mug_mat_quat) = TrajectoryPlayer.extract_essential_obs_data(self.obs)

        self.start_right_pos, self.start_right_quat, self.start_left_pos, self.start_left_quat = (self.initial_poses["right_eef_pos"], self.initial_poses["right_eef_quat"], self.initial_poses["left_eef_pos"], self.initial_poses["left_eef_quat"]) if self.initial_poses else (current_right_eef_pos_w, current_right_eef_quat_wxyz_w, current_left_pos, current_left_quat)

        # Start from the pose of the last skill (left hand holding mug)
        self.init_waypoints.append(self._add_waypoint(self.start_right_pos, self.start_right_quat, False, self.start_left_pos, self.start_left_quat, True))

        # 2.5.1 Approach the mug mat upper 0.035 m - with respect to the mat (Hands closed)
        self.pre_mug_on_mat_pos = self.mug_mat_pos + PRE_MAT_PLACE_POS
        mat_yaw = Rotation.from_quat(quat_wxyz_to_xyzw(self.mug_mat_quat)).as_euler('zyx', degrees=True)[0]
        self.mug_on_mat_quat = quat_xyzw_to_wxyz((Rotation.from_euler('z', mat_yaw, degrees=True) * Rotation.from_euler('xyz', MAT_PLACE_QUAT, degrees=True)).as_quat())
        self.init_waypoints.append(self._add_waypoint(self.start_right_pos, self.start_right_quat, False, self.pre_mug_on_mat_pos, self.mug_on_mat_quat, True))

    def motion_phase(self):
        # 2.5.2 Place on the mug mat
        place_mug_on_mat_pos = self.mug_mat_pos + MAT_PLACE_POS
        self.motion_waypoints.append(self._add_waypoint(self.start_right_pos, self.start_right_quat, False, place_mug_on_mat_pos, self.mug_on_mat_quat, True))
        # 2.6. Place the mug on the mug mat (Open the left hand)
        self.motion_waypoints.append(self._add_waypoint(self.start_right_pos, self.start_right_quat, False, place_mug_on_mat_pos, self.mug_on_mat_quat, False))

    def terminal_phase(self):
        # 2.7. Push back the opened drawer (right EEF), and lift the left EEF away from the mug
        push_approach_pos = self.start_right_pos + DRAWER_PUSH_DIRECTION_LOCAL
        self.terminal_waypoints.append(self._add_waypoint(push_approach_pos, self.start_right_quat, False, self.pre_mug_on_mat_pos, self.mug_on_mat_quat, False))


class PourBottleSkill(Skill):
    """Skill to pick up a bottle and pour it into a mug."""

    def init_phase(self):
        (current_left_pos, current_left_quat, current_right_eef_pos_w, current_right_eef_quat_wxyz_w,
         *_, self.bottle_pos, self.bottle_quat, self.mug_pos, self.mug_quat, _, _) = TrajectoryPlayer.extract_essential_obs_data(self.obs)

        self.start_right_pos, self.start_right_quat, self.start_left_pos, self.start_left_quat = (self.initial_poses["right_eef_pos"], self.initial_poses["right_eef_quat"], self.initial_poses["left_eef_pos"], self.initial_poses["left_eef_quat"]) if self.initial_poses else (current_right_eef_pos_w, current_right_eef_quat_wxyz_w, current_left_pos, current_left_quat)
        self.init_waypoints.append(self._add_waypoint(self.start_right_pos, self.start_right_quat, False, self.start_left_pos, self.start_left_quat, False))

        # 3.1. Approach the bottle - with respect to the bottle (Hands open)
        self.grasp_bottle_pos = self.bottle_pos + BOTTLE_GRASP_POS
        bottle_yaw = Rotation.from_quat(quat_wxyz_to_xyzw(self.bottle_quat)).as_euler('zyx', degrees=True)[0]
        self.grasp_bottle_quat = quat_xyzw_to_wxyz((Rotation.from_euler('z', bottle_yaw, degrees=True) * Rotation.from_euler('xyz', BOTTLE_GRASP_QUAT, degrees=True)).as_quat())
        self.init_waypoints.append(self._add_waypoint(self.grasp_bottle_pos, self.grasp_bottle_quat, False, self.start_left_pos, self.start_left_quat, False))

    def motion_phase(self):
        # 3.2. Grasp the bottle (Close the right hand)
        self.motion_waypoints.append(self._add_waypoint(self.grasp_bottle_pos, self.grasp_bottle_quat, True, self.start_left_pos, self.start_left_quat, False))

        # 3.3. Lift up the bottle
        self.lift_bottle_pos = self.grasp_bottle_pos + BOTTLE_LIFT_UP_OFFSET
        self.motion_waypoints.append(self._add_waypoint(self.lift_bottle_pos, self.grasp_bottle_quat, True, self.start_left_pos, self.start_left_quat, False))

        # 3.4. Move the bottle toward the mug for pouring
        self.pre_pour_pos = self.mug_pos + BOTTLE_PRE_POUR_OFFSET
        self.motion_waypoints.append(self._add_waypoint(self.pre_pour_pos, self.grasp_bottle_quat, True, self.start_left_pos, self.start_left_quat, False))

        # 3.5. Pour the bottle rotation in x- and y-axis w.r.t pre_pour_pos
        pouring_pos = self.pre_pour_pos + BOTTLE_POURING_OFFSET
        pouring_rot = Rotation.from_quat(quat_wxyz_to_xyzw(self.grasp_bottle_quat)) * Rotation.from_euler('xyz', BOTTLE_POURING_QUAT, degrees=True)
        pouring_quat = quat_xyzw_to_wxyz(pouring_rot.as_quat())
        self.motion_waypoints.append(self._add_waypoint(pouring_pos, pouring_quat, True, self.start_left_pos, self.start_left_quat, False))

    def terminal_phase(self):
        # 3.6. Restore the bottle to the vertical pose
        self.terminal_waypoints.append(self._add_waypoint(self.pre_pour_pos, self.grasp_bottle_quat, True, self.start_left_pos, self.start_left_quat, False))


class ReturnBottleSkill(Skill):
    """Skill to return a held bottle to its original location."""

    def init_phase(self):
        (current_left_pos, current_left_quat, current_right_eef_pos_w, current_right_eef_quat_wxyz_w,
         *_, self.bottle_pos, _, _, _, _, _) = TrajectoryPlayer.extract_essential_obs_data(self.obs)

        self.start_right_pos, self.start_right_quat, self.start_left_pos, self.start_left_quat = (self.initial_poses["right_eef_pos"], self.initial_poses["right_eef_quat"], self.initial_poses["left_eef_pos"], self.initial_poses["left_eef_quat"]) if self.initial_poses else (current_right_eef_pos_w, current_right_eef_quat_wxyz_w, current_left_pos, current_left_quat)
        
        # Start from the pose of the last skill (right hand holding bottle vertically)
        self.init_waypoints.append(self._add_waypoint(self.start_right_pos, self.start_right_quat, True, self.start_left_pos, self.start_left_quat, False))

        # 3.7. Shift the bottle back to the original position with 5-cm height
        self.lift_bottle_pos = self.bottle_pos + BOTTLE_GRASP_POS + BOTTLE_LIFT_UP_OFFSET
        self.grasp_bottle_quat = self.start_right_quat # Assuming orientation doesn't change for the return
        self.init_waypoints.append(self._add_waypoint(self.lift_bottle_pos, self.grasp_bottle_quat, True, self.start_left_pos, self.start_left_quat, False))

    def motion_phase(self):
        # 3.8. Place the bottle back to the original position
        grasp_bottle_pos = self.bottle_pos + BOTTLE_GRASP_POS
        self.motion_waypoints.append(self._add_waypoint(grasp_bottle_pos, self.grasp_bottle_quat, True, self.start_left_pos, self.start_left_quat, False))
        
        # 3.9. Release the bottle
        self.motion_waypoints.append(self._add_waypoint(grasp_bottle_pos, self.grasp_bottle_quat, False, self.start_left_pos, self.start_left_quat, False))

    def terminal_phase(self):
        # The "return to home" logic has been extracted to ReturnToHomeSkill.
        pass


class ReturnToHomeSkill(Skill):
    """Skill to return both arms to their neutral home positions."""

    def init_phase(self):
        (current_left_pos, current_left_quat, current_right_eef_pos_w, current_right_eef_quat_wxyz_w,
         *_) = TrajectoryPlayer.extract_essential_obs_data(self.obs)

        if self.initial_poses:
            self.start_right_pos = self.initial_poses["right_eef_pos"]
            self.start_right_quat = self.initial_poses["right_eef_quat"]
            self.start_left_pos = self.initial_poses["left_eef_pos"]
            self.start_left_quat = self.initial_poses["left_eef_quat"]
            # Get gripper states from the final pose of the previous skill
            left_hand_closed = self.initial_poses.get("left_hand_closed", False)
            right_hand_closed = self.initial_poses.get("right_hand_closed", False)
        else:
            self.start_right_pos = current_right_eef_pos_w
            self.start_right_quat = current_right_eef_quat_wxyz_w
            self.start_left_pos = current_left_pos
            self.start_left_quat = current_left_quat
            # If no initial poses, assume grippers are open
            left_hand_closed = False
            right_hand_closed = False

        self.init_waypoints.append(self._add_waypoint(self.start_right_pos, self.start_right_quat, right_hand_closed, self.start_left_pos, self.start_left_quat, left_hand_closed))

    def motion_phase(self):
        # The home_poses are expected to be provided in the initial_poses dictionary.
        # These are the neutral or reset positions for the robot's arms.
        if self.initial_poses and "home_poses" in self.initial_poses:
             home_poses = self.initial_poses["home_poses"]
             # Retract arms to home poses, ensuring grippers are open.
             self.motion_waypoints.append(self._add_waypoint(home_poses["right_pos"], home_poses["right_quat"], False, home_poses["left_pos"], home_poses["left_quat"], False))
        else:
            # If no home poses are provided, we might want to log a warning or do nothing.
            # For now, we do nothing.
            pass

    def terminal_phase(self):
        # No terminal phase for this skill, it is a final action.
        pass


class RetractFromPlacementSkill(Skill):
    """Skill to retract arms to specific poses after placing the mug."""

    def init_phase(self):
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
        # Restore arms to specific intermediate poses (originally from PlaceMugOnMatSkill)
        # Restore the left EEF to the original pose (fixed, given poses)
        right_retract_pos, right_retract_quat = np.array([0.075, -0.205, 0.90]), [0.7329629, 0.5624222, 0.3036032, -0.2329629]
        left_retract_pos, left_retract_quat = np.array([0.075, 0.22108203, 0.950]), [1.0, 0.0, 0.0, 0.0]
        self.motion_waypoints.append(self._add_waypoint(right_retract_pos, right_retract_quat, False, left_retract_pos, left_retract_quat, False))

        # Restore the right EEF to a middle waypoint
        right_restore_pos, right_restore_quat = np.array([0.060, -0.340, 0.90]), np.array([0.9848078, 0.0, 0.0, -0.1736482])
        self.motion_waypoints.append(self._add_waypoint(right_restore_pos, right_restore_quat, False, left_retract_pos, left_retract_quat, False))

    def terminal_phase(self):
        pass

