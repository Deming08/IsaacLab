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
        
        final_wp = self.waypoints[-1]
        final_poses = {
            "left_eef_pos": final_wp["left_arm_eef"][:3],
            "left_eef_quat": final_wp["left_arm_eef"][3:7],
            "right_eef_pos": final_wp["right_arm_eef"][:3],
            "right_eef_quat": final_wp["right_arm_eef"][3:7]
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

        # Pull out the drawer handle
        self.pulled_handle_pos = self.drawer_pos + self.R_world_drawer.apply(PULL_OFFSET_POS)
        self.motion_waypoints.append(self._add_waypoint(self.pulled_handle_pos, self.approach_handle_quat, True, self.prepare_left_pos, self.prepare_left_quat, False))

    def terminal_phase(self):
        pass

