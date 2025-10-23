
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from scipy.spatial.transform import Rotation
import os

from .quaternion_utils import quat_xyzw_to_wxyz
import carb
carb_settings_iface = carb.settings.get_settings()

g1_hand_type = carb_settings_iface.get("/unitree_g1_env/hand_type")

        
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
if g1_hand_type == "trihand":
    HAND_JOINT_POSITIONS = {
        "left_hand_index_0_joint":   {"open": 0.0, "closed": -0.7},
        "left_hand_middle_0_joint":  {"open": 0.0, "closed": -0.7},
        "left_hand_thumb_0_joint":   {"open": 0.0, "closed": 0.0},

        "right_hand_index_0_joint":  {"open": 0.0, "closed": 1.0},
        "right_hand_middle_0_joint": {"open": 0.0, "closed": 1.0},
        "right_hand_thumb_0_joint":  {"open": 0.0, "closed": 0.0},

        "left_hand_index_1_joint":   {"open": 0.0, "closed": -0.7},
        "left_hand_middle_1_joint":  {"open": 0.0, "closed": -0.7},
        "left_hand_thumb_1_joint":   {"open": 0.0, "closed": 0.3},

        "right_hand_index_1_joint":  {"open": 0.0, "closed": 0.9},
        "right_hand_middle_1_joint": {"open": 0.0, "closed": 0.9},
        "right_hand_thumb_1_joint":  {"open": 0.0, "closed": -0.1},

        "left_hand_thumb_2_joint":   {"open": 0.0, "closed": 0.3},
        "right_hand_thumb_2_joint":  {"open": 0.0, "closed": -0.5},
    }
else: # elif g1_hand_type == "inspire":
    HAND_JOINT_POSITIONS = {
        "L_index_proximal_joint":       {"open": 0.0, "closed": 0.76},
        "L_middle_proximal_joint":      {"open": 0.0, "closed": 0.74},
        "L_pinky_proximal_joint":       {"open": 0.0, "closed": 1.36},
        "L_ring_proximal_joint":        {"open": 0.0, "closed": 0.95},
        "L_thumb_proximal_yaw_joint":   {"open": 0.7, "closed": 1.25},

        "R_index_proximal_joint":       {"open": 0.0, "closed": 0.78},
        "R_middle_proximal_joint":      {"open": 0.0, "closed": 0.78},
        "R_pinky_proximal_joint":       {"open": 0.0, "closed": 0.85},
        "R_ring_proximal_joint":        {"open": 0.0, "closed": 0.81},
        "R_thumb_proximal_yaw_joint":   {"open": 0.5, "closed": 1.18},
        
        "L_thumb_proximal_pitch_joint": {"open": 0.1, "closed": 0.1},
        "R_thumb_proximal_pitch_joint": {"open": 0.0, "closed": 0.0},
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
PRE_APPROACH_OFFSET_POS     = np.array([-0.16279561, -0.04801384,  0.1128001])
APPROACH_OFFSET_POS         = np.array([-0.11349986, -0.04357545,  0.1128001])  # Grasp the right side of the handle (y: -0.0357 -> -0.4357)
PULL_OFFSET_POS             = np.array([-0.31349986, -0.05500001,  0.1128001])
PRE_APPROACH_OFFSET_QUAT    = np.array([90, 50, 0])  # degrees, relative to drawer's orientation
# 2. Pick-and-place the mug (Relative to mug's origin and orientation)
MUG_APPROACH_POS            = np.array([-0.02391565,  0.105453  ,  0.15847])
MUG_LIFT_POS                = np.array([-0.02391565,  0.105453  ,  0.32347])
MUG_GRASP_QUAT              = np.array([90.61094041,  18.00920332, -31.66006655])   # -74.96883661 - (-43.30877006) = -31.66 degrees

PRE_MAT_PLACE_POS           = np.array([-0.0700,  0.0678,  0.210])  #
MAT_PLACE_POS               = np.array([-0.0700,  0.0678,  0.165])  # Enlarge the height (0.16 -> 0.165) to avoid collision with the mat
MAT_PLACE_QUAT              = np.array([89.95486601, 20.35125902, -40.08051963])

DRAWER_PUSH_DIRECTION_LOCAL = np.array([0.200, 0, 0])   # -0.120 - (-0.305) = 0.185
# 3. pick the bottle, pour it, and place it back
BOTTLE_GRASP_QUAT           = np.array([0.0, 0.0, 160])  # degrees, relative to bottle's orientation (yaw = 180 deg)
BOTTLE_GRASP_POS            = np.array([-0.13, -0.02, -0.005])
BOTTLE_LIFT_UP_OFFSET       = np.array([0.0, 0.0, 0.04]) # Relative to BOTTLE_GRASP_POS
BOTTLE_PRE_POUR_OFFSET      = np.array([-0.11000000, -0.11000000, 0.12281656])    # w.r.t. the mug's frame
# pouring process: [0.29, -0.01, 0.93, 0.9848, 0.0, 0.0, -0.1736] -> [0.29, 0.01, 0.97, 0.8859813, -0.4281835, -0.0121569, -0.1776184]
BOTTLE_POURING_OFFSET       = np.array([0.0, 0.02, 0.04]) # w.r.t BOTTLE_PRE_POUR_OFFSET
BOTTLE_POURING_QUAT         = np.array([-50, -10, 0])  # w.r.t BOTTLE_PRE_POUR_OFFSET

# === Constants for Retract and Home Skills ===
# Default home positions for the robot arms
HOME_POSES = {
    "right_pos": np.array([0.075, -0.205, 0.90]),
    "right_quat": np.array([0.7329629, 0.5624222, 0.3036032, -0.2329629]),
    "left_pos": np.array([0.075, 0.22108203, 0.950]),
    "left_quat": np.array([1.0, 0.0, 0.0, 0.0]),
}

# Standby poses for arms when they are not in use for a task
ARM_PREPARE_POSES = {
    "left_pos": np.array([0.075, 0.220, 0.950]),
    "left_quat": np.array([1.0, 0.0, 0.0, 0.0]),
}


# Intermediate waypoints for retraction to avoid obstacles
RETRACT_WAYPOINTS = {
    "right_retract_pos": np.array([0.075, -0.205, 0.90]),
    "right_retract_quat": np.array([0.7329629, 0.5624222, 0.3036032, -0.2329629]),
    "left_retract_pos": np.array([0.075, 0.22108203, 0.950]),
    "left_retract_quat": np.array([1.0, 0.0, 0.0, 0.0]),
    "right_restore_pos": np.array([0.060, -0.340, 0.90]),
    "right_restore_quat": np.array([0.9848078, 0.0, 0.0, -0.1736482]),
}

TRAJECTORY_SMALL_MOVEMENT_POS_THRESHOLD = 0.10
TRAJECTORY_SMALL_MOVEMENT_ANGLE_THRESHOLD = 45
