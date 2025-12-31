
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import os

# === Constants for OpenArm Trajectory Generation ===
DEFAULT_LEFT_HAND_BOOL = False  # False for open
TRAJECTORY_SMALL_MOVEMENT_POS_THRESHOLD = 0.10
TRAJECTORY_SMALL_MOVEMENT_ANGLE_THRESHOLD = 45

# Default home positions for the robot arms
HOME_POSES = {
    "left_eef_pos": np.array([-0.05000006, 0.15349832, 0.71199965]), "left_eef_quat": np.array([0.70699996, -0.00000123, 0.70700002, -0.00000148]),
    "right_eef_pos": np.array([0.05000068, -0.32000595, 1.04999268]), "right_eef_quat": np.array([0.99984908, 0.00004743, -0.00002597, -0.00003224]),  # 0.00, 0.00, -20.00
    "right_hand_closed": False,
    "left_hand_closed": False
}

# Standby poses for arms when they are not in use for a task
ARM_PREPARE_POSES = {
    "left_pos": np.array([0.01, 0.220, 0.950]),
    "left_quat": np.array([1.0, 0.0, 0.0, 0.0]),
}

# Intermediate waypoints for retraction to avoid obstacles
RETRACT_WAYPOINTS = {
    "right_retract_pos": np.array([0.0, -0.21160968, 0.8]),
    "right_retract_quat": np.array([0.64719629, 0.67876124, 0.24754274, -0.24319324]),
    "left_retract_pos": np.array([0.075, 0.22108203, 0.950]),
    "left_retract_quat": np.array([1.0, 0.0, 0.0, 0.0]),
    "right_restore_pos": np.array([0.0, -0.300, 0.9]),
    "right_restore_quat": np.array([0.9848078, 0.0, 0.0, -0.1736482]),
}

# Define joint positions for open and closed states (Left/right hand joint positions are opposite.)
# Using a leading underscore to indicate it's intended for internal use within this module.
HAND_JOINT_POSITIONS = {
    "index_mcp_forward":    {"open": 0.0, "closed": 1.46},
    "middle_mcp_forward":   {"open": 0.0, "closed": 1.46},
    "ring_mcp_forward":     {"open": 0.0, "closed": 1.46},
    "thumb_mcp_side":       {"open": 1.57, "closed": 1.7},
    "index_mcp_side":       {"open": 0.0, "closed": 0.0},
    "middle_mcp_side":      {"open": 0.0, "closed": 0.0},
    "ring_mcp_side":        {"open": 0.0, "closed": 0.0},
    "thumb_mcp_forward":    {"open": 0.0, "closed": 0.0},
    "index_pip":            {"open": 0.0, "closed": 0.174},
    "middle_pip":           {"open": 0.0, "closed": 0.174},
    "ring_pip":             {"open": 0.0, "closed": 0.174},
    "thumb_pip_joint":      {"open": 0.0, "closed": 0.436},
    "index_dip":            {"open": 0.0, "closed": 0.52},
    "middle_dip":           {"open": 0.0, "closed": 0.52},
    "ring_dip":             {"open": 0.0, "closed": 0.52},
    "thumb_dip_joint":      {"open": 0.0, "closed": 0.17},
}

# Default paths for saving waypoints and joint tracking logs
WAYPOINTS_JSON_PATH = os.path.join("logs", "teleoperation", "waypoints.json")
JOINT_TRACKING_LOG_PATH = os.path.join("logs", "teleoperation", "joint_tracking_log.json")

# === Constants for Pick-and-Place tasks ===
# TODO: Calculate these positions based on object pose and recorded data in @can_sorting_waypoints_openarm_leaphand.yaml
CAN_GRASP_POS           = np.array([-0.13639741, -0.09227632, 0.03803611])  # relative to can center
CAN_GRASP_QUAT          = np.array([0.004, -0.004, -0.024])  # degrees, relative to can's orientation
CAN_APPROACH_OFFSET_POS = np.array([-0.05, -0.05, 0.00])  # relative to CAN_GRASP_POS
CAN_LEAVE_OFFSET_POS    = np.array([ 0.00,  0.00, 0.05])  # relative to CAN_GRASP_POS

BASKET_PLACE_POS            = np.array([-0.16004323, -0.10493181, 0.13485849])  # relative to basket center
BASKET_PLACE_QUAT           = np.array([-0.010, 0.057, -0.026])  # degrees, absolute orientation
BASKET_APPROACH_OFFSET_POS  = np.array([ 0.00,  0.00, 0.05])  # relative to BASKET_PLACE_POS
BASKET_LEAVE_OFFSET_POS     = np.array([-0.02, -0.02, 0.05])


# === Constants for Cube Stacking Trajectory ===
CUBE_HEIGHT = 0.06 # Actual height of the cube
CUBE_STACK_ON_CUBE_Z_OFFSET = CUBE_HEIGHT + 0.005 # Target Z for top cube relative to bottom cube's origin (0.06 cube height + 0.005 buffer)
CUBE_STACK_PRE_GRASP_OFFSET_POS_CUBE_FRAME = np.array([-0.090, -0.001, 0.18])  # Relative to cube's origin and orientation
CUBE_STACK_PRE_GRASP_EULER_XYZ_DEG_CUBE_FRAME = np.array([-90.0, 30.0, 0.0]) # Relative to cube's orientation
CUBE_STACK_GRASP_APPROACH_DISTANCE_Z_WORLD = 0.05 # World Z-axis downward movement from pre-grasp EEF Z
CUBE_STACK_INTERMEDIATE_LIFT_HEIGHT_ABOVE_BASE = 0.25 # Z-offset for intermediate waypoints, relative to base of target stack cube

# === Constants for cabinet pouring tasks ===
# Naming Conventions:
# - OBJECT_ACTION_POS/QUAT/ABS: Position/Orientation for a specific action on an object
#   * OBJECT: The item being oriented (e.g., DRAWER_HANDLE, MUG, BOTTLE, MAT)
#   * ACTION: The specific action or phase (e.g., APPROACH, GRASP, LIFT, PLACE, POUR)
#   * POS/QUAT: Position in 3D space (x, y, z); Orientation represented as Euler angles (roll, pitch, yaw)
#   * ABS: Absolute position/orientation in the world frame

# 1. Open the drawer handle (Relative to drawer's origin and orientation)
DRAWER_HANDLE_GRASP_POS             = np.array([-0.18850000, -0.06160968, 0.08380000])
DRAWER_HANDLE_GRASP_QUAT            = np.array([93, 41, 1.5])  # degrees, relative to drawer's orientation
DRAWER_HANDLE_APPROACH_OFFSET_POS   = np.array([-0.055, 0.01359585, 0.0])
DRAWER_HANDLE_LEAVE_OFFSET_POS      = np.array([-0.19381688, 0.0, 0.0])

# 2. Pick-and-place the mug
# Absolute POS and QUAT for avoiding singularity during transition
MUG_APPROACH_ABS_POS                = np.array([0.20, 0.27, 0.90])  
MUG_APPROACH_ABS_QUAT               = np.array([90, 0, -60])
# Relative to mug's origin and orientation
MUG_GRASP_POS                       = np.array([-0.00718370, 0.17486785, 0.13347340])
MUG_GRASP_QUAT                      = np.array([89.99999785, 19.99999544, -34.41778335])   # -74.96883661 - (-43.30877006) = -31.66 degrees
MUG_APPROACH_OFFSET_POS             = np.array([0.0, 0.0, 0.10])  # Relative to MUG_GRASP_POS
MUG_LEAVE_OFFSET_POS                = np.array([0.0, 0.0, 0.19])  # Relative to MUG_GRASP_POS
# Relative to mat's origin (but relative orientation)
MAT_PLACE_POS                       = np.array([-0.11000000, 0.09500000, 0.16500000])
MAT_PLACE_ABS_QUAT                  = np.array([84.54269505, 17.98494010, -31.31594710])    # Absolute orientation
MAT_APPROACH_OFFSET_POS             = np.array([0.0, 0.0, 0.03])
MAT_LEAVE_OFFSET_POS                = np.array([0.0, 0.0, 0.03])
# Relative to the current pose before pushing
DRAWER_PUSH_DIRECTION_OFFSET = np.array([0.200, 0, 0.02])   # Rise the hand 0.02 in z to keep holding the handle while pushing

# 3. Pick the bottle, pour it, and place it back
BOTTLE_GRASP_POS                    = np.array([-0.19999976, -0.06999982, -0.04498661])
BOTTLE_GRASP_QUAT                   = np.array([0.0, 0.0, -176])  # Relative to bottle's orientation (yaw = 180 deg)
BOTTLE_APPROACH_OFFSET_POS          = np.array([-0.02, -0.02, 0.02]) # Relative to BOTTLE_GRASP_POS
BOTTLE_LEAVE_OFFSET_POS             = np.array([0.0, 0.0, 0.04]) # Relative to BOTTLE_GRASP_POS
# Pouring pose relative to mat's origin
MAT_POURING_POS                     = np.array([-0.19000000, -0.16000000, 0.09900000]) # TODO: Adjust based on mat pose
MAT_POURING_QUAT                    = np.array([-50, -10, 0]) # degrees, absolute orientation
MAT_POURING_APPROACH_OFFSET_POS     = np.array([0.0, -0.02, -0.04])
MAT_POURING_LEAVE_OFFSET_POS        = np.array([0.0, -0.02, -0.04])
# Return bottle pose relative to drawer handle's origin 
DRAWER_RETURN_POS                   = np.array([-0.19999976, -0.06999982, -0.04498661])  # (TODO: Not defined yet.)
DRAWER_RETURN_QUAT                  = np.array([ 0.0, 0.0, 0.0])
DRAWER_RETURN_APPROACH_OFFSET_POS   = np.array([ 0.0, 0.0, 0.03])
DRAWER_RETURN_LEAVE_OFFSET_POS      = np.array([ -0.03, -0.03, 0.03])