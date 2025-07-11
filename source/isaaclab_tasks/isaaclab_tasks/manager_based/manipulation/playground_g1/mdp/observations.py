# Copyright (c) 2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject, RigidObjectCollection
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def get_left_eef_pos(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene["ee_frame"]
    left_eef_pos = ee_frame.data.target_pos_w[:, 0, :] - env.scene.env_origins[:, 0:3]

    return left_eef_pos


def get_left_eef_quat(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene["ee_frame"]
    left_eef_quat = ee_frame.data.target_quat_w[:, 0, :]

    return left_eef_quat


def get_right_eef_pos(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene["ee_frame"]
    right_eef_pos = ee_frame.data.target_pos_w[:, 1, :] - env.scene.env_origins[:, 0:3]

    return right_eef_pos


def get_right_eef_quat(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene["ee_frame"]
    right_eef_quat = ee_frame.data.target_quat_w[:, 1, :]

    return right_eef_quat


def get_hand_state(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    hand_joint_states = env.scene["robot"].data.joint_pos[:, -14:]  # Hand joints are last 14 entries of joint state

    return hand_joint_states


def get_processed_action(env: ManagerBasedRLEnv, action_name: str) -> torch.Tensor:
    """The last input action after process to the environment.

    The name of the action term for which the action is required.
    """

    return env.action_manager.get_term(action_name).processed_actions

def drawer_opened(
    env: ManagerBasedRLEnv,
    drawer_cfg: SceneEntityCfg,
    threshold: float = 0.18,
) -> torch.Tensor:

    drawer_pos = env.scene[drawer_cfg.name].data.joint_pos[:, drawer_cfg.joint_ids[0]]

    return drawer_pos > threshold

def drawer_closed(
    env: ManagerBasedRLEnv,
    drawer_cfg: SceneEntityCfg,
    threshold: float = 0.005,
) -> torch.Tensor:

    drawer_pos = env.scene[drawer_cfg.name].data.joint_pos[:, drawer_cfg.joint_ids[0]]

    return drawer_pos < threshold

def get_drawer_pos(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    drawer_frame: FrameTransformer = env.scene["cabinet_frame"]
    drawer_pos = drawer_frame.data.target_pos_w[:, 0, :] - env.scene.env_origins[:, 0:3]

    return drawer_pos

def get_object_pose(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
):
    """
    Object observations (in world frame):
        object pos,
        object quat,
    """
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    object_pos_w = object.data.root_pos_w
    object_quat_w = object.data.root_quat_w

    return torch.cat(
        (
            object_pos_w - env.scene.env_origins,
            object_quat_w,
        ),
        dim=1,
    )


def object_placed(
    env: ManagerBasedRLEnv,
    hand_frame_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    xy_threshold: float = 0.01,
    height_threshold: float = 0.005,
) -> torch.Tensor:

    hand_frame: FrameTransformer = env.scene[hand_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]

    pos_diff = object.data.root_pos_w - target.data.root_pos_w
    height_dist = torch.linalg.vector_norm(pos_diff[:, 2:], dim=1)
    xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)

    placed = torch.logical_and(xy_dist < xy_threshold, height_dist < height_threshold)

    return placed

def object_grasped(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    hand_frame_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    diff_threshold: float = 0.08,
) -> torch.Tensor:
    """Check if an object is grasped by the specified robot.

    This function determines if an object is grasped by the robot's right hand by checking:
    1. The distance between the object and the end effector is within the diff_threshold.
    2. The right hand is in a grasping state, as determined by hand_is_grasping.

    Args:
        env (ManagerBasedRLEnv): The reinforcement learning environment instance.
        robot_cfg (SceneEntityCfg): Configuration for the robot entity.
        ee_frame_cfg (SceneEntityCfg): Configuration for the end effector frame entity.
        object_cfg (SceneEntityCfg): Configuration for the object entity.
        diff_threshold (float, optional): Maximum allowed distance between object and end effector.
                                        Defaults to 0.06.
    Returns:
        torch.Tensor: A boolean tensor of shape (num_envs,) indicating whether each environment
                     has a grasped object.
    """
    # Get the robot, end effector frame, and object entities from the environment
    robot: Articulation = env.scene[robot_cfg.name]
    hand_frame: FrameTransformer = env.scene[hand_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    idx = hand_frame_cfg.body_ids[0]

    # Calculate the position difference between the object and end effector
    object_pos = object.data.root_pos_w
    hand_pos = hand_frame.data.target_pos_w[:, idx, :] # 1 is right eef,(0 is left eef)
    pose_diff = torch.linalg.vector_norm(object_pos - hand_pos, dim=1)

    # Check if the object is within the distance threshold
    close_to = pose_diff < diff_threshold

    # Check if the right hand is grasping using hand_is_grasping
    grasping_status = hand_is_grasping(env)  # Shape: (num_envs, 2), column 1 is right hand
    hand_grasping = grasping_status[:, idx].bool()  # 1.0 if grasping, 0.0 if not grasping

    # Combine distance condition with right hand grasping condition
    grasped = torch.logical_and(close_to, hand_grasping)

    return grasped

def hand_is_grasping(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    tolerance: float = 0.1,
) -> torch.Tensor:
    """Check if the robot's hand is grasping.

    This function determines whether the left and right hands of the robot are in a grasping
    state based on the joint positions. A hand is considered grasping if all relevant joints
    are close to their 'closed' angle values (within a specified tolerance). All 14 joints
    are considered, including thumb_0 joints with special handling.

    Args:
        env (ManagerBasedRLEnv): The reinforcement learning environment instance.
        robot_cfg (SceneEntityCfg, optional): Configuration for the robot entity.
                                             Defaults to SceneEntityCfg("robot").
        tolerance (float, optional): The tolerance (in radians) for considering a joint as closed.
                                    Defaults to 0.1.

    Returns:
        torch.Tensor: A float tensor of shape (num_envs, 2) where:
                     - Column 0 indicates if the left hand is grasping (1.0) or not (0.0).
                     - Column 1 indicates if the right hand is grasping (1.0) or not (0.0).

    Raises:
        ValueError: If the robot entity is not found in the environment scene.
    """
    # Get the robot articulation from the environment
    robot: Articulation = env.scene[robot_cfg.name]

    # Extract the last 14 joint positions, representing both hands
    hand_joint = robot.data.joint_pos[:, -14:]  # Shape: (num_envs, 14)

    # Define joint indices and closed angles for left and right hands
    joint_configs = {
        "left": {
            "indices": [0, 1, 2, 6, 7, 8, 12],  # All left hand joints
            "closed_angles": [-0.8, -0.8, 0.0, -0.8, -0.8, -0.2, 0.8]  # Corresponding closed values
        },
        "right": {
            "indices": [3, 4, 5, 9, 10, 11, 13],  # All right hand joints
            "closed_angles": [0.65, 0.65, 0.0, 0.9, 0.9, 0.0, -0.0]  # Corresponding closed values
        }
    }

    # Initialize grasping status tensors
    left_grasping = torch.ones(env.scene.num_envs, dtype=torch.bool, device=env.device)
    right_grasping = torch.ones(env.scene.num_envs, dtype=torch.bool, device=env.device)

    # Vectorized check for left hand
    left_indices = joint_configs["left"]["indices"]
    left_closed = torch.tensor(joint_configs["left"]["closed_angles"], device=env.device).repeat(env.scene.num_envs, 1)
    left_joints = hand_joint[:, left_indices]
    left_is_closed = torch.abs(left_joints - left_closed) <= tolerance
    left_grasping = torch.all(left_is_closed, dim=1)

    # Vectorized check for right hand
    right_indices = joint_configs["right"]["indices"]
    right_closed = torch.tensor(joint_configs["right"]["closed_angles"], device=env.device).repeat(env.scene.num_envs, 1)
    right_joints = hand_joint[:, right_indices]
    right_is_closed = torch.abs(right_joints - right_closed) <= tolerance
    right_grasping = torch.all(right_is_closed, dim=1)

    # Combine into a single tensor and convert to float
    grasping_status = torch.stack([left_grasping, right_grasping], dim=1).float()  # Shape: (num_envs, 2)

    return grasping_status

def is_poured(
    env: ManagerBasedRLEnv,
    hand_frame_cfg: SceneEntityCfg,
    bottle_frame_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    tilt_angle: float = 50,
    xy_threshold: float = 0.035,
    height_threshold: float = 0.12,
) -> torch.Tensor:

    hand_frame: FrameTransformer = env.scene[hand_frame_cfg.name]
    bottle_notch: FrameTransformer = env.scene[bottle_frame_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]

    bottle_pos = bottle_notch.data.target_pos_w[:, 0, :]
    bottle_quat = bottle_notch.data.target_quat_w[:, 0, :]

    pos_diff = bottle_pos - target.data.root_pos_w

    height_dist = torch.linalg.vector_norm(pos_diff[:, 2:], dim=1)
    xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)

    reached = torch.logical_and(xy_dist < xy_threshold, height_dist < height_threshold)

    roll, pitch, yaw = math_utils.euler_xyz_from_quat(bottle_quat)
    
    bottle_tilted = torch.rad2deg(pitch) > tilt_angle

    pouring = torch.logical_and(reached, bottle_tilted)

    return pouring