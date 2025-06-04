# Copyright (c) 2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

import carb
carb_settings_iface = carb.settings.get_settings()


def object_obs(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """
    Object observations (in world frame):
        object pos,
        object quat,
        left_eef to object,
        right_eef_to object,
    """

    body_pos_w = env.scene["robot"].data.body_pos_w
    left_eef_idx = env.scene["robot"].data.body_names.index("left_wrist_yaw_link")
    right_eef_idx = env.scene["robot"].data.body_names.index("right_wrist_yaw_link")
    left_eef_pos = body_pos_w[:, left_eef_idx] - env.scene.env_origins
    right_eef_pos = body_pos_w[:, right_eef_idx] - env.scene.env_origins

    object_pos = env.scene["object"].data.root_pos_w - env.scene.env_origins
    object_quat = env.scene["object"].data.root_quat_w

    left_eef_to_object = object_pos - left_eef_pos
    right_eef_to_object = object_pos - right_eef_pos

    return torch.cat(
        (
            object_pos,
            object_quat,
            left_eef_to_object,
            right_eef_to_object,
        ),
        dim=1,
    )


def get_left_eef_pos(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    body_pos_w = env.scene["robot"].data.body_pos_w
    left_eef_idx = env.scene["robot"].data.body_names.index("left_wrist_yaw_link")
    left_eef_pos = body_pos_w[:, left_eef_idx] - env.scene.env_origins
    
    return left_eef_pos


def get_left_eef_quat(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    body_quat_w = env.scene["robot"].data.body_quat_w
    left_eef_idx = env.scene["robot"].data.body_names.index("left_wrist_yaw_link")
    left_eef_quat = body_quat_w[:, left_eef_idx]

    return left_eef_quat


def get_right_eef_pos(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    body_pos_w = env.scene["robot"].data.body_pos_w
    right_eef_idx = env.scene["robot"].data.body_names.index("right_wrist_yaw_link")
    right_eef_pos = body_pos_w[:, right_eef_idx] - env.scene.env_origins

    return right_eef_pos


def get_right_eef_quat(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    body_quat_w = env.scene["robot"].data.body_quat_w
    right_eef_idx = env.scene["robot"].data.body_names.index("right_wrist_yaw_link")
    right_eef_quat = body_quat_w[:, right_eef_idx]

    return right_eef_quat


def get_hand_state(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    hand_joint_states = env.scene["robot"].data.joint_pos[:, -14:]  # Hand joints are last 14 entries of joint state

    return hand_joint_states


def get_head_state(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    robot_joint_names = env.scene["robot"].data.joint_names
    head_joint_names = ["head_pitch_joint", "head_roll_joint", "head_yaw_joint"]
    indexes = torch.tensor([robot_joint_names.index(name) for name in head_joint_names], dtype=torch.long)
    head_joint_states = env.scene["robot"].data.joint_pos[:, indexes]

    return head_joint_states


def get_all_robot_link_state(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    body_pos_w = env.scene["robot"].data.body_link_state_w[:, :, :]
    all_robot_link_pos = body_pos_w

    return all_robot_link_pos


def get_processed_action(env: ManagerBasedRLEnv, action_name: str) -> torch.Tensor:
    """The last input action after process to the environment.

    The name of the action term for which the action is required.
    """

    return env.action_manager.get_term(action_name).processed_actions

def target_object_obs(
    env: 'ManagerBasedRLEnv',
) -> torch.Tensor:
    """
    Object observations (in world frame) for the target object (red_can or blue_can):
        - Target object position (x, y, z),
        - Target object quaternion (w, x, y, z),
        - Vector from left end-effector to target object (x, y, z),
        - Vector from right end-effector to target object (x, y, z),
        - Target object color ID (0.0 for red_can, 1.0 for blue_can).

    The target object is dynamically determined by carb_settings_iface.get("/pickplace_env/target_object"),
    which should return "red_can" or "blue_can".

    Args:
        env: The RL environment instance.

    Returns:
        torch.Tensor: A tensor of shape (num_envs, 14) containing the concatenated observations.
    """
    # Get the target object name from carb settings
    target_object = carb_settings_iface.get("/pickplace_env/target_object")
    if target_object not in ["red_can", "blue_can"]:
        #raise ValueError(f"Invalid target object: {target_object}. Must be 'red_can' or 'blue_can'.")
        target_object = "red_can"
    
    # Assign a numerical identifier for the target object
    target_id = 0.0 if target_object == "red_can" else 1.0  # 0 for red_can, 1 for blue_can
    target_id_tensor = torch.full((env.scene.num_envs, 1), target_id, device=env.device, dtype=torch.float32)

    # Get robot's body positions and end-effector indices
    body_pos_w = env.scene["robot"].data.body_pos_w
    left_eef_idx = env.scene["robot"].data.body_names.index("left_wrist_yaw_link")
    right_eef_idx = env.scene["robot"].data.body_names.index("right_wrist_yaw_link")
    left_eef_pos = body_pos_w[:, left_eef_idx] - env.scene.env_origins
    right_eef_pos = body_pos_w[:, right_eef_idx] - env.scene.env_origins

    # Get target object position and quaternion
    object_pos = env.scene[target_object].data.root_pos_w - env.scene.env_origins
    object_quat = env.scene[target_object].data.root_quat_w

    # Compute vectors from end-effectors to target object
    left_eef_to_object = object_pos - left_eef_pos
    right_eef_to_object = object_pos - right_eef_pos

    # Concatenate all observations along dim=1
    return torch.cat(
        (
            object_pos,          # Shape: (num_envs, 3)
            object_quat,         # Shape: (num_envs, 4)
            left_eef_to_object,  # Shape: (num_envs, 3)
            right_eef_to_object, # Shape: (num_envs, 3)
            target_id_tensor,    # Shape: (num_envs, 1)
        ),
        dim=1,
    )  # Total shape: (num_envs, 14)