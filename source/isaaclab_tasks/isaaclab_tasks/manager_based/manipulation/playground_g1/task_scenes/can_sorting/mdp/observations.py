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

# z is from bottom
RED_BASKET_CENTER = (0.4, -0.05, 0.81)
BLUE_BASKET_CENTER = (0.4, -0.2, 0.81)
BASKET_LENTH_WIDTH_HEIGHT = (0.14, 0.1, 0.08)

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
        #! Observation is before the event is triggered, so it will be raised on initialization.
        #raise ValueError(f"Invalid target object: {target_object}. Must be 'red_can' or 'blue_can'.")
        target_object = "red_can" # work around
    
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


def task_completion(
    env: 'ManagerBasedRLEnv',
    right_wrist_max_x: float = 0.2,
    right_wrist_max_y: float = -0.15,
    min_vel: float = 0.1,
) -> torch.Tensor:
    """
    Determine if the target object (red_can or blue_can) is placed in the corresponding basket.

    This function checks whether the target object (obtained from carb settings) is placed in
    the correct basket (red_basket for red_can, blue_basket for blue_can) by verifying:
    1. Object is within the target basket's x, y, z range (cubic volume).
    2. Object velocity is below threshold.
    3. Right robot wrist is retracted back towards body (past a given x, y pos threshold).

    Args:
        env: The RL environment instance.
        right_wrist_max_x: Maximum x position of the right wrist for task completion.
        right_wrist_max_y: Maximum y position of the right wrist for task completion.
        min_vel: Minimum velocity magnitude of the object for task completion.

    Returns:
        Boolean tensor indicating which environments have completed the task.
    """
    # Get the target object name from carb settings
    target_object = carb_settings_iface.get("/pickplace_env/target_object")
    if target_object not in ["red_can", "blue_can"]:
        #! Observation is before the event is triggered, so it will be raised on initialization.
        #raise ValueError(f"Invalid target object: {target_object}. Must be 'red_can' or 'blue_can'.")
        target_object = "red_can" # work around

    # Get object entity from the scene based on target_object
    object = env.scene[target_object]

    # Extract object position relative to environment origin
    obj_x = object.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    obj_y = object.data.root_pos_w[:, 1] - env.scene.env_origins[:, 1]
    obj_z = object.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    obj_vel = torch.abs(object.data.root_vel_w)

    # Get right wrist position relative to environment origin
    robot_body_pos_w = env.scene["robot"].data.body_pos_w
    right_eef_idx = env.scene["robot"].data.body_names.index("right_wrist_yaw_link")
    right_wrist_x = robot_body_pos_w[:, right_eef_idx, 0] - env.scene.env_origins[:, 0]
    right_wrist_y = robot_body_pos_w[:, right_eef_idx, 1] - env.scene.env_origins[:, 1]

    # Define basket ranges
    if target_object == "red_can":
        basket_center_x, basket_center_y, basket_center_z = RED_BASKET_CENTER
    else:  # blue_can
        basket_center_x, basket_center_y, basket_center_z = BLUE_BASKET_CENTER
    basket_length, basket_width, basket_height = BASKET_LENTH_WIDTH_HEIGHT

    min_x = basket_center_x - basket_length / 2
    max_x = basket_center_x + basket_length / 2 
    min_y = basket_center_y - basket_width / 2
    max_y = basket_center_y + basket_width / 2
    min_z = basket_center_z
    max_z = basket_center_z + basket_height

    # Check all success conditions and combine with logical AND
    done = torch.logical_and(obj_x < max_x, obj_x > min_x)
    done = torch.logical_and(done, obj_y < max_y)
    done = torch.logical_and(done, obj_y > min_y)
    done = torch.logical_and(done, obj_z < max_z)
    done = torch.logical_and(done, obj_z > min_z)
    done = torch.logical_and(done, right_wrist_x < right_wrist_max_x)
    done = torch.logical_and(done, right_wrist_y < right_wrist_max_y)
    done = torch.logical_and(done, obj_vel[:, 0] < min_vel)  # x velocity
    done = torch.logical_and(done, obj_vel[:, 1] < min_vel)  # y velocity
    done = torch.logical_and(done, obj_vel[:, 2] < min_vel)  # z velocity

    return done