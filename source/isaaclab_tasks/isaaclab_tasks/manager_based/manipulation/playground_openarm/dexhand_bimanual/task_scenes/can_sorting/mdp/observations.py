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


def target_object_obs(
    env: ManagerBasedRLEnv,
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

    target_destination = "red_basket" if target_object == "red_can" else "blue_basket"

    # Assign a numerical identifier for the target object
    target_id = 0.0 if target_object == "red_can" else 1.0  # 0 for red_can, 1 for blue_can
    target_id_tensor = torch.full((env.scene.num_envs, 1), target_id, device=env.device, dtype=torch.float32)

    # Get robot's hand pos
    hand_frame = env.scene["hand_frame"]
    left_hand_pos = hand_frame.data.target_pos_w[:, 0, :] # 1 is right eef,(0 is left eef)
    right_hand_pos = hand_frame.data.target_pos_w[:, 1, :] # 1 is right eef,(0 is left eef)

    # Get target object position and quaternion
    object_pos = env.scene[target_object].data.root_pos_w
    object_quat = env.scene[target_object].data.root_quat_w

    destination_pos = env.scene[target_destination].data.root_pos_w
    destination_quat = env.scene[target_destination].data.root_quat_w

    # Compute vectors from end-effectors to target object
    left_eef_to_object = object_pos - left_hand_pos
    right_eef_to_object = object_pos - right_hand_pos

    # Concatenate all observations along dim=1
    return torch.cat(
        (
            object_pos,          # Shape: (num_envs, 3)
            object_quat,         # Shape: (num_envs, 4)
            destination_pos,     # Shape: (num_envs, 3)
            destination_quat,    # Shape: (num_envs, 4)
            left_eef_to_object,  # Shape: (num_envs, 3)
            right_eef_to_object, # Shape: (num_envs, 3)
            target_id_tensor,    # Shape: (num_envs, 1)
        ),
        dim=1,
    )  # Total shape: (num_envs, 14)
