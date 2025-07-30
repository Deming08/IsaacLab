# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to enable different events.

Events include anything related to altering the simulation state. This includes changing the physics
materials, applying external forces, and resetting the state of the asset.

The functions can be passed to the :class:`isaaclab.managers.EventTermCfg` object to enable
the event introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
import random


import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils

from isaaclab.assets import Articulation, DeformableObject, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

import carb
carb_settings_iface = carb.settings.get_settings()

#(0.6, 0.4, 0.88)
def reset_random_choose_object(
    env: 'ManagerBasedEnv',
    env_ids: torch.Tensor,
    target_pose: tuple[float, float, float],
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg_list: tuple[SceneEntityCfg],
    idle_pose: tuple[float, float, float] = (0.6, 0.4, 0.88),
):
    """
    Randomly select one asset from asset_cfg_list as the target, set its pose with random offset
    based on target_pose, and set other assets to idle_pose without offset.

    Args:
        env (ManagerBasedEnv): The environment instance.
        env_ids (torch.Tensor): Tensor of environment IDs.
        pose_range (Dict[str, Tuple[float, float]]): Range for pose offsets (x, y, z, roll, pitch, yaw).
        velocity_range (Dict[str, Tuple[float, float]]): Range for velocity offsets.
        asset_cfg_list (Tuple[SceneEntityCfg]): List of asset configurations to choose from.
    """
    # Define target and idle poses
    target_pose:torch.Tensor = torch.tensor(target_pose, device=env.device, dtype=torch.float32)
    idle_pose:torch.Tensor = torch.tensor(idle_pose, device=env.device, dtype=torch.float32)

    # Randomly choose one asset as the target
    choosed_asset_cfg = random.choice(asset_cfg_list)
    target_asset_name = choosed_asset_cfg.name
    #other_asset_cfgs = [cfg for cfg in asset_cfg_list if cfg.name != target_asset_name]

    # Process each asset
    for asset_cfg in asset_cfg_list:
        asset_name = asset_cfg.name
        asset: RigidObject | Articulation = env.scene[asset_name]
        root_states = asset.data.default_root_state[env_ids].clone()

        # Extract position and orientation from root states
        #positions = root_states[:, 0:3] + env.scene.env_origins[env_ids]
        orientations = root_states[:, 3:7]

        # Velocities
        range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=asset.device)
        rand_vel_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)
        velocities = root_states[:, 7:13] + rand_vel_samples

        # Set pose based on whether it's the target asset
        if asset_name == target_asset_name:
            # Target asset: apply random offset based on target_pose
            range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
            ranges = torch.tensor(range_list, device=asset.device)
            rand_pose_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

            # Adjust positions with target_pose as base
            positions = target_pose + rand_pose_samples[:, 0:3]
            orientations_delta = math_utils.quat_from_euler_xyz(rand_pose_samples[:, 3], rand_pose_samples[:, 4], rand_pose_samples[:, 5])
            orientations = math_utils.quat_mul(orientations, orientations_delta)
            
            carb_settings_iface.set_string("/pickplace_env/target_object", asset_name)
        else:
            # Other assets: set to idle_pose without offset
            positions = idle_pose.repeat(len(env_ids), 1)

        # Update physics simulation
        asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
        asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


