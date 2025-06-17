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
import math


import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils

from isaaclab.assets import Articulation, DeformableObject, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

import carb
carb_settings_iface = carb.settings.get_settings()

IDLE_OBJECT_POSE = (0.6, 0.4, 0.88)

def reset_random_choose_object(
    env: 'ManagerBasedEnv',
    env_ids: torch.Tensor,
    target_pose: tuple[float, float, float],
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg_list: tuple[SceneEntityCfg],
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
    idle_pose:torch.Tensor = torch.tensor(IDLE_OBJECT_POSE, device=env.device, dtype=torch.float32)

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



def randomize_joint_by_gaussian_offset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    mean: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    asset: Articulation = env.scene[asset_cfg.name]

    # Add gaussian noise to joint states
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()
    joint_pos += math_utils.sample_gaussian(mean, std, joint_pos.shape, joint_pos.device)

    # Clamp joint pos to limits
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])

    # Don't noise the gripper poses
    joint_pos[:, -2:] = asset.data.default_joint_pos[env_ids, -2:]

    # Set into the physics simulation
    asset.set_joint_position_target(joint_pos, env_ids=env_ids)
    asset.set_joint_velocity_target(joint_vel, env_ids=env_ids)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def sample_object_poses(
    num_objects: int,
    min_separation: float = 0.0,
    pose_range: dict[str, tuple[float, float]] = {},
    max_sample_tries: int = 5000,
):
    """
    Generate a list of random poses (positions and orientations) for multiple objects.

    This function samples random poses within specified ranges, ensuring minimum separation
    between objects. If max tries are reached, it attempts to adjust min_separation or raises
    a warning with the last valid pose.

    Args:
        num_objects (int): The number of objects to generate poses for.
        min_separation (float, optional): The minimum distance required between any two objects.
                                        Defaults to 0.0 (no separation enforced).
        pose_range (dict[str, tuple[float, float]], optional): A dictionary specifying the range
                                                              (min, max) for each degree of freedom:
                                                              "x", "y", "z", "roll", "pitch", "yaw".
                                                              Defaults to {} (using (0.0, 0.0) if unspecified).
        max_sample_tries (int, optional): The maximum number of attempts to sample a valid pose
                                         for each object. Defaults to 5000.

    Returns:
        list: A list of poses, where each pose is a list [x, y, z, roll, pitch, yaw].

    Raises:
        ValueError: If sampling fails due to insufficient space after all retries.
    """
    # Extract pose ranges for x, y, z, roll, pitch, yaw
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    
    # Initialize list to store generated poses
    pose_list = []

    for i in range(num_objects):
        for j in range(max_sample_tries):
            # Sample a random pose within the specified ranges
            sample = [random.uniform(range[0], range[1]) if range[1] > range[0] else range[0] for range in range_list]

            # Accept pose if it is the first one
            if len(pose_list) == 0:
                pose_list.append(sample)
                break

            # Check if pose of object is sufficiently far away from all other objects
            separation_check = [math.dist(sample[:3], pose[:3]) > min_separation for pose in pose_list]
            if False not in separation_check:
                pose_list.append(sample)
                break

            # If max tries reached, attempt to reduce min_separation or warn
            if j == max_sample_tries - 1:
                print(f"Reached max randomize tries ({max_sample_tries}) for object {i}. "
                      f"Last pose accepted may violate min_separation={min_separation}.")
                pose_list.append(sample)
                break

    # Validate if space is sufficient
    if len(pose_list) < num_objects:
        raise ValueError(f"Failed to generate {num_objects} poses with min_separation={min_separation} "
                         f"in the given pose_range. Consider increasing range or reducing min_separation.")

    return pose_list


def randomize_object_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfgs: list[SceneEntityCfg],
    min_separation: float = 0.0,
    pose_range: dict[str, tuple[float, float]] = {},
    max_sample_tries: int = 10000,
):
    if env_ids is None:
        return

    # Randomize poses in each environment independently
    for cur_env in env_ids.tolist():
        pose_list = sample_object_poses(
            num_objects=len(asset_cfgs),
            min_separation=min_separation,
            pose_range=pose_range,
            max_sample_tries=max_sample_tries,
        )

        # Randomize pose for each object
        for i in range(len(asset_cfgs)):
            asset_cfg = asset_cfgs[i]
            asset = env.scene[asset_cfg.name]

            # Write pose to simulation
            pose_tensor = torch.tensor([pose_list[i]], device=env.device)
            positions = pose_tensor[:, 0:3] + env.scene.env_origins[cur_env, 0:3]
            orientations = math_utils.quat_from_euler_xyz(pose_tensor[:, 3], pose_tensor[:, 4], pose_tensor[:, 5])
            asset.write_root_pose_to_sim(
                torch.cat([positions, orientations], dim=-1), env_ids=torch.tensor([cur_env], device=env.device)
            )
            asset.write_root_velocity_to_sim(
                torch.zeros(1, 6, device=env.device), env_ids=torch.tensor([cur_env], device=env.device)
            )