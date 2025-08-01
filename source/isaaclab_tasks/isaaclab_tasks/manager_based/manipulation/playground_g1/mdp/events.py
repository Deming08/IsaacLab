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
    max_retries: int = 50,
):
    """
    Generate a list of random poses (positions and orientations) for multiple objects.

    This function samples random poses within specified ranges, ensuring minimum separation
    between objects. If max tries are reached per object, it retries from the first object.
    If all retries (max_retries) fail, it returns the last attempt.

    Args:
        num_objects (int): The number of objects to generate poses for.
        min_separation (float, optional): The minimum distance required between any two objects.
                                        Defaults to 0.0 (no separation enforced).
        pose_range (dict[str, tuple[float, float]], optional): A dictionary specifying the range
                                                              (min, max) for each degree of freedom:
                                                              "x", "y", "z", "roll", "pitch", "yaw".
                                                              Defaults to {} (using (0.0, 0.0) if unspecified).
        max_sample_tries (int, optional): The maximum number of attempts to sample a valid pose
                                         for each object per retry. Defaults to 5000.
        max_retries (int, optional): The maximum number of retry attempts if sampling fails.
                                    Defaults to 10.

    Returns:
        list: A list of poses, where each pose is a list [x, y, z, roll, pitch, yaw].

    Raises:
        ValueError: If sampling fails due to insufficient space after all retries.
    """
    # Extract pose ranges for x, y, z, roll, pitch, yaw
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    
    for attempt in range(max_retries):
        attempt+=1
        pose_list = []  # Reset pose list for new retry
        sampling_success = True

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

                # If max tries reached, prepare for retry
                if j == max_sample_tries - 1:
                    """print(f"Reached max randomize tries ({max_sample_tries}) for object {i} in retry {attempt}. "
                          f"Last pose accepted may violate min_separation={min_separation}.")"""
                    pose_list.append(sample)
                    sampling_success = False
                    break

        # Check if all poses were successfully generated
        if sampling_success:
            return pose_list
        
        #print(f"Attempt to randomize object {attempt}/{max_retries} failed. Attempting to resample all poses.")

    # If all retries exhausted, return the last attempt
    print(f"Max retries ({max_retries}) reached. Returning last attempt with {len(pose_list)} poses, it may violate min_separation={min_separation}.")

    return pose_list

def randomize_object_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfgs: list[SceneEntityCfg],
    min_separation: float = 0.0,
    pose_range: dict[str, tuple[float, float]] = {},
    max_sample_tries: int = 5000,
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


def reset_robot_state(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    offset_x_list: list[float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # get default root state
    root_states = asset.data.default_root_state[env_ids].clone()

    offset_x = torch.tensor(offset_x_list, device=asset.device)[
        torch.randint(0, len(offset_x_list), (len(env_ids),), device=asset.device)
    ]

    # poses
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    positions[:, 0] += offset_x

    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
