# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the lift task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, cast

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
from termcolor import colored


def task_done(
    env: ManagerBasedRLEnv,
    hand_frame_cfg: SceneEntityCfg = SceneEntityCfg("hand_frame"),
    bottle_cfg: SceneEntityCfg = SceneEntityCfg("bottle"),
    xy_threshold: float = 0.05, # Not critical, so enlarge this value from 0.03 to 0.05
    height_threshold: float = 0.005,
    hand_dist_threshold: float = 0.10,  # For 3D distance of hand positions
    left_hand_init_pose: tuple = (0.01, 0.30, 0.80),  # Left hand initial position
    right_hand_init_pose: tuple = (0.01, -0.30, 0.80),  # Right hand initial position
    debug: bool = False,
):
    hand_frame: FrameTransformer = env.scene[hand_frame_cfg.name]
    bottle: RigidObject = env.scene[bottle_cfg.name]

    subtask_terms = cast(dict, env.obs_buf["subtask_terms"])

    drawer_closed = subtask_terms["drawer_closed"]
    mug_placed = subtask_terms["mug_placed"]

    default_bottle_state = bottle.data.default_root_state.clone()
    default_bottle_state[:, 0:3] += env.scene.env_origins
    
    pos_diff = bottle.data.root_pos_w - default_bottle_state[:, 0:3]
    xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)
    height_dist = torch.linalg.vector_norm(pos_diff[:, 2:], dim=1)

    bottle_placed = torch.logical_and(xy_dist < xy_threshold, height_dist < height_threshold)
    

    hand_frame: FrameTransformer = env.scene[hand_frame_cfg.name]

    left_hand_pos = hand_frame.data.target_pos_w[:, 0, :] - env.scene.env_origins
    right_hand_pos = hand_frame.data.target_pos_w[:, 1, :] - env.scene.env_origins 
    
    left_hand_init = torch.tensor(left_hand_init_pose, device=env.device).expand(env.scene.num_envs, -1)
    right_hand_init = torch.tensor(right_hand_init_pose, device=env.device).expand(env.scene.num_envs, -1)

    # Calculate 3D distance for hands
    left_pos_diff = left_hand_pos - left_hand_init
    right_pos_diff = right_hand_pos - right_hand_init
    left_dist = torch.linalg.vector_norm(left_pos_diff, dim=1)  # 3D Euclidean distance
    right_dist = torch.linalg.vector_norm(right_pos_diff, dim=1)  # 3D Euclidean distance

    # Check if hands are within threshold
    left_hand_back = left_dist < hand_dist_threshold
    right_hand_back = right_dist < hand_dist_threshold

    condition_1 = torch.logical_and(drawer_closed, mug_placed)
    #condition_2 = torch.logical_and(condition_1, bottle_placed)
    condition_3 = torch.logical_and(left_hand_back, right_hand_back)

    done = torch.logical_and(condition_1, condition_3)

    if debug:
        failed_envs = torch.where(~done)[0]
        if len(failed_envs) > 0:
            # Using ANSI escape codes is suitable for most terminals, but you can use termcolor for portability.
            # Example with termcolor:
            print(colored("----------------------------------------", "red"))
            print(colored(f"Task failed for envs: {failed_envs}", "red"))
            for env_idx in failed_envs:
                print(colored(f"  Env {env_idx}:", "red"))
                print(colored(f"    drawer_closed: {drawer_closed[env_idx].item()}", "red"))
                print(colored(f"    mug_placed: {mug_placed[env_idx].item()}", "red"))
                print(colored(f"    bottle_placed: {bottle_placed[env_idx].item()}", "red"))
                print(colored(f"      - xy_dist: {xy_dist[env_idx].item():.4f} (threshold: {xy_threshold})", "red"))
                print(colored(f"      - height_dist: {height_dist[env_idx].item():.4f} (threshold: {height_threshold})", "red"))
                print(colored(f"    left_hand_back: {left_hand_back[env_idx].item()}", "red"))
                print(colored(f"      - hand_dist: {left_dist[env_idx].item():.4f} (threshold: {hand_dist_threshold})", "red"))
                print(colored(f"    right_hand_back: {right_hand_back[env_idx].item()}", "red"))
                print(colored(f"      - hand_dist: {right_dist[env_idx].item():.4f} (threshold: {hand_dist_threshold})", "red"))
            print(colored("----------------------------------------", "red"))

    return done


def rigid_object_dropping(
    env: ManagerBasedRLEnv, minimum_height: float
    ) -> torch.Tensor:
    """Terminate when the asset's height is below the minimum height.

    Note:
        This is currently only supported for flat terrains, i.e. the minimum height is in the world frame.
    """

    object_dropping = torch.zeros(env.scene.num_envs, dtype=torch.bool, device=env.device)

    # Check each rigid object until one is found below the minimum height
    for rigid_object in env.scene.rigid_objects.values():
        object_dropping = rigid_object.data.root_pos_w[:, 2] < minimum_height
        if object_dropping.any():
            return object_dropping

    return object_dropping