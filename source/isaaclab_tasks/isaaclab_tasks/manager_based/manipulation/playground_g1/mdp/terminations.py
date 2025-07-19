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
    left_eef_max_x: float = 0.30,
    left_eef_max_y: float = 0.15,
    right_eef_max_x: float = 0.30,
    right_eef_max_y: float = -0.15,
    debug: bool = False,
):
    hand_frame: FrameTransformer = env.scene[hand_frame_cfg.name]
    bottle: RigidObject = env.scene[bottle_cfg.name]

    subtask_terms = cast(dict, env.obs_buf["subtask_terms"])
    if debug:   # TODO: Short-term solution: drawer_closed is always False ......
        drawer_closed = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        mug_placed = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    else:
        drawer_closed = subtask_terms["drawer_closed"]
        mug_placed = subtask_terms["mug_placed"]


    default_bottle_state = bottle.data.default_root_state.clone()
    default_bottle_state[:, 0:3] += env.scene.env_origins
    
    pos_diff = bottle.data.root_pos_w - default_bottle_state[:, 0:3]
    xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)
    height_dist = torch.linalg.vector_norm(pos_diff[:, 2:], dim=1)

    bottle_placed = torch.logical_and(xy_dist < xy_threshold, height_dist < height_threshold)
    

    hand_frame: FrameTransformer = env.scene[hand_frame_cfg.name]
    left_eef_x = hand_frame.data.target_pos_w[:, 0, 0] - env.scene.env_origins[:, 0]
    left_eef_y = hand_frame.data.target_pos_w[:, 0, 1] - env.scene.env_origins[:, 1]
    right_eef_x = hand_frame.data.target_pos_w[:, 1, 0] - env.scene.env_origins[:, 0]
    right_eef_y = hand_frame.data.target_pos_w[:, 1, 1] - env.scene.env_origins[:, 1]

    left_hand_back = torch.logical_and(left_eef_x < left_eef_max_x, left_eef_y > left_eef_max_y)
    right_hand_back = torch.logical_and(right_eef_x < right_eef_max_x, right_eef_y < right_eef_max_y)
    

    condition_1 = torch.logical_and(drawer_closed, mug_placed)
    condition_2 = torch.logical_and(condition_1, bottle_placed)
    condition_3 = torch.logical_and(left_hand_back, right_hand_back)

    done = torch.logical_and(condition_2, condition_3)

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
                print(colored(f"      - left_eef_x: {left_eef_x[env_idx].item():.4f} (max: {left_eef_max_x})", "red"))
                print(colored(f"      - left_eef_y: {left_eef_y[env_idx].item():.4f} (max: {left_eef_max_y})", "red"))
                print(colored(f"    right_hand_back: {right_hand_back[env_idx].item()}", "red"))
                print(colored(f"      - right_eef_x: {right_eef_x[env_idx].item():.4f} (max: {right_eef_max_x})", "red"))
                print(colored(f"      - right_eef_y: {right_eef_y[env_idx].item():.4f} (max: {right_eef_max_y})", "red"))
            print(colored("----------------------------------------", "red"))

    return done
