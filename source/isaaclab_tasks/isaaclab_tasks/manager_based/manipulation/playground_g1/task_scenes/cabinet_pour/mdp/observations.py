# Copyright (c) 2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence, cast

from isaaclab.assets import Articulation, RigidObject, RigidObjectCollection
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
import isaaclab.utils.math as math_utils
from isaaclab_tasks.manager_based.manipulation.playground_g1.mdp import hand_is_grasping

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
from termcolor import colored


def object_placed(
    env: ManagerBasedRLEnv,
    hand_frame_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    xy_threshold: float = 0.0300,  # Enlarge threshold (0.015 -> 0.0300)
    height_threshold: float = 0.005,
    debug: bool = False,
) -> torch.Tensor:
    hand_frame: FrameTransformer = env.scene[hand_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]

    pos_diff = object.data.root_pos_w - target.data.root_pos_w
    height_dist = torch.linalg.vector_norm(pos_diff[:, 2:], dim=1)
    xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)

    reach = torch.logical_and(xy_dist < xy_threshold, height_dist < height_threshold)

    # Check if the left hand is open (not grasping) using hand_is_grasping
    grasping_status = hand_is_grasping(env)  # Shape: (num_envs, 2), column 0 is left hand
    left_hand_open = (1.0 - grasping_status[:, 0]).bool()  # True if open (not grasping), False if grasping

    placed = torch.logical_and(reach, left_hand_open)

    if debug:
        failed_envs = torch.where(~placed)[0]
        if len(failed_envs) > 0:
            print(colored("----------------------------------------", "red"))
            print(colored(f"object_placed {object_cfg.name} failed for envs: {failed_envs}", "red"))
            for env_idx in failed_envs:
                print(colored(f"  Env {env_idx}:", "red"))
                reach_status = reach[env_idx].item()
                left_hand_open_status = left_hand_open[env_idx].item()
                print(colored(f"    reach: {reach_status}", "red"))
                if not reach_status:
                    print(colored(f"      - xy_dist: {xy_dist[env_idx].item():.4f} (threshold: {xy_threshold})", "red"))
                    print(colored(f"      - height_dist: {height_dist[env_idx].item():.4f} (threshold: {height_threshold})", "red"))
                print(colored(f"    left_hand_open: {left_hand_open_status}", "red"))
            print(colored("----------------------------------------", "red"))

    return placed

def drawer_opened(
    env: ManagerBasedRLEnv,
    drawer_cfg: SceneEntityCfg,
    threshold: float = 0.18,
    debug: bool = False,
) -> torch.Tensor:

    drawer_pos = env.scene[drawer_cfg.name].data.joint_pos[:, drawer_cfg.joint_ids[0]]

    if debug:
        failed_envs = torch.where(~(drawer_pos > threshold))[0]
        if len(failed_envs) > 0:
            print(colored("----------------------------------------", "red"))
            print(colored(f"drawer_opened failed for envs: {failed_envs}", "red"))
            for env_idx in failed_envs:
                print(colored(f"  Env {env_idx}:", "red"))
                print(colored(f"    drawer_pos: {drawer_pos[env_idx].item():.4f} (threshold: {threshold})", "red"))
            print(colored("----------------------------------------", "red"))

    return drawer_pos > threshold

def drawer_closed(
    env: ManagerBasedRLEnv,
    drawer_cfg: SceneEntityCfg,
    threshold: float = 0.015,
) -> torch.Tensor:

    drawer_pos = env.scene[drawer_cfg.name].data.joint_pos[:, drawer_cfg.joint_ids[0]]

    #subtask_terms = cast(dict, env.obs_buf["subtask_terms"])
    #mug_placed = subtask_terms["mug_placed"]

    mug_placed = object_placed(env, 
                               SceneEntityCfg("hand_frame"),
                               SceneEntityCfg("mug"),
                               SceneEntityCfg("mug_mat")
                               )

    return torch.logical_and(mug_placed, drawer_pos < threshold)

def get_drawer_pose(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    drawer_frame: FrameTransformer = env.scene["cabinet_frame"]
    drawer_pos = drawer_frame.data.target_pos_w[:, 0, :] - env.scene.env_origins[:, 0:3]
    drawer_quat = drawer_frame.data.target_quat_w[:, 0, :]

    return torch.cat(
        (
            drawer_pos,
            drawer_quat,
        ),
        dim=1,
    )

def is_poured(
    env: ManagerBasedRLEnv,
    hand_frame_cfg: SceneEntityCfg,
    bottle_frame_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    tilt_angle: float = 50,
    xy_threshold: float = 0.035,
    height_threshold: float = 0.15,
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