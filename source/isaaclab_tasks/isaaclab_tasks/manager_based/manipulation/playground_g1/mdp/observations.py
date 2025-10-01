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

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
from termcolor import colored


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