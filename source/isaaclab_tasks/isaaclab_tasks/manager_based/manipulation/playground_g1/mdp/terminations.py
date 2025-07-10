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
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def task_done(
    env: ManagerBasedRLEnv,
    hand_frame_cfg: SceneEntityCfg = SceneEntityCfg("hand_frame"),
    mug_cfg: SceneEntityCfg = SceneEntityCfg("mug"),
    mug_mat_cfg: SceneEntityCfg = SceneEntityCfg("mug_mat"),
    bottle_cfg: SceneEntityCfg = SceneEntityCfg("bottle"),
    xy_threshold: float = 0.01,
    height_threshold: float = 0.005,
    left_eef_max_x: float = 0.30,
    left_eef_max_y: float = -0.10,
    right_eef_max_x: float = 0.30,
    right_eef_max_y: float = -0.10,
):

    hand_frame: FrameTransformer = env.scene[hand_frame_cfg.name]
    mug: RigidObject = env.scene[mug_cfg.name]
    mug_mat: RigidObject = env.scene[mug_mat_cfg.name]
    bottle: RigidObject = env.scene[bottle_cfg.name]

    pos_diff = mug.data.root_pos_w - mug_mat.data.root_pos_w
    height_dist = torch.linalg.vector_norm(pos_diff[:, 2:], dim=1)
    xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)

    #TODO ...

    done = torch.logical_and(xy_dist < xy_threshold, height_dist < height_threshold)

    return done