# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configurations for OpenArm-DexHands task scenes environments."""

import gymnasium as gym
import os

from .can_sorting import can_sorting_env_cfg
from .cube_stack import cube_stack_env_cfg
from .cabinet_pour import cabinet_pour_env_cfg

gym.register(
    id="Isaac-Can-Sorting-OpenArm-DexHand-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": can_sorting_env_cfg.CanSortingOpenArmEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Cube-Stack-OpenArm-DexHand-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": cube_stack_env_cfg.CubeStackOpenArmEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Cabinet-Pour-OpenArm-DexHand-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": cabinet_pour_env_cfg.CabinetPourOpenArmEnvCfg,
    },
    disable_env_checker=True,
)
