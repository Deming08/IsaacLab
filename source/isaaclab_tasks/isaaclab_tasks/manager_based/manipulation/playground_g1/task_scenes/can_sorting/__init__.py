# Copyright (c) 2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym
import os

from . import pickplace_g1_env_cfg

gym.register(
    id="Isaac-Can-Sorting-G1-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": pickplace_g1_env_cfg.CanSortingG1EnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-BlockStack-G1-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": pickplace_g1_env_cfg.BlockStackG1EnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-ObjectPlace-G1-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": pickplace_g1_env_cfg.ObjectPlacementG1EnvCfg,
    },
    disable_env_checker=True,
)
