# Copyright (c) 2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym
import os

from . import cabinet_pour_g1_env_cfg, playground_g1_env_cfg

"""Configurations for the G1 environments."""

gym.register(
    id="Isaac-Cabinet-Pour-G1-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": cabinet_pour_g1_env_cfg.CabinetPourG1EnvCfg,
    },
    disable_env_checker=True,
)

# Mixture scene environment for gr00t inference
gym.register(
    id="Isaac-Playground-G1-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": playground_g1_env_cfg.PlaygroundG1EnvCfg,
    },
    disable_env_checker=True,
)