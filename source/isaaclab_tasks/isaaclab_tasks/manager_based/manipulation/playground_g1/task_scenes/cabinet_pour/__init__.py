# Copyright (c) 2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym
import os

from . import cabinet_pour_g1_env_cfg

"""Configurations for the G1 environments."""

gym.register(
    id="Isaac-Cabinet-Pour-G1-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": cabinet_pour_g1_env_cfg.CabinetPourG1EnvCfg,
    },
    disable_env_checker=True,
)
