# Copyright (c) 2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym
import os

from . import base_openarm_env_cfg

"""Configurations for the OpenArm Base environments."""

gym.register(
    id="Isaac-Base-OpenArm-DexHand-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": base_openarm_env_cfg.BaseOpenArmEnvCfg,
    },
    disable_env_checker=True,
)