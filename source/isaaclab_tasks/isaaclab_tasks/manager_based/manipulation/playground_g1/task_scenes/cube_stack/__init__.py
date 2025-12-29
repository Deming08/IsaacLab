# Copyright (c) 2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym
import os

from . import stack_g1_env_cfg

"""Configurations for the object stack environments."""

gym.register(
    id="Isaac-Cube-Stack-G1-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": stack_g1_env_cfg.CubeStackG1EnvCfg,
    },
    disable_env_checker=True,
)