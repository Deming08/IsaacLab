# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Sub-package with environment wrappers for Isaac Lab Mimic."""

import gymnasium as gym

from .stack_g1_mimic_env import CubeStackG1MimicEnv
from .stack_g1_mimic_env_cfg import CubeStackG1MimicEnvCfg

from .cabinet_pour_g1_mimic_env import CabinetPourG1MimicEnv
from .cabinet_pour_g1_mimic_env_cfg import CabinetPourG1MimicEnvCfg

gym.register(
    id="Isaac-Stack-Cube-G1-Abs-Mimic-v0",
    entry_point="isaaclab_mimic.envs.g1_mimic:CubeStackG1MimicEnv",
    kwargs={
        "env_cfg_entry_point": stack_g1_mimic_env_cfg.CubeStackG1MimicEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Cabinet-Pour-G1-Abs-Mimic-v0",
    entry_point="isaaclab_mimic.envs.g1_mimic:CabinetPourG1MimicEnv",
    kwargs={
        "env_cfg_entry_point": cabinet_pour_g1_mimic_env_cfg.CabinetPourG1MimicEnvCfg,
    },
    disable_env_checker=True,
)
