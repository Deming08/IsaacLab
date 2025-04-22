# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .rough_env_cfg import TurtleBotRoughEnvCfg  # 引用 rough_env_cfg.py 中的配置

##
# Environment Configuration
##

@configclass
class TurtleBotFlatEnvCfg(TurtleBotRoughEnvCfg):
    """Configuration for TurtleBot in flat terrain."""

    def __post_init__(self):
        """Post initialization."""
        # Call parent class initialization
        super().__post_init__()

        # Change terrain to flat
        #self.scene.terrain.terrain_type = "plane"
        #self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.scene.contact_forces = None
        self.observations.policy.height_scan = None
        self.observations.policy.projected_gravity = None
        self.curriculum.terrain_levels = None

        # Adjust rewards for flat terrain
        self.rewards.track_lin_vel_xy_exp.weight = 0.5
        self.rewards.track_ang_vel_z_exp.weight = 2.0

        self.rewards.lin_vel_z_l2.weight = -0.5  # Penalize vertical velocity
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.wheel_acc_l2.weight = -1.25e-7
        self.rewards.wheel_torques_l2.weight = -1.0e-6
        self.rewards.wheel_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=["wheel_left_joint", "wheel_right_joint"]
        )

        # Adjust commands for flat terrain
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-2.84, 2.84)
        self.commands.base_velocity.heading_command = True

        self.terminations.base_contact = None
        self.terminations.out_of_bounds = None
@configclass
class TurtleBotFlatEnvCfg_PLAY(TurtleBotFlatEnvCfg):
    """Configuration for TurtleBot in flat terrain (play mode)."""

    def __post_init__(self):
        """Post initialization."""
        # Call parent class initialization
        super().__post_init__()

        # Smaller scene for playback
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        # Disable randomization for playback
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None