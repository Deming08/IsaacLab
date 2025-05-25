# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg
from isaaclab_assets import TURTLEBOT3_WAFFLE_CFG  # 導入 TurtleBot3 Waffle 配置

from isaaclab.terrains.config.obstacle import OBSTACLE_TERRAINS_CFG  # isort: skip

##
# Reward Configuration
##

@configclass
class TurtleBotRewards(RewardsCfg):
    """Reward terms for the TurtleBot MDP."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": 0.05},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    # -- penalties
    orientation_penalty = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    wheel_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.25e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["wheel_left_joint", "wheel_right_joint"])},
    )
    wheel_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1.0e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["wheel_left_joint", "wheel_right_joint"])},
    )

    #lin_vel_z_l2 = None
    #ang_vel_xy_l2 = None
    feet_air_time = None
    undesired_contacts = None
    dof_pos_limits = None

##
# Environment Configuration
##

@configclass
class TurtleBotActionsCfg:
    """Action specifications for the MDP."""

    wheel_velocities = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["wheel_left_joint", "wheel_right_joint"], 
        scale=5.0,  # 速度縮放因子
    )

@configclass
class TurtleBotRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Configuration for TurtleBot in rough terrain."""

    rewards: TurtleBotRewards = TurtleBotRewards()
    actions: TurtleBotActionsCfg = TurtleBotActionsCfg()

    def __post_init__(self):
        """Post initialization."""
        # Call parent class initialization
        super().__post_init__()

        # Scene: Replace robot with TurtleBot and disable height scanner
        self.scene.robot = TURTLEBOT3_WAFFLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = OBSTACLE_TERRAINS_CFG
        self.scene.terrain.max_init_terrain_level = None

        self.scene.height_scanner = None  # TurtleBot 不需要高度掃描
        self.scene.contact_forces = None
        self.observations.policy.height_scan = None

        # Randomization: Adjust for wheeled robot
        self.events.push_robot = None  # No push for stability
        self.events.add_base_mass = None  # No mass randomization
        self.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)  # Wheels don't need position reset
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["base_link"]
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # Commands: Adjust velocity ranges for TurtleBot
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.3)  # Max forward speed ~0.3 m/s
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)  # No lateral movement
        self.commands.base_velocity.ranges.ang_vel_z = (-1.5, 1.5)  # Angular velocity range
    
        # Terminations: Remove base_contact, add flipped condition
        self.terminations.base_contact = None  # Base contact is normal for wheeled robot

   
@configclass
class TurtleBotRoughEnvCfg_PLAY(TurtleBotRoughEnvCfg):
    """Configuration for TurtleBot in rough terrain (play mode)."""

    def __post_init__(self):
        """Post initialization."""
        # Call parent class initialization
        super().__post_init__()

        # Smaller scene for playback
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 60.0
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # Fixed commands for playback
        self.commands.base_velocity.ranges.lin_vel_x = (0.2, 0.2)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        # Disable randomization for playback
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None