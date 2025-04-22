# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.classic.cartpole.mdp as mdp

##
# Pre-defined configs
##
from isaaclab_assets import TURTLEBOT3_WAFFLE_CFG

##
# Scene definition
##

@configclass
class TurtlebotSceneCfg(InteractiveSceneCfg):
    """Configuration for a TurtleBot3 scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # turtlebot
    robot: ArticulationCfg = TURTLEBOT3_WAFFLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


##
# MDP settings
##

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    wheel_velocities = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["wheel_left_joint", "wheel_right_joint"],  # 更新關節名稱
        scale=20.0,  # 速度縮放因子
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        root_pos = ObsTerm(func=mdp.root_pos_w)  # 機器人位置
        root_vel = ObsTerm(func=mdp.root_lin_vel_w)  # 線速度
        root_ang_vel = ObsTerm(func=mdp.root_ang_vel_w)  # 角速度
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)  # 輪子速度

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # reset
    reset_root_state = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (0.2, 0.2), "yaw": (-math.pi, math.pi)},
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.1, 0.1)},
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    reset_wheel_velocities = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["wheel_left_joint", "wheel_right_joint"]),  # 更新關節名稱
            "position_range": (0.0, 0.0),
            "velocity_range": (-0.5, 0.5),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Constant running reward
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    
    # (2) Encourage forward movement (x-direction velocity)
    forward_vel = RewTerm(
        func=mdp.root_lin_vel_w,
        weight=2.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "index": 0},  # x-axis velocity
    )
    
    # (3) Penalize lateral movement (y-direction velocity)
    lateral_vel = RewTerm(
        func=mdp.root_lin_vel_w,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot"), "index": 1},  # y-axis velocity
    )
    
    # (4) Penalize excessive angular velocity (keep stable)
    angular_vel = RewTerm(
        func=mdp.root_ang_vel_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
    # (5) Penalize excessive wheel velocity differences (encourage symmetry)
    wheel_vel_diff = RewTerm(
        func=mdp.joint_vel_diff_l1,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["wheel_left_joint", "wheel_right_joint"])},  # 更新關節名稱
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    # (2) Robot out of bounds
    out_of_bounds = DoneTerm(
        func=mdp.root_pos_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot"), "bounds": (-5.0, 5.0, -5.0, 5.0, 0.0, 1.0)},
    )


##
# Environment configuration
##

@configclass
class TurtlebotEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the TurtleBot3 environment."""

    # Scene settings
    scene: TurtlebotSceneCfg = TurtlebotSceneCfg(num_envs=4096, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation