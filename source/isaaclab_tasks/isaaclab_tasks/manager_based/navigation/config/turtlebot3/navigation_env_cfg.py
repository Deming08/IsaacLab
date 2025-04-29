# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
import isaaclab_tasks.manager_based.navigation.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.turtlebot3.flat_env_cfg import TurtleBotFlatEnvCfg

LOW_LEVEL_ENV_CFG = TurtleBotFlatEnvCfg()
from isaaclab.terrains.config.obstacle import OBSTACLE_TERRAINS_CFG  # isort: skip

@configclass
class TurtlebotSceneCfg(TurtleBotFlatEnvCfg):
    """Configuration for TurtleBot in flat terrain."""

    def __post_init__(self):
        """Post initialization."""
        # Call parent class initialization
        super().__post_init__()

        self.scene.terrain.terrain_type = "generator" # ["plane", "generator", "usd"]
        self.scene.terrain.terrain_generator = OBSTACLE_TERRAINS_CFG # [None, OBSTACLE_TERRAINS_CFG]
        self.scene.terrain.max_init_terrain_level = None

        self.scene.lidar_scan = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            update_period=1 / 60,      
            offset=RayCasterCfg.OffsetCfg(pos=(0., 0., 0.15)),   
            mesh_prim_paths=["/World/ground"],
            attach_yaw_only=True,          
            pattern_cfg=patterns.LidarPatternCfg(
                channels=1, 
                vertical_fov_range=[-0.0, 0.0], 
                horizontal_fov_range=[-180, 180], 
                horizontal_res=1.0      
            ),
            debug_vis=True,
            max_distance=3.5 
        )


@configclass
class EventCfg:
    """Configuration for events."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.3, 1.0), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )


@configclass
class ActionsCfg:
    """Action terms for the MDP."""

    cmd_vel_action = mdp.CmdVelActionCfg(
        asset_name="robot",  # 假設機器人名稱為 "robot"
        left_wheel_name="wheel_left_joint",
        right_wheel_name="wheel_right_joint",
        wheel_radius=0.033,  # TurtleBot3 輪子半徑
        wheel_base=0.287,     # TurtleBot3 輪距 burger:0.16 waffle 0.287
        max_lin_x=0.26,       # 最大線速度 burger:0.22 waffle 0.26
        max_ang_z=1.82,       # 最大角速度 burger:2.84 waffle 1.82
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_ang_vel = ObsTerm(func=mdp.base_lin_ang_vel, noise=Unoise(n_min=-0.05, n_max=0.05))
        
        pose_command = ObsTerm(func=mdp.get_commands, noise=Unoise(n_min=-0.05, n_max=0.05), params={"command_name": "pose_command"})

        lidar_scan = ObsTerm(
            func=mdp.lidar_scan, 
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(0.1, 3.5),
            params={"sensor_cfg": SceneEntityCfg("lidar_scan"),
                    "directions": ["all"],
                    "angle_range": 360,
                    "res": 5,
                    })

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    
    # for fast to reach goal
    position_tracking = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.8,
        params={"std": 2.0, "command_name": "pose_command"},
    )
    # for accurate adjust
    position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=1.0,
        params={"std": 0.15, "command_name": "pose_command"},
    )

    orientation_tracking = RewTerm(
        func=mdp.heading_command_error_abs,
        weight=-0.2,
        params={"command_name": "pose_command"},
    )

    goal_reached = RewTerm(
        func=mdp.goal_reached_reward,
        weight=1.0,
    )

    """penalize_lin_y = RewTerm(
        func=mdp.penalize_lin_y,
        weight=-0.5,
    )"""

    obstacle_avoidance_penalty = RewTerm(
        func=mdp.obstacle_avoidance_penalty,
        params={"lidar_num_rays": 72, "min_distance_threshold": 0.4, "penalty_scale": 0.5, "alpha": 3.0},
        weight=-0.5,
    )

    action_smoothness = RewTerm(
        func=mdp.action_smoothness_penalty,
        weight=-0.1,
    )

    forward_movement = RewTerm(
        func=mdp.forward_movement_reward,
        weight=0.1,
    )

@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    pose_command = mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        simple_heading=False,
        resampling_time_range=(30.0, 30.0),
        debug_vis=True,
        ranges=mdp.UniformPose2dCommandCfg.Ranges(pos_x=(-5.0, 5.0), pos_y=(-5.0, 5.0), heading=(-math.pi, math.pi)),
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class NavigationEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the navigation environment."""

    # environment settings
    #scene: SceneEntityCfg = LOW_LEVEL_ENV_CFG.scene
    scene: SceneEntityCfg = TurtlebotSceneCfg().scene
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    # mdp settings
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    
    def __post_init__(self):
        """Post initialization."""

        self.sim.dt = LOW_LEVEL_ENV_CFG.sim.dt
        self.sim.render_interval = LOW_LEVEL_ENV_CFG.decimation
        self.decimation = LOW_LEVEL_ENV_CFG.decimation * 10
        self.episode_length_s = self.commands.pose_command.resampling_time_range[1]


class NavigationEnvCfg_PLAY(NavigationEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.0
        # disable randomization for play
        self.observations.policy.enable_corruption = False
