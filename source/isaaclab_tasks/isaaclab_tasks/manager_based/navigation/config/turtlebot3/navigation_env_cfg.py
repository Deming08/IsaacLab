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
import torch

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
import isaaclab.sim as sim_utils


def randomize_obstacle_positions(env, env_ids, obstacle_name: str, pos_range_x: tuple, pos_range_y: tuple):
    """隨機設置障礙物的位置，並重置速度"""
    obstacle = env.scene[obstacle_name]
    num_envs = env.num_envs
    pos_x = torch.rand(num_envs, device=env.device) * (pos_range_x[1] - pos_range_x[0]) + pos_range_x[0]
    pos_y = torch.rand(num_envs, device=env.device) * (pos_range_y[1] - pos_range_y[0]) + pos_range_y[0]
    pos_z = torch.zeros(num_envs, device=env.device) + 0.25
    positions = torch.stack([pos_x, pos_y, pos_z], dim=1)

    """# 重置速度和角速度
    obstacle.set_linear_velocities(torch.zeros((num_envs, 3), device=env.device))
    obstacle.set_angular_velocities(torch.zeros((num_envs, 3), device=env.device))"""

    # 設置位置
    obstacle.set_world_poses(positions=positions)
    print(f"障礙物新位置: {positions}")

def randomize_obstacle_sizes(env, obstacle_name: str, size_range: tuple):
    """隨機設置障礙物的大小"""
    obstacle = env.scene[obstacle_name]
    num_envs = env.num_envs
    size = torch.rand(num_envs, device=env.device) * (size_range[1] - size_range[0]) + size_range[0]
    obstacle.set_scales(torch.stack([size, size, size], dim=1))


@configclass
class TurtlebotSceneCfg(TurtleBotFlatEnvCfg):
    """Configuration for TurtleBot in flat terrain."""

    def __post_init__(self):
        """Post initialization."""
        # Call parent class initialization
        super().__post_init__()

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

        """self.scene.obstacle = AssetBaseCfg(
            prim_path="/World/ground/Obstacle",
            spawn=sim_utils.CuboidCfg(
                size=(0.5, 0.5, 0.5),  # 初始大小
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),  # 黃色外觀
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=False,  # 動態剛體
                    disable_gravity=True,     # 禁用重力
                    rigid_body_enabled=True,  # 啟用剛體屬性
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True,  # 啟用碰撞屬性
                ),
                physics_material=sim_utils.RigidBodyMaterialCfg(),  # 確保有物理材質
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(1.5, 1.5, 0.25),  # 初始位置
            ),
        )"""


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

    """reset_obstacles = EventTerm(
        func=randomize_obstacle_positions,
        mode="reset",
        params={"obstacle_name": "obstacle", "pos_range_x": (-2.0, 2.0), "pos_range_y": (-2.0, 2.0)}
    )"""


@configclass
class ActionsCfg:
    """Action terms for the MDP."""

    """pre_trained_policy_action: mdp.PreTrainedPolicyActionCfg = mdp.PreTrainedPolicyActionCfg(
        asset_name="robot",
        policy_path=f"D:/IsaacLab/test_ws/trained_model/turtlebot3/velocity/2025-04-01_10-27-46/exported/policy.pt",
        low_level_decimation=4,
        low_level_actions=LOW_LEVEL_ENV_CFG.actions.wheel_velocities,
        low_level_observations=LOW_LEVEL_ENV_CFG.observations.policy,
    )"""
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
        #base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        #base_ang_vel = ObsTerm(func=mdp.base_ang_vel)

        base_lin_ang_vel = ObsTerm(func=mdp.base_lin_ang_vel, noise=Unoise(n_min=-0.05, n_max=0.05))
        
        #projected_gravity = ObsTerm(func=mdp.projected_gravity)
        #pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "pose_command"})
        pose_command = ObsTerm(func=mdp.get_commands, noise=Unoise(n_min=-0.05, n_max=0.05), params={"command_name": "pose_command"})

        ##joint_vel = ObsTerm(func=mdp.joint_vel_rel)
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

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-400.0)
    
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
    """base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )"""


@configclass
class NavigationEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the navigation environment."""

    # environment settings
    #scene: SceneEntityCfg = LOW_LEVEL_ENV_CFG.scene
    scene: SceneEntityCfg = TurtlebotSceneCfg().scene
    #scene: TurtlebotSceneCfg = TurtlebotSceneCfg(num_envs=2048, env_spacing=2.0)
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

        """if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = (
                self.actions.pre_trained_policy_action.low_level_decimation * self.sim.dt
            )
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt"""


class NavigationEnvCfg_PLAY(NavigationEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.0
        # disable randomization for play
        self.observations.policy.enable_corruption = False
